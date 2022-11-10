
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # declare the result list
        final = ListNode()
        temp = final
        while l1 != None or l2 != None:
            if l1 == None:
                temp.next = l2
                break
            if l2 == None:
                temp.next = l1
                break
            if l1.val <= l2.val:
                temp.next = ListNode(l1.val)
                l1 = l1.next
            else:
                temp.next = ListNode(l2.val)
                l2 = l2.next
            temp = temp.next
        return final.next

v3=ListNode(3)
v2=ListNode(2,v3)
v1=ListNode(1,v2)

v4=ListNode(4)
v5=ListNode(2,v4)
v6=ListNode(1,v5)



a = Solution()

x= a.mergeTwoLists(v1,v6)

print(0)