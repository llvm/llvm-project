; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

@bb = constant [1 x ptr] [ptr blockaddress(@main, %l2)]

define void @main() {
entry:
  br label %l1

l1:                                               ; preds = %l2, %entry
  %a = zext i1 false to i32
  br label %l2

l2:                                               ; preds = %l1
  %b = zext i1 false to i32
  br label %l1
}

; It is correct for either l1 or l2 to be removed.
; If l2 is removed, the message should be "Address of block that was removed by CodeGen"
; If l1 is removed, it should be "Block address taken."
; CHECK: .Ltmp0:                                 # {{Address of block that was removed by CodeGen|Block address taken}}
; CHECK: .quad	.Ltmp0
