; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

@bb = constant [1 x i8*] [i8* blockaddress(@main, %l2)]

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

; CHECK: .Ltmp0:                                 # Address of block that was removed by CodeGen
; CHECK: .quad	.Ltmp0
