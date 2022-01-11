; RUN: not llc -mtriple=x86_64-linux-gnu -o - %s 2>&1 | FileCheck %s

; Test that the blockaddress with X, i, or s constraint is printed as an
; immediate (.Ltmp0).
; Test that blockaddress with n constraint is an error.
define void @test1() {
; CHECK: error: constraint 'n' expects an integer constant expression
; CHECK-LABEL: test1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:  .Ltmp0: # Block address taken
; CHECK-NEXT:  # %bb.1: # %b
; CHECK-NEXT:    #APP
; CHECK-NEXT:    # .Ltmp0 .Ltmp0 .Ltmp0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    retq
entry:
  br label %b
b:
  call void asm "# $0 $1 $2", "X,i,s"(i8* blockaddress(@test1, %b), i8* blockaddress(@test1, %b), i8* blockaddress(@test1, %b))
  call void asm "# $0", "n"(i8* blockaddress(@test1, %b))
  ret void
}
