; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Regression: poison constants are also reported by LLVMIsUndef()

define i32 @test_poison() {
entry:
  ret i32 poison
}
