; RUN: llvm-as -function-summary < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; Check for function summary block/records.

; BC: <FUNCTION_SUMMARY_BLOCK
; BC-NEXT: <PERMODULE_ENTRY
; BC-NEXT: <PERMODULE_ENTRY
; BC-NEXT: <PERMODULE_ENTRY
; BC-NEXT: </FUNCTION_SUMMARY_BLOCK

; RUN: llvm-as -function-summary < %s | llvm-dis | FileCheck %s
; Check that this round-trips correctly.

; ModuleID = '<stdin>'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define i32 @foo()

; Function Attrs: nounwind uwtable
define i32 @foo() #0 {
entry:
  ret i32 1
}

; CHECK: define i32 @bar(i32 %x)

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %x) #0 {
entry:
  ret i32 %x
}

; Check an anonymous function as well, since in that case only the alias
; ends up in the value symbol table and having a summary.
@f = alias void (), void ()* @0   ; <void ()*> [#uses=0]
@h = external global void ()*     ; <void ()*> [#uses=0]

define internal void @0() nounwind {
entry:
  store void()* @0, void()** @h
        br label %return

return:         ; preds = %entry
        ret void
}
