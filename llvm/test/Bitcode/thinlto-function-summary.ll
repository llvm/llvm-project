; RUN: opt -name-anon-globals -module-summary < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; RUN: opt -passes=name-anon-globals -module-summary < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; Check for summary block/records.

; BC: <SOURCE_FILENAME
; "h"
; BC-NEXT: <GLOBALVAR {{.*}} op0=0 op1=1
; "foo"
; BC-NEXT: <FUNCTION op0=1 op1=3
; "bar"
; BC-NEXT: <FUNCTION op0=4 op1=3
; "anon.[32 chars].0"
; BC-NEXT: <FUNCTION op0=7 op1=39
; "variadic"
; BC-NEXT: <FUNCTION op0=46 op1=8
; "f"
; BC-NEXT: <ALIAS op0=54 op1=1
; BC: <GLOBALVAL_SUMMARY_BLOCK
; BC-NEXT: <VERSION
; BC-NEXT: <PERMODULE {{.*}} op0=1 op1=0
; BC-NEXT: <PERMODULE {{.*}} op0=2 op1=0
; BC-NEXT: <PERMODULE {{.*}} op0=3 op1=7
; BC-NEXT: <PERMODULE {{.*}} op0=4 op1=16
; BC-NEXT: <ALIAS {{.*}} op0=5 op1=0 op2=3
; BC-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; BC: <STRTAB_BLOCK
; BC-NEXT: blob data = 'hfoobaranon.{{................................}}.0variadicf{{.*}}'


; RUN: opt -name-anon-globals -module-summary < %s | llvm-dis | FileCheck %s
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

; FIXME: Anonymous function and alias not currently in summary until
; follow on fixes to rename anonymous globals and emit alias summary
; entries are committed.
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

define i32 @variadic(...) {
    ret i32 42
}
