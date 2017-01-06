; RUN: opt -name-anon-globals -module-summary < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; RUN: opt -passes=name-anon-globals -module-summary < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; Check for summary block/records.

; Check the value ids in the summary entries against the
; same in the ValueSumbolTable, to ensure the ordering is stable.
; Also check the linkage field on the summary entries.
; BC: <GLOBALVAL_SUMMARY_BLOCK
; BC-NEXT: <VERSION
; BC-NEXT: <PERMODULE {{.*}} op0=1 op1=0
; BC-NEXT: <PERMODULE {{.*}} op0=2 op1=0
; BC-NEXT: <PERMODULE {{.*}} op0=3 op1=7
; BC-NEXT: <PERMODULE {{.*}} op0=4 op1=16
; BC-NEXT: <ALIAS {{.*}} op0=5 op1=0 op2=3
; BC-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; BC-NEXT: <VALUE_SYMTAB
; BC-NEXT: <FNENTRY {{.*}} op0=4 {{.*}}> record string = 'variadic'
; BC-NEXT: <FNENTRY {{.*}} op0=1 {{.*}}> record string = 'foo'
; BC-NEXT: <FNENTRY {{.*}} op0=2 {{.*}}> record string = 'bar'
; BC-NEXT: <FNENTRY {{.*}} op0=5 {{.*}}> record string = 'f'
; BC-NEXT: <ENTRY {{.*}} record string = 'h'
; BC-NEXT: <FNENTRY {{.*}} op0=3 {{.*}}> record string = 'anon.


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
