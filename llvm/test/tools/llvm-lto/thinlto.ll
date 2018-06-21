; Test combined function index generation for ThinLTO via llvm-lto.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED
; RUN: not test -e %t3

; COMBINED: <MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: </MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT: <VERSION
; COMBINED-NEXT: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; COMBINED-NEXT: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: </GLOBALVAL_SUMMARY_BLOCK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f() {
entry:
  ret void
}
