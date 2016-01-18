; RUN: llvm-as -function-summary %s -o %t.o
; RUN: llvm-as -function-summary %p/Inputs/thinlto.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    -shared %t.o %t2.o -o %t3
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED
; RUN: not test -e %t3

; COMBINED: <MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: </MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <FUNCTION_SUMMARY_BLOCK
; COMBINED-NEXT: <COMBINED_ENTRY
; COMBINED-NEXT: <COMBINED_ENTRY
; COMBINED-NEXT: </FUNCTION_SUMMARY_BLOCK
; COMBINED-NEXT: <VALUE_SYMTAB
; COMBINED-NEXT: <COMBINED_FNENTRY {{.*}} record string = '{{f|g}}'
; COMBINED-NEXT: <COMBINED_FNENTRY {{.*}} record string = '{{f|g}}'
; COMBINED-NEXT: </VALUE_SYMTAB

define void @f() {
entry:
  ret void
}
