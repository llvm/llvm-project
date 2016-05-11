; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/distributed_indexes.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=distributedindexes -thinlto-index %t.index.bc %t1.bc %t2.bc
; RUN: llvm-bcanalyzer -dump %t1.bc.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t2.bc.thinlto.bc | FileCheck %s --check-prefix=BACKEND2

; The backend index for this module contains summaries from itself and
; Inputs/distributed_indexes.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}distributed_indexes.ll.tmp{{.*}}.bc'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}distributed_indexes.ll.tmp{{.*}}.bc'
; BACKEND1-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND1-NEXT: <VERSION
; BACKEND1-NEXT: <COMBINED
; BACKEND1-NEXT: <COMBINED
; BACKEND1-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; BACKEND1-NEXT: <VALUE_SYMTAB
; Check that the format is: op0=valueid, op1=offset, op2=funcguid,
; where funcguid is the lower 64 bits of the function name MD5.
; BACKEND1-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1-NEXT: </VALUE_SYMTAB

; The backend index for Input/distributed_indexes.ll contains summaries from
; itself only, as it does not import anything.
; BACKEND2: <MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '{{.*}}distributed_indexes.ll.tmp2.bc'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VALUE_SYMTAB
; Check that the format is: op0=valueid, op1=offset, op2=funcguid,
; where funcguid is the lower 64 bits of the function name MD5.
; BACKEND2-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: </VALUE_SYMTAB

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
