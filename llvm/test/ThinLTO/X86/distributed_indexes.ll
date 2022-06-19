; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/distributed_indexes.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=distributedindexes -thinlto-index %t.index.bc %t1.bc %t2.bc
; RUN: llvm-bcanalyzer -dump %t1.bc.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t2.bc.thinlto.bc | FileCheck %s --check-prefix=BACKEND2

; The backend index for this module contains summaries from itself and
; Inputs/distributed_indexes.ll, as it imports from the latter.
; We should import @g and alias @analias. While we don't import the aliasee
; directly (and therefore don't have a third COMBINED record from module
; id 1), we will have a VALUE_GUID for it (hence the 4 VALUE_GUID entries).
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}distributed_indexes.ll.tmp{{.*}}.bc'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}distributed_indexes.ll.tmp{{.*}}.bc'
; BACKEND1-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND1-NEXT: <VERSION
; BACKEND1-DAG: <VALUE_GUID op0={{.*}}
; BACKEND1-DAG: <VALUE_GUID op0={{.*}}
; BACKEND1-DAG: <VALUE_GUID op0={{.*}}
; BACKEND1-DAG: <VALUE_GUID op0={{.*}}
; BACKEND1-DAG: <VALUE_GUID op0={{.*}}
; BACKEND1-NEXT: <COMBINED {{.*}} op1=0
; BACKEND1-NEXT: <COMBINED {{.*}} op1=0
; BACKEND1-NEXT: <COMBINED {{.*}} op1=1
; BACKEND1-NEXT: <COMBINED_ALIAS {{.*}} op1=1
; BACKEND1-NEXT: <BLOCK_COUNT op0=4/>
; BACKEND1-NEXT: </GLOBALVAL_SUMMARY_BLOCK

; The backend index for Input/distributed_indexes.ll contains summaries from
; itself only, as it does not import anything.
; BACKEND2: <MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '{{.*}}distributed_indexes.ll.tmp2.bc'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-DAG: <VALUE_GUID op0={{.*}}
; BACKEND2-DAG: <VALUE_GUID op0={{.*}}
; BACKEND2-DAG: <VALUE_GUID op0={{.*}}
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: <COMBINED_ALIAS
; BACKEND2-NEXT: <BLOCK_COUNT op0=4/>
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

; Make sure that when the alias is imported as a copy of the aliasee, but the
; aliasee is not being imported by itself, that we can still print the summary.
; The aliasee should be "null".
; RUN: llvm-dis %t1.bc.thinlto.bc -o - | FileCheck %s --check-prefix=DIS
; DIS: aliasee: null

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

declare void @g(...)
declare void @analias(...)

define void @f() {
entry:
  call void (...) @g()
  call void (...) @analias()
  ret void
}

; Ensure we don't get a COMBINED_ORIGINAL_NAME record in the distributed index.
; The BACKEND1-NEXT checks would fail if we did.
define internal void @x() {
  ret void
}
