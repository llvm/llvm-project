; RUN: llc < %s -mtriple=bpfel | FileCheck %s
; RUN: llc < %s -mtriple=bpfeb | FileCheck %s

; CHECK-LABEL: fence_singlethread:
; CHECK-COUNT-4: #MEMBARRIER
; CHECK-NEXT:    exit
define void @fence_singlethread() nounwind {
entry:
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") release
  fence syncscope("singlethread") acq_rel
  fence syncscope("singlethread") seq_cst
  ret void
}
