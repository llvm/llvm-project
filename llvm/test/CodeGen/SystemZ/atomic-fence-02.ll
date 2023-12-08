; Serialization is emitted only for fence seq_cst.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @test() {
; CHECK: #MEMBARRIER
  fence acquire
; CHECK: #MEMBARRIER
  fence release
; CHECK: #MEMBARRIER
  fence acq_rel
  ret void
}
