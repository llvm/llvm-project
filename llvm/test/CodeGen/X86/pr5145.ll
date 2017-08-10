; RUN: llc -mtriple=x86_64-- < %s | FileCheck %s
@sc8 = external global i8

define void @atomic_maxmin_i8() {
; CHECK: atomic_maxmin_i8
  %1 = atomicrmw max  i8* @sc8, i8 5 acquire
; CHECK: [[LABEL1:\.?LBB[0-9]+_[0-9]+]]:
; CHECK: cmpb
; CHECK: jg
; CHECK: lock cmpxchgb
; CHECK: jne [[LABEL1]]
  %2 = atomicrmw min  i8* @sc8, i8 6 acquire
; CHECK: [[LABEL3:\.?LBB[0-9]+_[0-9]+]]:
; CHECK: cmpb
; CHECK: jl
; CHECK: lock cmpxchgb
; CHECK: jne [[LABEL3]]
  %3 = atomicrmw umax i8* @sc8, i8 7 acquire
; CHECK: [[LABEL5:\.?LBB[0-9]+_[0-9]+]]:
; CHECK: cmpb
; CHECK: ja
; CHECK: lock cmpxchgb
; CHECK: jne [[LABEL5]]
  %4 = atomicrmw umin i8* @sc8, i8 8 acquire
; CHECK: [[LABEL7:\.?LBB[0-9]+_[0-9]+]]:
; CHECK: cmpb
; CHECK: jb
; CHECK: lock cmpxchgb
; CHECK: jne [[LABEL7]]
  ret void
}
