; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

define void @vld(i8* %p, i64 %i) nounwind {
; CHECK-LABEL: vld:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %lvl = tail call i64 asm sideeffect "lea $0, 256", "=r"() nounwind
  tail call void asm sideeffect "lvl $0", "r"(i64 %lvl) nounwind
  tail call <256 x double> asm sideeffect "vld $0, $2, $1", "=v,r,r"(i8* %p, i64 %i) nounwind
  ret void
}

define void @vldvst(i8* %p, i64 %i) nounwind {
; CHECK-LABEL: vldvst:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %lvl = tail call i64 asm sideeffect "lea $0, 256", "=r"() nounwind
  tail call void asm sideeffect "lvl $0", "r"(i64 %lvl) nounwind
  %1 = tail call <256 x double> asm sideeffect "vld $0, $2, $1", "=v,r,r"(i8* %p, i64 %i) nounwind
  tail call void asm sideeffect "vst $0, $2, $1", "v,r,r"(<256 x double> %1, i8* %p, i64 %i) nounwind
  ret void
}

define void @vld2vst2(i8* %p, i64 %i) nounwind {
; CHECK-LABEL: vld2vst2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vld %v1, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v1, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %lvl = tail call i64 asm sideeffect "lea $0, 256", "=r"() nounwind
  tail call void asm sideeffect "lvl $0", "r"(i64 %lvl) nounwind
  %1 = tail call <256 x double> asm sideeffect "vld $0, $2, $1", "=v,r,r"(i8* %p, i64 %i) nounwind
  %2 = tail call <256 x double> asm sideeffect "vld $0, $2, $1", "=v,r,r"(i8* %p, i64 %i) nounwind
  tail call void asm sideeffect "vst $0, $2, $1", "v,r,r"(<256 x double> %1, i8* %p, i64 %i) nounwind
  tail call void asm sideeffect "vst $0, $2, $1", "v,r,r"(<256 x double> %2, i8* %p, i64 %i) nounwind
  ret void
}
