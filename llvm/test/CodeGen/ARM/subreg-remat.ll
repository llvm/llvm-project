; RUN: llc < %s -relocation-model=pic -disable-fp-elim -mcpu=cortex-a8 -pre-RA-sched=source -no-integrated-as | FileCheck %s
target triple = "thumbv7-apple-ios"
; <rdar://problem/10032939>
;
; The vector %v2 is built like this:
;
;   %6:ssub_1 = ...
;   %6:ssub_0 = VLDRS <cp#0>, 0, pred:14, pred:%noreg; mem:LD4[ConstantPool] DPR_VFP2:%6
;
; When %6 spills, the VLDRS constant pool load cannot be rematerialized
; since it implicitly reads the ssub_1 sub-register.
;
; CHECK: f1
; CHECK: vmov    d0, r0, r0
; CHECK: vldr s1, LCPI
; The vector must be spilled:
; CHECK: vstr d0,
; CHECK: asm clobber d0
; And reloaded after the asm:
; CHECK: vldr [[D16:d[0-9]+]],
; CHECK: vstr [[D16]], [r1]
define void @f1(float %x, <2 x float>* %p) {
  %v1 = insertelement <2 x float> undef, float %x, i32 0
  %v2 = insertelement <2 x float> %v1, float 0x400921FB60000000, i32 1
  %y = call double asm sideeffect "asm clobber $0", "=w,0,~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15},~{d16},~{d17},~{d18},~{d19},~{d20},~{d21},~{d22},~{d23},~{d24},~{d25},~{d26},~{d27},~{d28},~{d29},~{d30},~{d31}"(<2 x float> %v2) nounwind
  store <2 x float> %v2, <2 x float>* %p, align 8
  ret void
}

; On the other hand, when the partial redef doesn't read the full register
; because the bits are undef, we should rematerialize.  The vector is now built
; like this:
;
;   %2:ssub_0 = VLDRS <cp#0>, 0, pred:14, pred:%noreg, implicit-def %2; mem:LD4[ConstantPool]
;
; The extra <imp-def> operand indicates that the instruction fully defines the
; virtual register.  It doesn't read the old value.
;
; CHECK: f2
; CHECK: vldr s0, LCPI
; The vector must not be spilled:
; CHECK-NOT: vstr
; CHECK: asm clobber d0
; But instead rematerialize after the asm:
; CHECK: vldr [[S0:s[0-9]+]], LCPI
; CHECK: vstr [[D0:d[0-9]+]], [r0]
define void @f2(<2 x float>* %p) {
  %v2 = insertelement <2 x float> undef, float 0x400921FB60000000, i32 0
  %y = call double asm sideeffect "asm clobber $0", "=w,0,~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15},~{d16},~{d17},~{d18},~{d19},~{d20},~{d21},~{d22},~{d23},~{d24},~{d25},~{d26},~{d27},~{d28},~{d29},~{d30},~{d31}"(<2 x float> %v2) nounwind
  store <2 x float> %v2, <2 x float>* %p, align 8
  ret void
}
