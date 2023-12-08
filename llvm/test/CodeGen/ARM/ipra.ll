; RUN: llc -mtriple armv7a--none-eabi < %s              | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLED
; RUN: llc -mtriple armv7a--none-eabi < %s -enable-ipra | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLED

define void @leaf()  {
entry:
  tail call void asm sideeffect "", ""()
  ret void
}

define void @leaf_r3() {
entry:
  tail call void asm sideeffect "", "~{r3}"()
  ret void
}

define void @leaf_r4() {
entry:
  tail call void asm sideeffect "", "~{r4}"()
  ret void
}

define void @leaf_s0() {
entry:
  tail call void asm sideeffect "", "~{s0}"()
  ret void
}

define void @leaf_d0() {
entry:
  tail call void asm sideeffect "", "~{d0}"()
  ret void
}

; r3 is normally caller-saved, but with IPRA we can see that it isn't used in
; the callee, so can leave a live value in it.
define void @test_r3_presrved() {
; CHECK-LABEL: test_r3_presrved:
entry:
; CHECK: ASM1: r3
; DISABLED: mov [[TEMP:r[0-9]+]], r3
; ENABLED-NOT: r3
; CHECK: bl      leaf
; DISABLED: mov r3, [[TEMP]]
; ENABLED-NOT: r3
; CHECK: ASM2: r3
  %a = tail call i32 asm sideeffect "// ASM1: $0", "={r3},0"(i32 undef)
  tail call void @leaf()
  %b = tail call i32 asm sideeffect "// ASM2: $0", "={r3},0"(i32 %a)
  ret void
}

; Same as above, but r3 is clobbered in the callee, so it is clobbered by the
; call as normal.
define void @test_r3_clobbered() {
; CHECK-LABEL: test_r3_clobbered:
entry:
; CHECK: ASM1: r3
; CHECK: mov [[TEMP:r[0-9]+]], r3
; CHECK: bl      leaf
; CHECK: mov r3, [[TEMP]]
; CHECK: ASM2: r3
  %a = tail call i32 asm sideeffect "// ASM1: $0", "={r3},0"(i32 undef)
  tail call void @leaf_r3()
  %b = tail call i32 asm sideeffect "// ASM2: $0", "={r3},0"(i32 %a)
  ret void
}

; r4 is a callee-saved register, so IPRA has no effect.
define void @test_r4_preserved() {
; CHECK-LABEL: test_r4_preserved:
entry:
; CHECK: ASM1: r4
; CHECK-NOT: r4
; CHECK: bl      leaf
; CHECK-NOT: r4
; CHECK: ASM2: r4
  %a = tail call i32 asm sideeffect "// ASM1: $0", "={r4},0"(i32 undef)
  tail call void @leaf()
  %b = tail call i32 asm sideeffect "// ASM2: $0", "={r4},0"(i32 %a)
  ret void
}
define void @test_r4_clobbered() {
; CHECK-LABEL: test_r4_clobbered:
entry:
; CHECK: ASM1: r4
; CHECK-NOT: r4
; CHECK: bl      leaf_r4
; CHECK-NOT: r4
; CHECK: ASM2: r4
  %a = tail call i32 asm sideeffect "// ASM1: $0", "={r4},0"(i32 undef)
  tail call void @leaf_r4()
  %b = tail call i32 asm sideeffect "// ASM2: $0", "={r4},0"(i32 %a)
  ret void
}

; r12 is the intra-call scratch register, so we have to assume it is clobbered
; even if we can see that the callee does not touch it.
define void @test_r12() {
; CHECK-LABEL: test_r12:
entry:
; CHECK: ASM1: r12
; CHECK: mov [[TEMP:r[0-9]+]], r12
; CHECK: bl      leaf
; CHECK: mov r12, [[TEMP]]
; CHECK: ASM2: r12
  %a = tail call i32 asm sideeffect "// ASM1: $0", "={r12},0"(i32 undef)
  tail call void @leaf()
  %b = tail call i32 asm sideeffect "// ASM2: $0", "={r12},0"(i32 %a)
  ret void
}

; s0 and d0 are caller-saved, IPRA allows us to keep them live in the caller if
; the callee doesn't modify them.
define void @test_s0_preserved() {
; CHECK-LABEL: test_s0_preserved:
entry:
; CHECK: ASM1: s0
; DISABLED: vmov.f32 [[TEMP:s[0-9]+]], s0
; ENABLED-NOT: s0
; CHECK: bl      leaf
; DISABLED: vmov.f32 s0, [[TEMP]]
; ENABLED-NOT: s0
; CHECK: ASM2: s0
  %a = tail call float asm sideeffect "// ASM1: $0", "={s0},0"(float undef)
  tail call void @leaf()
  %b = tail call float asm sideeffect "// ASM2: $0", "={s0},0"(float %a)
  ret void
}

define void @test_s0_clobbered() {
; CHECK-LABEL: test_s0_clobbered:
entry:
; CHECK: ASM1: s0
; CHECK: vmov.f32 [[TEMP:s[0-9]+]], s0
; CHECK: bl      leaf_s0
; CHECK: vmov.f32 s0, [[TEMP]]
; CHECK: ASM2: s0
  %a = tail call float asm sideeffect "// ASM1: $0", "={s0},0"(float undef)
  tail call void @leaf_s0()
  %b = tail call float asm sideeffect "// ASM2: $0", "={s0},0"(float %a)
  ret void
}

define void @test_d0_preserved() {
; CHECK-LABEL: test_d0_preserved:
entry:
; CHECK: ASM1: d0
; DISABLED: vmov.f64 [[TEMP:d[0-9]+]], d0
; ENABLED-NOT: d0
; CHECK: bl      leaf
; DISABLED: vmov.f64 d0, [[TEMP]]
; ENABLED-NOT: d0
; CHECK: ASM2: d0
  %a = tail call double asm sideeffect "// ASM1: $0", "={d0},0"(double undef)
  tail call void @leaf()
  %b = tail call double asm sideeffect "// ASM2: $0", "={d0},0"(double %a)
  ret void
}

define void @test_d0_clobbered() {
; CHECK-LABEL: test_d0_clobbered:
entry:
; CHECK: ASM1: d0
; CHECK: vmov.f64 [[TEMP:d[0-9]+]], d0
; CHECK: bl      leaf_d0
; CHECK: vmov.f64 d0, [[TEMP]]
; CHECK: ASM2: d0
  %a = tail call double asm sideeffect "// ASM1: $0", "={d0},0"(double undef)
  tail call void @leaf_d0()
  %b = tail call double asm sideeffect "// ASM2: $0", "={d0},0"(double %a)
  ret void
}

; s0 and d0 overlap, so clobbering one in the callee prevents the other from
; being kept live across the call.
define void @test_s0_clobber_d0() {
; CHECK-LABEL: test_s0_clobber_d0:
entry:
; CHECK: ASM1: s0
; CHECK: vmov.f32 [[TEMP:s[0-9]+]], s0
; CHECK: bl      leaf_d0
; CHECK: vmov.f32 s0, [[TEMP]]
; CHECK: ASM2: s0
  %a = tail call float asm sideeffect "// ASM1: $0", "={s0},0"(float undef)
  tail call void @leaf_d0()
  %b = tail call float asm sideeffect "// ASM2: $0", "={s0},0"(float %a)
  ret void
}

define void @test_d0_clobber_s0() {
; CHECK-LABEL: test_d0_clobber_s0:
entry:
; CHECK: ASM1: d0
; CHECK: vmov.f64 [[TEMP:d[0-9]+]], d0
; CHECK: bl      leaf_s0
; CHECK: vmov.f64 d0, [[TEMP]]
; CHECK: ASM2: d0
  %a = tail call double asm sideeffect "// ASM1: $0", "={d0},0"(double undef)
  tail call void @leaf_s0()
  %b = tail call double asm sideeffect "// ASM2: $0", "={d0},0"(double %a)
  ret void
}
