; RUN: llc < %s -mtriple=arm64-apple-ios -aarch64-neon-syntax=apple -no-integrated-as -disable-post-ra | FileCheck %s

; rdar://9167275

define i32 @t1() nounwind ssp {
entry:
; CHECK-LABEL: t1:
; CHECK: mov {{w[0-9]+}}, 7
  %0 = tail call i32 asm "mov ${0:w}, 7", "=r"() nounwind
  ret i32 %0
}

define i64 @t2() nounwind ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: mov {{x[0-9]+}}, 7
  %0 = tail call i64 asm "mov $0, 7", "=r"() nounwind
  ret i64 %0
}

define i64 @t3() nounwind ssp {
entry:
; CHECK-LABEL: t3:
; CHECK: mov {{w[0-9]+}}, 7
  %0 = tail call i64 asm "mov ${0:w}, 7", "=r"() nounwind
  ret i64 %0
}

; rdar://9281206

define void @t4(i64 %op) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: mov x0, {{x[0-9]+}}; svc #0
  %0 = tail call i64 asm sideeffect "mov x0, $1; svc #0;", "=r,r,r,~{x0}"(i64 %op, i64 undef) nounwind
  ret void
}

; rdar://9394290

define float @t5(float %x) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: fadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %0 = tail call float asm "fadd ${0:s}, ${0:s}, ${0:s}", "=w,0"(float %x) nounwind
  ret float %0
}

; rdar://9553599

define zeroext i8 @t6(i8* %src) nounwind {
entry:
; CHECK-LABEL: t6:
; CHECK: ldtrb {{w[0-9]+}}, [{{x[0-9]+}}]
  %0 = tail call i8 asm "ldtrb ${0:w}, [$1]", "=r,r"(i8* %src) nounwind
  ret i8 %0
}

define void @t7(i8* %f, i32 %g) nounwind {
entry:
  %f.addr = alloca i8*, align 8
  store i8* %f, i8** %f.addr, align 8
  ; CHECK-LABEL: t7:
  ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}]
  call void asm "str ${1:w}, $0", "=*Q,r"(i8** %f.addr, i32 %g) nounwind
  ret void
}

; rdar://10258229
; ARM64TargetLowering::getRegForInlineAsmConstraint() should recognize 'v'
; registers.
define void @t8() nounwind ssp {
entry:
; CHECK-LABEL: t8:
; CHECK: stp {{d[0-9]+}}, {{d[0-9]+}}, [sp, #-16]
  tail call void asm sideeffect "nop", "~{v8}"() nounwind
  ret void
}

define i32 @constraint_I(i32 %i, i32 %j) nounwind {
entry:
  ; CHECK-LABEL: constraint_I:
  %0 = tail call i32 asm sideeffect "add ${0:w}, ${1:w}, $2", "=r,r,I"(i32 %i, i32 16773120) nounwind
  ; CHECK: add   {{w[0-9]+}}, {{w[0-9]+}}, #16773120
  %1 = tail call i32 asm sideeffect "add ${0:w}, ${1:w}, $2", "=r,r,I"(i32 %i, i32 4096) nounwind
  ; CHECK: add   {{w[0-9]+}}, {{w[0-9]+}}, #4096
  ret i32 %1
}

define i32 @constraint_J(i32 %i, i32 %j, i64 %k) nounwind {
entry:
  ; CHECK-LABEL: constraint_J:
  %0 = tail call i32 asm sideeffect "sub ${0:w}, ${1:w}, $2", "=r,r,J"(i32 %i, i32 -16773120) nounwind
  ; CHECK: sub   {{w[0-9]+}}, {{w[0-9]+}}, #-16773120
  %1 = tail call i32 asm sideeffect "sub ${0:w}, ${1:w}, $2", "=r,r,J"(i32 %i, i32 -1) nounwind
  ; CHECK: sub   {{w[0-9]+}}, {{w[0-9]+}}, #-1
  %2 = tail call i64 asm sideeffect "sub ${0:x}, ${1:x}, $2", "=r,r,J"(i64 %k, i32 -1) nounwind
  ; CHECK: sub   {{x[0-9]+}}, {{x[0-9]+}}, #-1
  %3 = tail call i64 asm sideeffect "sub ${0:x}, ${1:x}, $2", "=r,r,J"(i64 %k, i64 -1) nounwind
  ; CHECK: sub   {{x[0-9]+}}, {{x[0-9]+}}, #-1
  ret i32 %1
}

define i32 @constraint_KL(i32 %i, i32 %j) nounwind {
entry:
  ; CHECK-LABEL: constraint_KL:
  %0 = tail call i32 asm sideeffect "eor ${0:w}, ${1:w}, $2", "=r,r,K"(i32 %i, i32 255) nounwind
  ; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, #255
  %1 = tail call i32 asm sideeffect "eor ${0:w}, ${1:w}, $2", "=r,r,L"(i32 %i, i64 16711680) nounwind
  ; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, #16711680
  ret i32 %1
}

define i32 @constraint_MN(i32 %i, i32 %j) nounwind {
entry:
  ; CHECK-LABEL: constraint_MN:
  %0 = tail call i32 asm sideeffect "movk ${0:w}, $1", "=r,M"(i32 65535) nounwind
  ; CHECK: movk  {{w[0-9]+}}, #65535
  %1 = tail call i32 asm sideeffect "movz ${0:w}, $1", "=r,N"(i64 0) nounwind
  ; CHECK: movz  {{w[0-9]+}}, #0
  ret i32 %1
}

define void @t9() nounwind {
entry:
  ; CHECK-LABEL: t9:
  %data = alloca <2 x double>, align 16
  %0 = load <2 x double>, <2 x double>* %data, align 16
  call void asm sideeffect "mov.2d v4, $0\0A", "w,~{v4}"(<2 x double> %0) nounwind
  ; CHECK: mov.2d v4, {{v[0-9]+}}
  ret void
}

define void @t10() nounwind {
entry:
  ; CHECK-LABEL: t10:
  %data = alloca <2 x float>, align 8
  %a = alloca [2 x float], align 4
  %arraydecay = getelementptr inbounds [2 x float], [2 x float]* %a, i32 0, i32 0
  %0 = load <2 x float>, <2 x float>* %data, align 8
  call void asm sideeffect "ldr ${1:q}, [$0]\0A", "r,w"(float* %arraydecay, <2 x float> %0) nounwind
  ; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}]
  call void asm sideeffect "ldr ${1:d}, [$0]\0A", "r,w"(float* %arraydecay, <2 x float> %0) nounwind
  ; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}]
  call void asm sideeffect "ldr ${1:s}, [$0]\0A", "r,w"(float* %arraydecay, <2 x float> %0) nounwind
  ; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}]
  call void asm sideeffect "ldr ${1:h}, [$0]\0A", "r,w"(float* %arraydecay, <2 x float> %0) nounwind
  ; CHECK: ldr {{h[0-9]+}}, [{{x[0-9]+}}]
  call void asm sideeffect "ldr ${1:b}, [$0]\0A", "r,w"(float* %arraydecay, <2 x float> %0) nounwind
  ; CHECK: ldr {{b[0-9]+}}, [{{x[0-9]+}}]
  ret void
}

define void @t11() nounwind {
entry:
  ; CHECK-LABEL: t11:
  %a = alloca i32, align 4
  %0 = load i32, i32* %a, align 4
  call void asm sideeffect "mov ${1:x}, ${0:x}\0A", "r,i"(i32 %0, i32 0) nounwind
  ; CHECK: mov xzr, {{x[0-9]+}}
  %1 = load i32, i32* %a, align 4
  call void asm sideeffect "mov ${1:w}, ${0:w}\0A", "r,i"(i32 %1, i32 0) nounwind
  ; CHECK: mov wzr, {{w[0-9]+}}
  ret void
}

define void @t12() nounwind {
entry:
  ; CHECK-LABEL: t12:
  %data = alloca <4 x float>, align 16
  %0 = load <4 x float>, <4 x float>* %data, align 16
  call void asm sideeffect "mov.2d v4, $0\0A", "x,~{v4}"(<4 x float> %0) nounwind
  ; CHECK: mov.2d v4, {{v([0-9])|(1[0-5])}}
  ret void
}

define void @t13() nounwind {
entry:
  ; CHECK-LABEL: t13:
  tail call void asm sideeffect "mov x4, $0\0A", "N"(i64 1311673391471656960) nounwind
  ; CHECK: mov x4, #1311673391471656960
  tail call void asm sideeffect "mov x4, $0\0A", "N"(i64 -4662) nounwind
  ; CHECK: mov x4, #-4662
  tail call void asm sideeffect "mov x4, $0\0A", "N"(i64 4660) nounwind
  ; CHECK: mov x4, #4660
  call void asm sideeffect "mov x4, $0\0A", "N"(i64 -71777214294589696) nounwind
  ; CHECK: mov x4, #-71777214294589696
  ret void
}

define void @t14() nounwind {
entry:
  ; CHECK-LABEL: t14:
  tail call void asm sideeffect "mov w4, $0\0A", "M"(i32 305397760) nounwind
  ; CHECK: mov w4, #305397760
  tail call void asm sideeffect "mov w4, $0\0A", "M"(i32 -4662) nounwind
  ; CHECK: mov w4, #4294962634
  tail call void asm sideeffect "mov w4, $0\0A", "M"(i32 4660) nounwind
  ; CHECK: mov w4, #4660
  call void asm sideeffect "mov w4, $0\0A", "M"(i32 -16711936) nounwind
  ; CHECK: mov w4, #4278255360
  ret void
}

define void @t15() nounwind {
entry:
  %0 = tail call double asm sideeffect "fmov $0, d8", "=r"() nounwind
  ; CHECK: fmov {{x[0-9]+}}, d8
  ret void
}

; rdar://problem/14285178

define void @test_zero_reg(i32* %addr) {
; CHECK-LABEL: test_zero_reg:

  tail call void asm sideeffect "USE($0)", "z"(i32 0) nounwind
; CHECK: USE(xzr)

  tail call void asm sideeffect "USE(${0:w})", "zr"(i32 0)
; CHECK: USE(wzr)

  tail call void asm sideeffect "USE(${0:w})", "zr"(i32 1)
; CHECK: orr [[VAL1:w[0-9]+]], wzr, #0x1
; CHECK: USE([[VAL1]])

  tail call void asm sideeffect "USE($0), USE($1)", "z,z"(i32 0, i32 0) nounwind
; CHECK: USE(xzr), USE(xzr)

  tail call void asm sideeffect "USE($0), USE(${1:w})", "z,z"(i32 0, i32 0) nounwind
; CHECK: USE(xzr), USE(wzr)

  ret void
}
