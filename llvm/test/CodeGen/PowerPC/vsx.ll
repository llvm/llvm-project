; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck \
; RUN:     -check-prefix=CHECK-REG %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx -fast-isel -O0 \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck \
; RUN:     -check-prefix=CHECK-FISL %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr8 \
; RUN:     -mtriple=powerpc64le-unknown-linux-gnu -mattr=+vsx \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck \
; RUN:     -check-prefix=CHECK-LE %s

define double @test1(double %a, double %b) {
entry:
  %v = fmul double %a, %b
  ret double %v

; CHECK-LABEL: @test1
; CHECK: xsmuldp f1, f1, f2
; CHECK: blr

; CHECK-LE-LABEL: @test1
; CHECK-LE: xsmuldp f1, f1, f2
; CHECK-LE: blr
}

define double @test2(double %a, double %b) {
entry:
  %v = fdiv double %a, %b
  ret double %v

; CHECK-LABEL: @test2
; CHECK: xsdivdp f1, f1, f2
; CHECK: blr

; CHECK-LE-LABEL: @test2
; CHECK-LE: xsdivdp f1, f1, f2
; CHECK-LE: blr
}

define double @test3(double %a, double %b) {
entry:
  %v = fadd double %a, %b
  ret double %v

; CHECK-LABEL: @test3
; CHECK: xsadddp f1, f1, f2
; CHECK: blr

; CHECK-LE-LABEL: @test3
; CHECK-LE: xsadddp f1, f1, f2
; CHECK-LE: blr
}

define <2 x double> @test4(<2 x double> %a, <2 x double> %b) {
entry:
  %v = fadd <2 x double> %a, %b
  ret <2 x double> %v

; CHECK-LABEL: @test4
; CHECK: xvadddp v2, v2, v3
; CHECK: blr

; CHECK-LE-LABEL: @test4
; CHECK-LE: xvadddp v2, v2, v3
; CHECK-LE: blr
}

define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = xor <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test5
; CHECK-REG: xxlxor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test5
; CHECK-FISL: xxlxor v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test5
; CHECK-LE: xxlxor v2, v2, v3
; CHECK-LE: blr
}

define <8 x i16> @test6(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = xor <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test6
; CHECK-REG: xxlxor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test6
; CHECK-FISL: xxlxor v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test6
; CHECK-LE: xxlxor v2, v2, v3
; CHECK-LE: blr
}

define <16 x i8> @test7(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = xor <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test7
; CHECK-REG: xxlxor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test7
; CHECK-FISL: xxlxor v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test7
; CHECK-LE: xxlxor v2, v2, v3
; CHECK-LE: blr
}

define <4 x i32> @test8(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test8
; CHECK-REG: xxlor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test8
; CHECK-FISL: xxlor v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test8
; CHECK-LE: xxlor v2, v2, v3
; CHECK-LE: blr
}

define <8 x i16> @test9(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test9
; CHECK-REG: xxlor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test9
; CHECK-FISL: xxlor v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test9
; CHECK-LE: xxlor v2, v2, v3
; CHECK-LE: blr
}

define <16 x i8> @test10(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test10
; CHECK-REG: xxlor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test10
; CHECK-FISL: xxlor v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test10
; CHECK-LE: xxlor v2, v2, v3
; CHECK-LE: blr
}

define <4 x i32> @test11(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = and <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test11
; CHECK-REG: xxland v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test11
; CHECK-FISL: xxland v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test11
; CHECK-LE: xxland v2, v2, v3
; CHECK-LE: blr
}

define <8 x i16> @test12(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = and <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test12
; CHECK-REG: xxland v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test12
; CHECK-FISL: xxland v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test12
; CHECK-LE: xxland v2, v2, v3
; CHECK-LE: blr
}

define <16 x i8> @test13(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = and <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test13
; CHECK-REG: xxland v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test13
; CHECK-FISL: xxland v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test13
; CHECK-LE: xxland v2, v2, v3
; CHECK-LE: blr
}

define <4 x i32> @test14(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  %w = xor <4 x i32> %v, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %w

; CHECK-REG-LABEL: @test14
; CHECK-REG: xxlnor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test14
; CHECK-FISL: xxlor vs0, v2, v3
; CHECK-FISL: xxlnor v2, v2, v3
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: li r3, -16
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: stxvd2x vs0, r1, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test14
; CHECK-LE: xxlnor v2, v2, v3
; CHECK-LE: blr
}

define <8 x i16> @test15(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  %w = xor <8 x i16> %v, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %w

; CHECK-REG-LABEL: @test15
; CHECK-REG: xxlnor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test15
; CHECK-FISL: xxlor vs0, v2, v3
; CHECK-FISL: xxlor v4, vs0, vs0
; CHECK-FISL: xxlnor vs0, v2, v3
; CHECK-FISL: xxlor v2, vs0, vs0
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: li r3, -16
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: stxvd2x v4, r1, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test15
; CHECK-LE: xxlnor v2, v2, v3
; CHECK-LE: blr
}

define <16 x i8> @test16(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  %w = xor <16 x i8> %v, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %w

; CHECK-REG-LABEL: @test16
; CHECK-REG: xxlnor v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test16
; CHECK-FISL: xxlor vs0, v2, v3
; CHECK-FISL: xxlor v4, vs0, vs0
; CHECK-FISL: xxlnor vs0, v2, v3
; CHECK-FISL: xxlor v2, vs0, vs0
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: li r3, -16
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: stxvd2x v4, r1, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test16
; CHECK-LE: xxlnor v2, v2, v3
; CHECK-LE: blr
}

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
entry:
  %w = xor <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %v = and <4 x i32> %a, %w
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test17
; CHECK-REG: xxlandc v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test17
; CHECK-FISL: xxlnor v3, v3, v3
; CHECK-FISL: xxland v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test17
; CHECK-LE: xxlandc v2, v2, v3
; CHECK-LE: blr
}

define <8 x i16> @test18(<8 x i16> %a, <8 x i16> %b) {
entry:
  %w = xor <8 x i16> %b, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %v = and <8 x i16> %a, %w
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test18
; CHECK-REG: xxlandc v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test18
; CHECK-FISL: xxlnor vs0, v3, v3
; CHECK-FISL: xxlor v4, vs0, vs0
; CHECK-FISL: xxlandc vs0, v2, v3
; CHECK-FISL: xxlor v2, vs0, vs0
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: li r3, -16
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: stxvd2x v4, r1, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test18
; CHECK-LE: xxlandc v2, v2, v3
; CHECK-LE: blr
}

define <16 x i8> @test19(<16 x i8> %a, <16 x i8> %b) {
entry:
  %w = xor <16 x i8> %b, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %v = and <16 x i8> %a, %w
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test19
; CHECK-REG: xxlandc v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test19
; CHECK-FISL: xxlnor vs0, v3, v3
; CHECK-FISL: xxlor v4, vs0, vs0
; CHECK-FISL: xxlandc vs0, v2, v3
; CHECK-FISL: xxlor v2, vs0, vs0
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: li r3, -16
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: stxvd2x v4, r1, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test19
; CHECK-LE: xxlandc v2, v2, v3
; CHECK-LE: blr
}

define <4 x i32> @test20(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d) {
entry:
  %m = icmp eq <4 x i32> %c, %d
  %v = select <4 x i1> %m, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test20
; CHECK-REG: vcmpequw v4, v4, v5
; CHECK-REG: xxsel v2, v3, v2, v4
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test20
; CHECK-FISL: vcmpequw v4, v4, v5
; CHECK-FISL: xxsel v2, v3, v2, v4
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test20
; CHECK-LE: vcmpequw v4, v4, v5
; CHECK-LE: xxsel v2, v3, v2, v4
; CHECK-LE: blr
}

define <4 x float> @test21(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp oeq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v

; CHECK-REG-LABEL: @test21
; CHECK-REG: xvcmpeqsp vs0, v4, v5
; CHECK-REG: xxsel v2, v3, v2, vs0
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test21
; CHECK-FISL: xvcmpeqsp v4, v4, v5
; CHECK-FISL: xxsel v2, v3, v2, v4
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test21
; CHECK-LE: xvcmpeqsp vs0, v4, v5
; CHECK-LE: xxsel v2, v3, v2, vs0
; CHECK-LE: blr
}

define <4 x float> @test22(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp ueq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v

; CHECK-REG-LABEL: @test22
; CHECK-REG-DAG: xvcmpeqsp vs0, v5, v5
; CHECK-REG-DAG: xvcmpeqsp vs1, v4, v4
; CHECK-REG-DAG: xvcmpeqsp vs2, v4, v5
; CHECK-REG-DAG: xxlnor vs0, vs0, vs0
; CHECK-REG-DAG: xxlnor vs1, vs1, vs1
; CHECK-REG-DAG: xxlor vs0, vs1, vs0
; CHECK-REG-DAG: xxlor vs0, vs2, vs0
; CHECK-REG: xxsel v2, v3, v2, vs0
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test22
; CHECK-FISL-DAG: xvcmpeqsp vs0, v4, v5
; CHECK-FISL-DAG: xvcmpeqsp v5, v5, v5
; CHECK-FISL-DAG: xvcmpeqsp v4, v4, v4
; CHECK-FISL-DAG: xxlnor v5, v5, v5
; CHECK-FISL-DAG: xxlnor v4, v4, v4
; CHECK-FISL-DAG: xxlor v4, v4, v5
; CHECK-FISL-DAG: xxlor vs0, vs0, v4
; CHECK-FISL: xxsel v2, v3, v2, vs0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test22
; CHECK-LE-DAG: xvcmpeqsp vs0, v5, v5
; CHECK-LE-DAG: xvcmpeqsp vs1, v4, v4
; CHECK-LE-DAG: xvcmpeqsp vs2, v4, v5
; CHECK-LE-DAG: xxlnor vs0, vs0, vs0
; CHECK-LE-DAG: xxlnor vs1, vs1, vs1
; CHECK-LE-DAG: xxlor vs0, vs1, vs0
; CHECK-LE-DAG: xxlor vs0, vs2, vs0
; CHECK-LE: xxsel v2, v3, v2, vs0
; CHECK-LE: blr
}

define <8 x i16> @test23(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, <8 x i16> %d) {
entry:
  %m = icmp eq <8 x i16> %c, %d
  %v = select <8 x i1> %m, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test23
; CHECK-REG: vcmpequh v4, v4, v5
; CHECK-REG: xxsel v2, v3, v2, v4
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test23
; CHECK-FISL: vcmpequh v4, v4, v5
; CHECK-FISL: xxsel v2, v3, v2, v4
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test23
; CHECK-LE: vcmpequh v4, v4, v5
; CHECK-LE: xxsel v2, v3, v2, v4
; CHECK-LE: blr
}

define <16 x i8> @test24(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d) {
entry:
  %m = icmp eq <16 x i8> %c, %d
  %v = select <16 x i1> %m, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test24
; CHECK-REG: vcmpequb v4, v4, v5
; CHECK-REG: xxsel v2, v3, v2, v4
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test24
; CHECK-FISL: vcmpequb v4, v4, v5
; CHECK-FISL: xxsel v2, v3, v2, v4
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test24
; CHECK-LE: vcmpequb v4, v4, v5
; CHECK-LE: xxsel v2, v3, v2, v4
; CHECK-LE: blr
}

define <2 x double> @test25(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %d) {
entry:
  %m = fcmp oeq <2 x double> %c, %d
  %v = select <2 x i1> %m, <2 x double> %a, <2 x double> %b
  ret <2 x double> %v

; CHECK-LABEL: @test25
; CHECK: xvcmpeqdp vs0, v4, v5
; CHECK: xxsel v2, v3, v2, vs0
; CHECK: blr

; CHECK-LE-LABEL: @test25
; CHECK-LE: xvcmpeqdp v4, v4, v5
; CHECK-LE: xxsel v2, v3, v2, v4
; CHECK-LE: blr
}

define <2 x i64> @test26(<2 x i64> %a, <2 x i64> %b) {
  %v = add <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test26

; Make sure we use only two stores (one for each operand).
; CHECK: stxvd2x v3, 0, r3
; CHECK: stxvd2x v2, 0, r4
; CHECK-NOT: stxvd2x

; FIXME: The code quality here is not good; just make sure we do something for now.
; CHECK: add r3, r4, r3
; CHECK: add r3, r4, r3
; CHECK: blr

; CHECK-LE: vaddudm v2, v2, v3
; CHECK-LE: blr
}

define <2 x i64> @test27(<2 x i64> %a, <2 x i64> %b) {
  %v = and <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test27
; CHECK: xxland v2, v2, v3
; CHECK: blr

; CHECK-LE-LABEL: @test27
; CHECK-LE: xxland v2, v2, v3
; CHECK-LE: blr
}

define <2 x double> @test28(<2 x double>* %a) {
  %v = load <2 x double>, <2 x double>* %a, align 16
  ret <2 x double> %v

; CHECK-LABEL: @test28
; CHECK: lxvd2x v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test28
; CHECK-LE: lxvd2x vs0, 0, r3
; CHECK-LE: xxswapd v2, vs0
; CHECK-LE: blr
}

define void @test29(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 16
  ret void

; CHECK-LABEL: @test29
; CHECK: stxvd2x v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test29
; CHECK-LE: xxswapd vs0, v2
; CHECK-LE: stxvd2x vs0, 0, r3
; CHECK-LE: blr
}

define <2 x double> @test28u(<2 x double>* %a) {
  %v = load <2 x double>, <2 x double>* %a, align 8
  ret <2 x double> %v

; CHECK-LABEL: @test28u
; CHECK: lxvd2x v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test28u
; CHECK-LE: lxvd2x vs0, 0, r3
; CHECK-LE: xxswapd v2, vs0
; CHECK-LE: blr
}

define void @test29u(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 8
  ret void

; CHECK-LABEL: @test29u
; CHECK: stxvd2x v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test29u
; CHECK-LE: xxswapd vs0, v2
; CHECK-LE: stxvd2x vs0, 0, r3
; CHECK-LE: blr
}

define <2 x i64> @test30(<2 x i64>* %a) {
  %v = load <2 x i64>, <2 x i64>* %a, align 16
  ret <2 x i64> %v

; CHECK-REG-LABEL: @test30
; CHECK-REG: lxvd2x v2, 0, r3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test30
; CHECK-FISL: lxvd2x vs0, 0, r3
; CHECK-FISL: xxlor v2, vs0, vs0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test30
; CHECK-LE: lxvd2x vs0, 0, r3
; CHECK-LE: xxswapd v2, vs0
; CHECK-LE: blr
}

define void @test31(<2 x i64>* %a, <2 x i64> %b) {
  store <2 x i64> %b, <2 x i64>* %a, align 16
  ret void

; CHECK-LABEL: @test31
; CHECK: stxvd2x v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test31
; CHECK-LE: xxswapd vs0, v2
; CHECK-LE: stxvd2x vs0, 0, r3
; CHECK-LE: blr
}

define <4 x float> @test32(<4 x float>* %a) {
  %v = load <4 x float>, <4 x float>* %a, align 16
  ret <4 x float> %v

; CHECK-REG-LABEL: @test32
; CHECK-REG: lxvw4x v2, 0, r3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test32
; CHECK-FISL: lxvw4x v2, 0, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test32
; CHECK-LE: lvx v2, 0, r3
; CHECK-LE-NOT: xxswapd
; CHECK-LE: blr
}

define void @test33(<4 x float>* %a, <4 x float> %b) {
  store <4 x float> %b, <4 x float>* %a, align 16
  ret void

; CHECK-REG-LABEL: @test33
; CHECK-REG: stxvw4x v2, 0, r3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test33
; CHECK-FISL: stxvw4x v2, 0, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test33
; CHECK-LE-NOT: xxswapd
; CHECK-LE: stvx v2, 0, r3
; CHECK-LE: blr
}

define <4 x float> @test32u(<4 x float>* %a) {
  %v = load <4 x float>, <4 x float>* %a, align 8
  ret <4 x float> %v

; CHECK-LABEL: @test32u
; CHECK-DAG: lvsl v3, 0, r3
; CHECK-DAG: lvx v2, r3, r4
; CHECK-DAG: lvx v4, 0, r3
; CHECK: vperm v2, v4, v2, v3
; CHECK: blr

; CHECK-LE-LABEL: @test32u
; CHECK-LE: lxvd2x vs0, 0, r3
; CHECK-LE: xxswapd v2, vs0
; CHECK-LE: blr
}

define void @test33u(<4 x float>* %a, <4 x float> %b) {
  store <4 x float> %b, <4 x float>* %a, align 8
  ret void

; CHECK-REG-LABEL: @test33u
; CHECK-REG: stxvw4x v2, 0, r3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test33u
; CHECK-FISL: stxvw4x v2, 0, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test33u
; CHECK-LE: xxswapd vs0, v2
; CHECK-LE: stxvd2x vs0, 0, r3
; CHECK-LE: blr
}

define <4 x i32> @test34(<4 x i32>* %a) {
  %v = load <4 x i32>, <4 x i32>* %a, align 16
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test34
; CHECK-REG: lxvw4x v2, 0, r3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test34
; CHECK-FISL: lxvw4x v2, 0, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test34
; CHECK-LE: lvx v2, 0, r3
; CHECK-LE-NOT: xxswapd
; CHECK-LE: blr
}

define void @test35(<4 x i32>* %a, <4 x i32> %b) {
  store <4 x i32> %b, <4 x i32>* %a, align 16
  ret void

; CHECK-REG-LABEL: @test35
; CHECK-REG: stxvw4x v2, 0, r3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test35
; CHECK-FISL: stxvw4x v2, 0, r3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test35
; CHECK-LE-NOT: xxswapd
; CHECK-LE: stvx v2, 0, r3
; CHECK-LE: blr
}

define <2 x double> @test40(<2 x i64> %a) {
  %v = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %v

; CHECK-LABEL: @test40
; CHECK: xvcvuxddp v2, v2
; CHECK: blr

; CHECK-LE-LABEL: @test40
; CHECK-LE: xvcvuxddp v2, v2
; CHECK-LE: blr
}

define <2 x double> @test41(<2 x i64> %a) {
  %v = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %v

; CHECK-LABEL: @test41
; CHECK: xvcvsxddp v2, v2
; CHECK: blr

; CHECK-LE-LABEL: @test41
; CHECK-LE: xvcvsxddp v2, v2
; CHECK-LE: blr
}

define <2 x i64> @test42(<2 x double> %a) {
  %v = fptoui <2 x double> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test42
; CHECK: xvcvdpuxds v2, v2
; CHECK: blr

; CHECK-LE-LABEL: @test42
; CHECK-LE: xvcvdpuxds v2, v2
; CHECK-LE: blr
}

define <2 x i64> @test43(<2 x double> %a) {
  %v = fptosi <2 x double> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test43
; CHECK: xvcvdpsxds v2, v2
; CHECK: blr

; CHECK-LE-LABEL: @test43
; CHECK-LE: xvcvdpsxds v2, v2
; CHECK-LE: blr
}

define <2 x float> @test44(<2 x i64> %a) {
  %v = uitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %v

; CHECK-LABEL: @test44
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x float> @test45(<2 x i64> %a) {
  %v = sitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %v

; CHECK-LABEL: @test45
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x i64> @test46(<2 x float> %a) {
  %v = fptoui <2 x float> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test46
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x i64> @test47(<2 x float> %a) {
  %v = fptosi <2 x float> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test47
; FIXME: The code quality here looks pretty bad.
; CHECK: blr
}

define <2 x double> @test50(double* %a) {
  %v = load double, double* %a, align 8
  %w = insertelement <2 x double> undef, double %v, i32 0
  %x = insertelement <2 x double> %w, double %v, i32 1
  ret <2 x double> %x

; CHECK-LABEL: @test50
; CHECK: lxvdsx v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test50
; CHECK-LE: lxvdsx v2, 0, r3
; CHECK-LE: blr
}

define <2 x double> @test51(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %v

; CHECK-LABEL: @test51
; CHECK: xxspltd v2, v2, 0
; CHECK: blr

; CHECK-LE-LABEL: @test51
; CHECK-LE: xxspltd v2, v2, 1
; CHECK-LE: blr
}

define <2 x double> @test52(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %v

; CHECK-LABEL: @test52
; CHECK: xxmrghd v2, v2, v3
; CHECK: blr

; CHECK-LE-LABEL: @test52
; CHECK-LE: xxmrgld v2, v3, v2
; CHECK-LE: blr
}

define <2 x double> @test53(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x double> %v

; CHECK-LABEL: @test53
; CHECK: xxmrghd v2, v3, v2
; CHECK: blr

; CHECK-LE-LABEL: @test53
; CHECK-LE: xxmrgld v2, v2, v3
; CHECK-LE: blr
}

define <2 x double> @test54(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %v

; CHECK-LABEL: @test54
; CHECK: xxpermdi v2, v2, v3, 2
; CHECK: blr

; CHECK-LE-LABEL: @test54
; CHECK-LE: xxpermdi v2, v3, v2, 2
; CHECK-LE: blr
}

define <2 x double> @test55(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %v

; CHECK-LABEL: @test55
; CHECK: xxmrgld v2, v2, v3
; CHECK: blr

; CHECK-LE-LABEL: @test55
; CHECK-LE: xxmrghd v2, v3, v2
; CHECK-LE: blr
}

define <2 x i64> @test56(<2 x i64> %a, <2 x i64> %b) {
  %v = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %v

; CHECK-LABEL: @test56
; CHECK: xxmrgld v2, v2, v3
; CHECK: blr

; CHECK-LE-LABEL: @test56
; CHECK-LE: xxmrghd v2, v3, v2
; CHECK-LE: blr
}

define <2 x i64> @test60(<2 x i64> %a, <2 x i64> %b) {
  %v = shl <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test60
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x v3, 0, r3
; CHECK: stxvd2x v2, 0, r4
; CHECK: sld r3, r4, r3
; CHECK: sld r3, r4, r3
; CHECK: lxvd2x v2, 0, r3
; CHECK: blr
}

define <2 x i64> @test61(<2 x i64> %a, <2 x i64> %b) {
  %v = lshr <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test61
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x v3, 0, r3
; CHECK: stxvd2x v2, 0, r4
; CHECK: srd r3, r4, r3
; CHECK: srd r3, r4, r3
; CHECK: lxvd2x v2, 0, r3
; CHECK: blr
}

define <2 x i64> @test62(<2 x i64> %a, <2 x i64> %b) {
  %v = ashr <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test62
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x v3, 0, r3
; CHECK: stxvd2x v2, 0, r4
; CHECK: srad r3, r4, r3
; CHECK: srad r3, r4, r3
; CHECK: lxvd2x v2, 0, r3
; CHECK: blr
}

define double @test63(<2 x double> %a) {
  %v = extractelement <2 x double> %a, i32 0
  ret double %v

; CHECK-REG-LABEL: @test63
; CHECK-REG: xxlor f1, v2, v2
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test63
; CHECK-FISL: xxlor f0, v2, v2
; CHECK-FISL: fmr f1, f0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test63
; CHECK-LE: xxswapd vs1, v2
; CHECK-LE: blr
}

define double @test64(<2 x double> %a) {
  %v = extractelement <2 x double> %a, i32 1
  ret double %v

; CHECK-REG-LABEL: @test64
; CHECK-REG: xxswapd vs1, v2
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test64
; CHECK-FISL: xxswapd  v2, v2
; CHECK-FISL: xxlor f0, v2, v2
; CHECK-FISL: fmr f1, f0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test64
; CHECK-LE: xxlor f1, v2, v2
}

define <2 x i1> @test65(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp eq <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-REG-LABEL: @test65
; CHECK-REG: vcmpequw v2, v2, v3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test65
; CHECK-FISL: vcmpequw v2, v2, v3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test65
; CHECK-LE: vcmpequd v2, v2, v3
; CHECK-LE: blr
}

define <2 x i1> @test66(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp ne <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-REG-LABEL: @test66
; CHECK-REG: vcmpequw v2, v2, v3
; CHECK-REG: xxlnor v2, v2, v2
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test66
; CHECK-FISL: vcmpequw v2, v2, v3
; CHECK-FISL: xxlnor v2, v2, v2
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test66
; CHECK-LE: vcmpequd v2, v2, v3
; CHECK-LE: xxlnor v2, v2, v2
; CHECK-LE: blr
}

define <2 x i1> @test67(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp ult <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-LABEL: @test67
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x v3, 0, r3
; CHECK: stxvd2x v2, 0, r4
; CHECK: cmpld r4, r3
; CHECK: cmpld r6, r5
; CHECK: lxvd2x v2, 0, r3
; CHECK: blr

; CHECK-LE-LABEL: @test67
; CHECK-LE: vcmpgtud v2, v3, v2
; CHECK-LE: blr
}

define <2 x double> @test68(<2 x i32> %a) {
  %w = sitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test68
; CHECK: xxmrghw vs0, v2, v2
; CHECK: xvcvsxwdp v2, vs0
; CHECK: blr

; CHECK-LE-LABEL: @test68
; CHECK-LE: xxmrglw v2, v2, v2
; CHECK-LE: xvcvsxwdp v2, v2
; CHECK-LE: blr
}

; This gets scalarized so the code isn't great
define <2 x double> @test69(<2 x i16> %a) {
  %w = sitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test69
; CHECK-DAG: lxvd2x v2, 0, r3
; CHECK-DAG: xvcvsxddp v2, v2
; CHECK: blr

; CHECK-LE-LABEL: @test69
; CHECK-LE: vperm
; CHECK-LE: vsld
; CHECK-LE: vsrad
; CHECK-LE: xvcvsxddp v2, v2
; CHECK-LE: blr
}

; This gets scalarized so the code isn't great
define <2 x double> @test70(<2 x i8> %a) {
  %w = sitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test70
; CHECK-DAG: lxvd2x v2, 0, r3
; CHECK-DAG: xvcvsxddp v2, v2
; CHECK: blr

; CHECK-LE-LABEL: @test70
; CHECK-LE: vperm
; CHECK-LE: vsld
; CHECK-LE: vsrad
; CHECK-LE: xvcvsxddp v2, v2
; CHECK-LE: blr
}

; This gets scalarized so the code isn't great
define <2 x i32> @test80(i32 %v) {
  %b1 = insertelement <2 x i32> undef, i32 %v, i32 0
  %b2 = shufflevector <2 x i32> %b1, <2 x i32> undef, <2 x i32> zeroinitializer
  %i = add <2 x i32> %b2, <i32 2, i32 3>
  ret <2 x i32> %i

; CHECK-REG-LABEL: @test80
; CHECK-REG-DAG: stw r3, -16(r1)
; CHECK-REG-DAG: addi r4, r1, -16
; CHECK-REG: addis r3, r2, .LCPI65_0@toc@ha
; CHECK-REG-DAG: addi r3, r3, .LCPI65_0@toc@l
; CHECK-REG-DAG: lxvw4x vs0, 0, r4
; CHECK-REG-DAG: lxvw4x v3, 0, r3
; CHECK-REG: xxspltw v2, vs0, 0
; CHECK-REG: vadduwm v2, v2, v3
; CHECK-REG-NOT: stxvw4x
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test80
; CHECK-FISL: mr r4, r3
; CHECK-FISL: stw r4, -16(r1)
; CHECK-FISL: addi r3, r1, -16
; CHECK-FISL-DAG: lxvw4x vs0, 0, r3
; CHECK-FISL-DAG: xxspltw v2, vs0, 0
; CHECK-FISL: addis r3, r2, .LCPI65_0@toc@ha
; CHECK-FISL: addi r3, r3, .LCPI65_0@toc@l
; CHECK-FISL-DAG: lxvw4x v3, 0, r3
; CHECK-FISL: vadduwm
; CHECK-FISL-NOT: stxvw4x
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test80
; CHECK-LE-DAG: mtvsrd f0, r3
; CHECK-LE-DAG: xxswapd  vs0, vs0
; CHECK-LE-DAG: addi r3, r4, .LCPI65_0@toc@l
; CHECK-LE-DAG: lvx v3, 0, r3
; CHECK-LE-DAG: xxspltw v2, vs0, 3
; CHECK-LE-NOT: xxswapd v3,
; CHECK-LE: vadduwm v2, v2, v3
; CHECK-LE: blr
}

define <2 x double> @test81(<4 x float> %b) {
  %w = bitcast <4 x float> %b to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test81
; CHECK: blr

; CHECK-LE-LABEL: @test81
; CHECK-LE: blr
}

define double @test82(double %a, double %b, double %c, double %d) {
entry:
  %m = fcmp oeq double %c, %d
  %v = select i1 %m, double %a, double %b
  ret double %v

; CHECK-REG-LABEL: @test82
; CHECK-REG: xscmpudp cr0, f3, f4
; CHECK-REG: beqlr cr0

; CHECK-FISL-LABEL: @test82
; CHECK-FISL: xscmpudp cr0, f3, f4
; CHECK-FISL: beq cr0

; CHECK-LE-LABEL: @test82
; CHECK-LE: xscmpudp cr0, f3, f4
; CHECK-LE: beqlr cr0
}
