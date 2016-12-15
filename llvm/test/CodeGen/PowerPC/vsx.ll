; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx < %s | FileCheck %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-REG %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx -fast-isel -O0 < %s | FileCheck %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx -fast-isel -O0 < %s | FileCheck -check-prefix=CHECK-FISL %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-LE %s

define double @test1(double %a, double %b) {
entry:
  %v = fmul double %a, %b
  ret double %v

; CHECK-LABEL: @test1
; CHECK: xsmuldp 1, 1, 2
; CHECK: blr

; CHECK-LE-LABEL: @test1
; CHECK-LE: xsmuldp 1, 1, 2
; CHECK-LE: blr
}

define double @test2(double %a, double %b) {
entry:
  %v = fdiv double %a, %b
  ret double %v

; CHECK-LABEL: @test2
; CHECK: xsdivdp 1, 1, 2
; CHECK: blr

; CHECK-LE-LABEL: @test2
; CHECK-LE: xsdivdp 1, 1, 2
; CHECK-LE: blr
}

define double @test3(double %a, double %b) {
entry:
  %v = fadd double %a, %b
  ret double %v

; CHECK-LABEL: @test3
; CHECK: xsadddp 1, 1, 2
; CHECK: blr

; CHECK-LE-LABEL: @test3
; CHECK-LE: xsadddp 1, 1, 2
; CHECK-LE: blr
}

define <2 x double> @test4(<2 x double> %a, <2 x double> %b) {
entry:
  %v = fadd <2 x double> %a, %b
  ret <2 x double> %v

; CHECK-LABEL: @test4
; CHECK: xvadddp 34, 34, 35
; CHECK: blr

; CHECK-LE-LABEL: @test4
; CHECK-LE: xvadddp 34, 34, 35
; CHECK-LE: blr
}

define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = xor <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test5
; CHECK-REG: xxlxor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test5
; CHECK-FISL: xxlxor 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test5
; CHECK-LE: xxlxor 34, 34, 35
; CHECK-LE: blr
}

define <8 x i16> @test6(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = xor <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test6
; CHECK-REG: xxlxor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test6
; CHECK-FISL: xxlxor 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test6
; CHECK-LE: xxlxor 34, 34, 35
; CHECK-LE: blr
}

define <16 x i8> @test7(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = xor <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test7
; CHECK-REG: xxlxor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test7
; CHECK-FISL: xxlxor 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test7
; CHECK-LE: xxlxor 34, 34, 35
; CHECK-LE: blr
}

define <4 x i32> @test8(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test8
; CHECK-REG: xxlor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test8
; CHECK-FISL: xxlor 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test8
; CHECK-LE: xxlor 34, 34, 35
; CHECK-LE: blr
}

define <8 x i16> @test9(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test9
; CHECK-REG: xxlor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test9
; CHECK-FISL: xxlor 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test9
; CHECK-LE: xxlor 34, 34, 35
; CHECK-LE: blr
}

define <16 x i8> @test10(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test10
; CHECK-REG: xxlor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test10
; CHECK-FISL: xxlor 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test10
; CHECK-LE: xxlor 34, 34, 35
; CHECK-LE: blr
}

define <4 x i32> @test11(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = and <4 x i32> %a, %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test11
; CHECK-REG: xxland 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test11
; CHECK-FISL: xxland 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test11
; CHECK-LE: xxland 34, 34, 35
; CHECK-LE: blr
}

define <8 x i16> @test12(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = and <8 x i16> %a, %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test12
; CHECK-REG: xxland 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test12
; CHECK-FISL: xxland 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test12
; CHECK-LE: xxland 34, 34, 35
; CHECK-LE: blr
}

define <16 x i8> @test13(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = and <16 x i8> %a, %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test13
; CHECK-REG: xxland 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test13
; CHECK-FISL: xxland 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test13
; CHECK-LE: xxland 34, 34, 35
; CHECK-LE: blr
}

define <4 x i32> @test14(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = or <4 x i32> %a, %b
  %w = xor <4 x i32> %v, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %w

; CHECK-REG-LABEL: @test14
; CHECK-REG: xxlnor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test14
; CHECK-FISL: xxlor 0, 34, 35
; CHECK-FISL: xxlnor 34, 34, 35
; CHECK-FISL: lis 0, -1
; CHECK-FISL: ori 0, 0, 65520
; CHECK-FISL: stxvd2x 0, 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test14
; CHECK-LE: xxlnor 34, 34, 35
; CHECK-LE: blr
}

define <8 x i16> @test15(<8 x i16> %a, <8 x i16> %b) {
entry:
  %v = or <8 x i16> %a, %b
  %w = xor <8 x i16> %v, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %w

; CHECK-REG-LABEL: @test15
; CHECK-REG: xxlnor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test15
; CHECK-FISL: xxlor 0, 34, 35
; CHECK-FISL: xxlor 36, 0, 0
; CHECK-FISL: xxlnor 0, 34, 35
; CHECK-FISL: xxlor 34, 0, 0
; CHECK-FISL: lis 0, -1
; CHECK-FISL: ori 0, 0, 65520
; CHECK-FISL: stxvd2x 36, 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test15
; CHECK-LE: xxlnor 34, 34, 35
; CHECK-LE: blr
}

define <16 x i8> @test16(<16 x i8> %a, <16 x i8> %b) {
entry:
  %v = or <16 x i8> %a, %b
  %w = xor <16 x i8> %v, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %w

; CHECK-REG-LABEL: @test16
; CHECK-REG: xxlnor 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test16
; CHECK-FISL: xxlor 0, 34, 35
; CHECK-FISL: xxlor 36, 0, 0
; CHECK-FISL: xxlnor 0, 34, 35
; CHECK-FISL: xxlor 34, 0, 0
; CHECK-FISL: lis 0, -1
; CHECK-FISL: ori 0, 0, 65520
; CHECK-FISL: stxvd2x 36, 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test16
; CHECK-LE: xxlnor 34, 34, 35
; CHECK-LE: blr
}

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
entry:
  %w = xor <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %v = and <4 x i32> %a, %w
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test17
; CHECK-REG: xxlandc 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test17
; CHECK-FISL: xxlnor 35, 35, 35
; CHECK-FISL: xxland 34, 34, 35
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test17
; CHECK-LE: xxlandc 34, 34, 35
; CHECK-LE: blr
}

define <8 x i16> @test18(<8 x i16> %a, <8 x i16> %b) {
entry:
  %w = xor <8 x i16> %b, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %v = and <8 x i16> %a, %w
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test18
; CHECK-REG: xxlandc 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test18
; CHECK-FISL: xxlnor 0, 35, 35
; CHECK-FISL: xxlor 36, 0, 0
; CHECK-FISL: xxlandc 0, 34, 35
; CHECK-FISL: xxlor 34, 0, 0
; CHECK-FISL: lis 0, -1
; CHECK-FISL: ori 0, 0, 65520
; CHECK-FISL: stxvd2x 36, 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test18
; CHECK-LE: xxlandc 34, 34, 35
; CHECK-LE: blr
}

define <16 x i8> @test19(<16 x i8> %a, <16 x i8> %b) {
entry:
  %w = xor <16 x i8> %b, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %v = and <16 x i8> %a, %w
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test19
; CHECK-REG: xxlandc 34, 34, 35
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test19
; CHECK-FISL: xxlnor 0, 35, 35
; CHECK-FISL: xxlor 36, 0, 0
; CHECK-FISL: xxlandc 0, 34, 35
; CHECK-FISL: xxlor 34, 0, 0
; CHECK-FISL: lis 0, -1
; CHECK-FISL: ori 0, 0, 65520
; CHECK-FISL: stxvd2x 36, 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test19
; CHECK-LE: xxlandc 34, 34, 35
; CHECK-LE: blr
}

define <4 x i32> @test20(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d) {
entry:
  %m = icmp eq <4 x i32> %c, %d
  %v = select <4 x i1> %m, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test20
; CHECK-REG: vcmpequw {{[0-9]+}}, 4, 5
; CHECK-REG: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test20
; CHECK-FISL: vcmpequw {{[0-9]+}}, 4, 5
; CHECK-FISL: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test20
; CHECK-LE: vcmpequw {{[0-9]+}}, 4, 5
; CHECK-LE: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-LE: blr
}

define <4 x float> @test21(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp oeq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v

; CHECK-REG-LABEL: @test21
; CHECK-REG: xvcmpeqsp [[V1:[0-9]+]], 36, 37
; CHECK-REG: xxsel 34, 35, 34, [[V1]]
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test21
; CHECK-FISL: xvcmpeqsp [[V1:[0-9]+]], 36, 37
; CHECK-FISL: xxsel 34, 35, 34, [[V1]]
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test21
; CHECK-LE: xvcmpeqsp [[V1:[0-9]+]], 36, 37
; CHECK-LE: xxsel 34, 35, 34, [[V1]]
; CHECK-LE: blr
}

define <4 x float> @test22(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp ueq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v

; CHECK-REG-LABEL: @test22
; CHECK-REG-DAG: xvcmpeqsp {{[0-9]+}}, 37, 37
; CHECK-REG-DAG: xvcmpeqsp {{[0-9]+}}, 36, 36
; CHECK-REG-DAG: xvcmpeqsp {{[0-9]+}}, 36, 37
; CHECK-REG-DAG: xxlnor
; CHECK-REG-DAG: xxlnor
; CHECK-REG-DAG: xxlor
; CHECK-REG-DAG: xxlor
; CHECK-REG: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test22
; CHECK-FISL-DAG: xvcmpeqsp {{[0-9]+}}, 37, 37
; CHECK-FISL-DAG: xvcmpeqsp {{[0-9]+}}, 36, 36
; CHECK-FISL-DAG: xvcmpeqsp {{[0-9]+}}, 36, 37
; CHECK-FISL-DAG: xxlnor
; CHECK-FISL-DAG: xxlnor
; CHECK-FISL-DAG: xxlor
; CHECK-FISL-DAG: xxlor
; CHECK-FISL: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test22
; CHECK-LE-DAG: xvcmpeqsp {{[0-9]+}}, 37, 37
; CHECK-LE-DAG: xvcmpeqsp {{[0-9]+}}, 36, 36
; CHECK-LE-DAG: xvcmpeqsp {{[0-9]+}}, 36, 37
; CHECK-LE-DAG: xxlnor
; CHECK-LE-DAG: xxlnor
; CHECK-LE-DAG: xxlor
; CHECK-LE-DAG: xxlor
; CHECK-LE: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-LE: blr
}

define <8 x i16> @test23(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, <8 x i16> %d) {
entry:
  %m = icmp eq <8 x i16> %c, %d
  %v = select <8 x i1> %m, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %v

; CHECK-REG-LABEL: @test23
; CHECK-REG: vcmpequh {{[0-9]+}}, 4, 5
; CHECK-REG: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test23
; CHECK-FISL: vcmpequh 4, 4, 5
; CHECK-FISL: xxsel 34, 35, 34, 36
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test23
; CHECK-LE: vcmpequh {{[0-9]+}}, 4, 5
; CHECK-LE: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-LE: blr
}

define <16 x i8> @test24(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d) {
entry:
  %m = icmp eq <16 x i8> %c, %d
  %v = select <16 x i1> %m, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %v

; CHECK-REG-LABEL: @test24
; CHECK-REG: vcmpequb {{[0-9]+}}, 4, 5
; CHECK-REG: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test24
; CHECK-FISL: vcmpequb 4, 4, 5
; CHECK-FISL: xxsel 34, 35, 34, 36
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test24
; CHECK-LE: vcmpequb {{[0-9]+}}, 4, 5
; CHECK-LE: xxsel 34, 35, 34, {{[0-9]+}}
; CHECK-LE: blr
}

define <2 x double> @test25(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %d) {
entry:
  %m = fcmp oeq <2 x double> %c, %d
  %v = select <2 x i1> %m, <2 x double> %a, <2 x double> %b
  ret <2 x double> %v

; CHECK-LABEL: @test25
; CHECK: xvcmpeqdp [[V1:[0-9]+]], 36, 37
; CHECK: xxsel 34, 35, 34, [[V1]]
; CHECK: blr

; CHECK-LE-LABEL: @test25
; CHECK-LE: xvcmpeqdp [[V1:[0-9]+]], 36, 37
; CHECK-LE: xxsel 34, 35, 34, [[V1]]
; CHECK-LE: blr
}

define <2 x i64> @test26(<2 x i64> %a, <2 x i64> %b) {
  %v = add <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test26

; Make sure we use only two stores (one for each operand).
; CHECK: stxvd2x 35,
; CHECK: stxvd2x 34,
; CHECK-NOT: stxvd2x

; FIXME: The code quality here is not good; just make sure we do something for now.
; CHECK: add
; CHECK: add
; CHECK: blr

; CHECK-LE: vaddudm 2, 2, 3
; CHECK-LE: blr
}

define <2 x i64> @test27(<2 x i64> %a, <2 x i64> %b) {
  %v = and <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test27
; CHECK: xxland 34, 34, 35
; CHECK: blr

; CHECK-LE-LABEL: @test27
; CHECK-LE: xxland 34, 34, 35
; CHECK-LE: blr
}

define <2 x double> @test28(<2 x double>* %a) {
  %v = load <2 x double>, <2 x double>* %a, align 16
  ret <2 x double> %v

; CHECK-LABEL: @test28
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr

; CHECK-LE-LABEL: @test28
; CHECK-LE: lxvd2x [[V1:[0-9]+]], 0, 3
; CHECK-LE: xxswapd 34, [[V1]]
; CHECK-LE: blr
}

define void @test29(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 16
  ret void

; CHECK-LABEL: @test29
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr

; CHECK-LE-LABEL: @test29
; CHECK-LE: xxswapd [[V1:[0-9]+]], 34
; CHECK-LE: stxvd2x [[V1]], 0, 3
; CHECK-LE: blr
}

define <2 x double> @test28u(<2 x double>* %a) {
  %v = load <2 x double>, <2 x double>* %a, align 8
  ret <2 x double> %v

; CHECK-LABEL: @test28u
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr

; CHECK-LE-LABEL: @test28u
; CHECK-LE: lxvd2x [[V1:[0-9]+]], 0, 3
; CHECK-LE: xxswapd 34, [[V1]]
; CHECK-LE: blr
}

define void @test29u(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 8
  ret void

; CHECK-LABEL: @test29u
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr

; CHECK-LE-LABEL: @test29u
; CHECK-LE: xxswapd [[V1:[0-9]+]], 34
; CHECK-LE: stxvd2x [[V1]], 0, 3
; CHECK-LE: blr
}

define <2 x i64> @test30(<2 x i64>* %a) {
  %v = load <2 x i64>, <2 x i64>* %a, align 16
  ret <2 x i64> %v

; CHECK-REG-LABEL: @test30
; CHECK-REG: lxvd2x 34, 0, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test30
; CHECK-FISL: lxvd2x 0, 0, 3
; CHECK-FISL: xxlor 34, 0, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test30
; CHECK-LE: lxvd2x [[V1:[0-9]+]], 0, 3
; CHECK-LE: xxswapd 34, [[V1]]
; CHECK-LE: blr
}

define void @test31(<2 x i64>* %a, <2 x i64> %b) {
  store <2 x i64> %b, <2 x i64>* %a, align 16
  ret void

; CHECK-LABEL: @test31
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr

; CHECK-LE-LABEL: @test31
; CHECK-LE: xxswapd [[V1:[0-9]+]], 34
; CHECK-LE: stxvd2x [[V1]], 0, 3
; CHECK-LE: blr
}

define <4 x float> @test32(<4 x float>* %a) {
  %v = load <4 x float>, <4 x float>* %a, align 16
  ret <4 x float> %v

; CHECK-REG-LABEL: @test32
; CHECK-REG: lxvw4x 34, 0, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test32
; CHECK-FISL: lxvw4x 34, 0, 3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test32
; CHECK-LE: lxvd2x [[V1:[0-9]+]], 0, 3
; CHECK-LE: xxswapd 34, [[V1]]
; CHECK-LE: blr
}

define void @test33(<4 x float>* %a, <4 x float> %b) {
  store <4 x float> %b, <4 x float>* %a, align 16
  ret void

; CHECK-REG-LABEL: @test33
; CHECK-REG: stxvw4x 34, 0, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test33
; CHECK-FISL: stxvw4x 34, 0, 3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test33
; CHECK-LE: xxswapd [[V1:[0-9]+]], 34
; CHECK-LE: stxvd2x [[V1]], 0, 3
; CHECK-LE: blr
}

define <4 x float> @test32u(<4 x float>* %a) {
  %v = load <4 x float>, <4 x float>* %a, align 8
  ret <4 x float> %v

; CHECK-LABEL: @test32u
; CHECK-DAG: lvsl
; CHECK-DAG: lvx
; CHECK-DAG: lvx
; CHECK: vperm 2,
; CHECK: blr

; CHECK-LE-LABEL: @test32u
; CHECK-LE: lxvd2x [[V1:[0-9]+]], 0, 3
; CHECK-LE: xxswapd 34, [[V1]]
; CHECK-LE: blr
}

define void @test33u(<4 x float>* %a, <4 x float> %b) {
  store <4 x float> %b, <4 x float>* %a, align 8
  ret void

; CHECK-REG-LABEL: @test33u
; CHECK-REG: stxvw4x 34, 0, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test33u
; CHECK-FISL: stxvw4x 34, 0, 3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test33u
; CHECK-LE: xxswapd [[V1:[0-9]+]], 34
; CHECK-LE: stxvd2x [[V1]], 0, 3
; CHECK-LE: blr
}

define <4 x i32> @test34(<4 x i32>* %a) {
  %v = load <4 x i32>, <4 x i32>* %a, align 16
  ret <4 x i32> %v

; CHECK-REG-LABEL: @test34
; CHECK-REG: lxvw4x 34, 0, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test34
; CHECK-FISL: lxvw4x 34, 0, 3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test34
; CHECK-LE: lxvd2x [[V1:[0-9]+]], 0, 3
; CHECK-LE: xxswapd 34, [[V1]]
; CHECK-LE: blr
}

define void @test35(<4 x i32>* %a, <4 x i32> %b) {
  store <4 x i32> %b, <4 x i32>* %a, align 16
  ret void

; CHECK-REG-LABEL: @test35
; CHECK-REG: stxvw4x 34, 0, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test35
; CHECK-FISL: stxvw4x 34, 0, 3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test35
; CHECK-LE: xxswapd [[V1:[0-9]+]], 34
; CHECK-LE: stxvd2x [[V1]], 0, 3
; CHECK-LE: blr
}

define <2 x double> @test40(<2 x i64> %a) {
  %v = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %v

; CHECK-LABEL: @test40
; CHECK: xvcvuxddp 34, 34
; CHECK: blr

; CHECK-LE-LABEL: @test40
; CHECK-LE: xvcvuxddp 34, 34
; CHECK-LE: blr
}

define <2 x double> @test41(<2 x i64> %a) {
  %v = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %v

; CHECK-LABEL: @test41
; CHECK: xvcvsxddp 34, 34
; CHECK: blr

; CHECK-LE-LABEL: @test41
; CHECK-LE: xvcvsxddp 34, 34
; CHECK-LE: blr
}

define <2 x i64> @test42(<2 x double> %a) {
  %v = fptoui <2 x double> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test42
; CHECK: xvcvdpuxds 34, 34
; CHECK: blr

; CHECK-LE-LABEL: @test42
; CHECK-LE: xvcvdpuxds 34, 34
; CHECK-LE: blr
}

define <2 x i64> @test43(<2 x double> %a) {
  %v = fptosi <2 x double> %a to <2 x i64>
  ret <2 x i64> %v

; CHECK-LABEL: @test43
; CHECK: xvcvdpsxds 34, 34
; CHECK: blr

; CHECK-LE-LABEL: @test43
; CHECK-LE: xvcvdpsxds 34, 34
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
; CHECK: lxvdsx 34, 0, 3
; CHECK: blr

; CHECK-LE-LABEL: @test50
; CHECK-LE: lxvdsx 34, 0, 3
; CHECK-LE: blr
}

define <2 x double> @test51(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %v

; CHECK-LABEL: @test51
; CHECK: xxspltd 34, 34, 0
; CHECK: blr

; CHECK-LE-LABEL: @test51
; CHECK-LE: xxspltd 34, 34, 1
; CHECK-LE: blr
}

define <2 x double> @test52(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %v

; CHECK-LABEL: @test52
; CHECK: xxmrghd 34, 34, 35
; CHECK: blr

; CHECK-LE-LABEL: @test52
; CHECK-LE: xxmrgld 34, 35, 34
; CHECK-LE: blr
}

define <2 x double> @test53(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x double> %v

; CHECK-LABEL: @test53
; CHECK: xxmrghd 34, 35, 34
; CHECK: blr

; CHECK-LE-LABEL: @test53
; CHECK-LE: xxmrgld 34, 34, 35
; CHECK-LE: blr
}

define <2 x double> @test54(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %v

; CHECK-LABEL: @test54
; CHECK: xxpermdi 34, 34, 35, 2
; CHECK: blr

; CHECK-LE-LABEL: @test54
; CHECK-LE: xxpermdi 34, 35, 34, 2
; CHECK-LE: blr
}

define <2 x double> @test55(<2 x double> %a, <2 x double> %b) {
  %v = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %v

; CHECK-LABEL: @test55
; CHECK: xxmrgld 34, 34, 35
; CHECK: blr

; CHECK-LE-LABEL: @test55
; CHECK-LE: xxmrghd 34, 35, 34
; CHECK-LE: blr
}

define <2 x i64> @test56(<2 x i64> %a, <2 x i64> %b) {
  %v = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %v

; CHECK-LABEL: @test56
; CHECK: xxmrgld 34, 34, 35
; CHECK: blr

; CHECK-LE-LABEL: @test56
; CHECK-LE: xxmrghd 34, 35, 34
; CHECK-LE: blr
}

define <2 x i64> @test60(<2 x i64> %a, <2 x i64> %b) {
  %v = shl <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test60
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: sld
; CHECK: sld
; CHECK: lxvd2x
; CHECK: blr
}

define <2 x i64> @test61(<2 x i64> %a, <2 x i64> %b) {
  %v = lshr <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test61
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: srd
; CHECK: srd
; CHECK: lxvd2x
; CHECK: blr
}

define <2 x i64> @test62(<2 x i64> %a, <2 x i64> %b) {
  %v = ashr <2 x i64> %a, %b
  ret <2 x i64> %v

; CHECK-LABEL: @test62
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: srad
; CHECK: srad
; CHECK: lxvd2x
; CHECK: blr
}

define double @test63(<2 x double> %a) {
  %v = extractelement <2 x double> %a, i32 0
  ret double %v

; CHECK-REG-LABEL: @test63
; CHECK-REG: xxlor 1, 34, 34
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test63
; CHECK-FISL: xxlor 0, 34, 34
; CHECK-FISL: fmr 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test63
; CHECK-LE: xxswapd 1, 34
; CHECK-LE: blr
}

define double @test64(<2 x double> %a) {
  %v = extractelement <2 x double> %a, i32 1
  ret double %v

; CHECK-REG-LABEL: @test64
; CHECK-REG: xxswapd 1, 34
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test64
; CHECK-FISL: xxswapd  34, 34
; CHECK-FISL: xxlor 0, 34, 34
; CHECK-FISL: fmr 1, 0
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test64
; CHECK-LE: xxlor 1, 34, 34
}

define <2 x i1> @test65(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp eq <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-REG-LABEL: @test65
; CHECK-REG: vcmpequw 2, 2, 3
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test65
; CHECK-FISL: vcmpequw 2, 2, 3
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test65
; CHECK-LE: vcmpequd 2, 2, 3
; CHECK-LE: blr
}

define <2 x i1> @test66(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp ne <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-REG-LABEL: @test66
; CHECK-REG: vcmpequw {{[0-9]+}}, 2, 3
; CHECK-REG: xxlnor 34, {{[0-9]+}}, {{[0-9]+}}
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test66
; CHECK-FISL: vcmpequw 2, 2, 3
; CHECK-FISL: xxlnor 34, 34, 34
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test66
; CHECK-LE: vcmpequd {{[0-9]+}}, 2, 3
; CHECK-LE: xxlnor 34, {{[0-9]+}}, {{[0-9]+}}
; CHECK-LE: blr
}

define <2 x i1> @test67(<2 x i64> %a, <2 x i64> %b) {
  %w = icmp ult <2 x i64> %a, %b
  ret <2 x i1> %w

; CHECK-LABEL: @test67
; This should scalarize, and the current code quality is not good.
; CHECK: stxvd2x
; CHECK: stxvd2x
; CHECK: cmpld
; CHECK: cmpld
; CHECK: lxvd2x
; CHECK: blr

; CHECK-LE-LABEL: @test67
; CHECK-LE: vcmpgtud 2, 3, 2
; CHECK-LE: blr
}

define <2 x double> @test68(<2 x i32> %a) {
  %w = sitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test68
; CHECK: xxmrghw [[V1:[0-9]+]]
; CHECK: xvcvsxwdp 34, [[V1]]
; CHECK: blr

; CHECK-LE-LABEL: @test68
; CHECK-LE: xxmrglw [[V1:[0-9]+]], 34, 34
; CHECK-LE: xvcvsxwdp 34, [[V1]]
; CHECK-LE: blr
}

; This gets scalarized so the code isn't great
define <2 x double> @test69(<2 x i16> %a) {
  %w = sitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test69
; CHECK-DAG: lfiwax
; CHECK-DAG: lfiwax
; CHECK-DAG: xscvsxddp
; CHECK-DAG: xscvsxddp
; CHECK: xxmrghd
; CHECK: blr

; CHECK-LE-LABEL: @test69
; CHECK-LE: mfvsrd
; CHECK-LE: mtvsrwa
; CHECK-LE: mtvsrwa
; CHECK-LE: xscvsxddp
; CHECK-LE: xscvsxddp
; CHECK-LE: xxmrghd
; CHECK-LE: blr
}

; This gets scalarized so the code isn't great
define <2 x double> @test70(<2 x i8> %a) {
  %w = sitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %w

; CHECK-LABEL: @test70
; CHECK-DAG: lfiwax
; CHECK-DAG: lfiwax
; CHECK-DAG: xscvsxddp
; CHECK-DAG: xscvsxddp
; CHECK: xxmrghd
; CHECK: blr

; CHECK-LE-LABEL: @test70
; CHECK-LE: mfvsrd
; CHECK-LE: mtvsrwa
; CHECK-LE: mtvsrwa
; CHECK-LE: xscvsxddp
; CHECK-LE: xscvsxddp
; CHECK-LE: xxmrghd
; CHECK-LE: blr
}

; This gets scalarized so the code isn't great
define <2 x i32> @test80(i32 %v) {
  %b1 = insertelement <2 x i32> undef, i32 %v, i32 0
  %b2 = shufflevector <2 x i32> %b1, <2 x i32> undef, <2 x i32> zeroinitializer
  %i = add <2 x i32> %b2, <i32 2, i32 3>
  ret <2 x i32> %i

; CHECK-REG-LABEL: @test80
; CHECK-REG: stw 3, -16(1)
; CHECK-REG: addi [[R1:[0-9]+]], 1, -16
; CHECK-REG: addis [[R2:[0-9]+]]
; CHECK-REG: addi [[R2]], [[R2]]
; CHECK-REG-DAG: lxvw4x [[VS1:[0-9]+]], 0, [[R1]]
; CHECK-REG-DAG: lxvw4x 35, 0, [[R2]]
; CHECK-REG: xxspltw 34, [[VS1]], 0
; CHECK-REG: vadduwm 2, 2, 3
; CHECK-REG-NOT: stxvw4x
; CHECK-REG: blr

; CHECK-FISL-LABEL: @test80
; CHECK-FISL: mr 4, 3
; CHECK-FISL: stw 4, -16(1)
; CHECK-FISL: addi [[R1:[0-9]+]], 1, -16
; CHECK-FISL-DAG: lxvw4x [[VS1:[0-9]+]], 0, [[R1]]
; CHECK-FISL-DAG: xxspltw {{[0-9]+}}, [[VS1]], 0
; CHECK-FISL: addis [[R2:[0-9]+]]
; CHECK-FISL: addi [[R2]], [[R2]]
; CHECK-FISL-DAG: lxvw4x {{[0-9]+}}, 0, [[R2]]
; CHECK-FISL: vadduwm
; CHECK-FISL-NOT: stxvw4x
; CHECK-FISL: blr

; CHECK-LE-LABEL: @test80
; CHECK-LE-DAG: mtvsrd [[R1:[0-9]+]], 3
; CHECK-LE-DAG: xxswapd  [[V1:[0-9]+]], [[R1]]
; CHECK-LE-DAG: addi [[R2:[0-9]+]], {{[0-9]+}}, .LCPI
; CHECK-LE-DAG: lxvd2x [[V2:[0-9]+]], 0, [[R2]]
; CHECK-LE-DAG: xxspltw 34, [[V1]]
; CHECK-LE-DAG: xxswapd 35, [[V2]]
; CHECK-LE: vadduwm 2, 2, 3
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
; CHECK-REG: xscmpudp [[REG:[0-9]+]], 3, 4
; CHECK-REG: beqlr [[REG]]

; CHECK-FISL-LABEL: @test82
; CHECK-FISL: xscmpudp [[REG:[0-9]+]], 3, 4
; CHECK-FISL: beq [[REG]], {{.*}}

; CHECK-LE-LABEL: @test82
; CHECK-LE: xscmpudp [[REG:[0-9]+]], 3, 4
; CHECK-LE: beqlr [[REG]]
}

; Function Attrs: nounwind readnone
define <4 x i32> @test83(i8* %a) {
  entry:
    %0 = tail call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(i8* %a)
      ret <4 x i32> %0
      ; CHECK-LABEL: test83
      ; CHECK: lxvw4x 34, 0, 3
      ; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.lxvw4x.be(i8*)

; Function Attrs: nounwind readnone
define <2 x double> @test84(i8* %a) {
  entry:
    %0 = tail call <2 x double> @llvm.ppc.vsx.lxvd2x.be(i8* %a)
      ret <2 x double> %0
      ; CHECK-LABEL: test84
      ; CHECK: lxvd2x 34, 0, 3
      ; CHECK: blr
}
; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.lxvd2x.be(i8*)

; Function Attrs: nounwind readnone
define void @test85(<4 x i32> %a, i8* %b) {
  entry:
    tail call void @llvm.ppc.vsx.stxvw4x.be(<4 x i32> %a, i8* %b)
    ret void
      ; CHECK-LABEL: test85
      ; CHECK: stxvw4x 34, 0, 5
      ; CHECK: blr
}
; Function Attrs: nounwind readnone
declare void @llvm.ppc.vsx.stxvw4x.be(<4 x i32>, i8*)

; Function Attrs: nounwind readnone
define void @test86(<2 x double> %a, i8* %b) {
  entry:
    tail call void @llvm.ppc.vsx.stxvd2x.be(<2 x double> %a, i8* %b)
    ret void
      ; CHECK-LABEL: test86
      ; CHECK: stxvd2x 34, 0, 5
      ; CHECK: blr
}
; Function Attrs: nounwind readnone
declare void @llvm.ppc.vsx.stxvd2x.be(<2 x double>, i8*)
