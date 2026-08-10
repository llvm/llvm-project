; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mattr=+v8.2a,+fullfp16 -fp-contract=fast  | FileCheck %s

define half @test_FMULADDH_OP1(half %a, half %b, half %c) {
; CHECK-LABEL: test_FMULADDH_OP1:
; CHECK: fmadd    {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %mul = fmul fast half %c, %b
  %add = fadd fast half %mul, %a
  ret half %add
}

define half @test_FMULADDH_OP2(half %a, half %b, half %c) {
; CHECK-LABEL: test_FMULADDH_OP2:
; CHECK: fmadd    {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %mul = fmul fast half %c, %b
  %add = fadd fast half %a, %mul
  ret half %add
}

define half @test_FMULSUBH_OP1(half %a, half %b, half %c) {
; CHECK-LABEL: test_FMULSUBH_OP1:
; CHECK: fnmsub    {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %mul = fmul fast half %c, %b
  %sub = fsub fast half %mul, %a
  ret half %sub
}

define half @test_FMULSUBH_OP2(half %a, half %b, half %c) {
; CHECK-LABEL: test_FMULSUBH_OP2:
; CHECK: fmsub    {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %mul = fmul fast half %c, %b
  %add = fsub fast half %a, %mul
  ret half %add
}

define half @test_FNMULSUBH_OP1(half %a, half %b, half %c) {
; CHECK-LABEL: test_FNMULSUBH_OP1:
; CHECK: fnmadd    {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %mul = fmul fast half %c, %b
  %neg = fsub fast half -0.0, %mul
  %add = fsub fast half %neg, %a
  ret half %add
}

define <4 x half> @test_FMLAv4f16_OP1(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: test_FMLAv4f16_OP1:
; CHECK: fmla    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = fmul fast <4 x half> %c, %b
  %add = fadd fast <4 x half> %mul, %a
  ret <4 x half> %add
}

define <4 x half> @test_FMLAv4f16_OP2(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: test_FMLAv4f16_OP2:
; CHECK: fmla    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = fmul fast <4 x half> %c, %b
  %add = fadd fast <4 x half> %a, %mul
  ret <4 x half> %add
}

define <8 x half> @test_FMLAv8f16_OP1(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: test_FMLAv8f16_OP1:
; CHECK: fmla    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = fmul fast <8 x half> %c, %b
  %add = fadd fast <8 x half> %mul, %a
  ret <8 x half> %add
}

define <8 x half> @test_FMLAv8f16_OP2(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: test_FMLAv8f16_OP2:
; CHECK: fmla    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = fmul fast <8 x half> %c, %b
  %add = fadd fast <8 x half> %a, %mul
  ret <8 x half> %add
}

define <4 x half> @test_FMLAv4i16_indexed_OP1(<4 x half> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_FMLAv4i16_indexed_OP1:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fadd
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fmla    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = mul <4 x i16> %c, %b
  %m = bitcast <4 x i16> %mul to <4 x half>
  %add = fadd fast <4 x half> %m, %a
  ret <4 x half> %add
}

define <4 x half> @test_FMLAv4i16_indexed_OP2(<4 x half> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_FMLAv4i16_indexed_OP2:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fadd
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fmla    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = mul <4 x i16> %c, %b
  %m = bitcast <4 x i16> %mul to <4 x half>
  %add = fadd fast <4 x half> %a, %m
  ret <4 x half> %add
}

define <8 x half> @test_FMLAv8i16_indexed_OP1(<8 x half> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_FMLAv8i16_indexed_OP1:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fadd
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fmla    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = mul <8 x i16> %c, %b
  %m = bitcast <8 x i16> %mul to <8 x half>
  %add = fadd fast <8 x half> %m, %a
  ret <8 x half> %add
}

define <8 x half> @test_FMLAv8i16_indexed_OP2(<8 x half> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_FMLAv8i16_indexed_OP2:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fadd
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fmla    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = mul <8 x i16> %c, %b
  %m = bitcast <8 x i16> %mul to <8 x half>
  %add = fadd fast <8 x half> %a, %m
  ret <8 x half> %add
}

define <4 x half> @test_FMLSv4f16_OP1(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: test_FMLSv4f16_OP1:
; CHECK: fneg    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK: fmla    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = fmul fast <4 x half> %c, %b
  %sub = fsub fast <4 x half> %mul, %a
  ret <4 x half> %sub
}

define <4 x half> @test_FMLSv4f16_OP2(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: test_FMLSv4f16_OP2:
; CHECK: fmls    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = fmul fast <4 x half> %c, %b
  %sub = fsub fast <4 x half> %a, %mul
  ret <4 x half> %sub
}

define <8 x half> @test_FMLSv8f16_OP1(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: test_FMLSv8f16_OP1:
; CHECK: fneg    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: fmla    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = fmul fast <8 x half> %c, %b
  %sub = fsub fast <8 x half> %mul, %a
  ret <8 x half> %sub
}

define <8 x half> @test_FMLSv8f16_OP2(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: test_FMLSv8f16_OP2:
; CHECK: fmls    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = fmul fast <8 x half> %c, %b
  %sub = fsub fast <8 x half> %a, %mul
  ret <8 x half> %sub
}

define <4 x half> @test_FMLSv4i16_indexed_OP2(<4 x half> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_FMLSv4i16_indexed_OP2:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fsub
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fmls    {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %mul = mul <4 x i16> %c, %b
  %m = bitcast <4 x i16> %mul to <4 x half>
  %sub = fsub fast <4 x half> %a, %m
  ret <4 x half> %sub
}

define <8 x half> @test_FMLSv8i16_indexed_OP1(<8 x half> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_FMLSv8i16_indexed_OP1:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fsub
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fneg    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK-FIXME: fmla    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = mul <8 x i16> %c, %b
  %m = bitcast <8 x i16> %mul to <8 x half>
  %sub = fsub fast <8 x half> %m, %a
  ret <8 x half> %sub
}

define <8 x half> @test_FMLSv8i16_indexed_OP2(<8 x half> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_FMLSv8i16_indexed_OP2:
; CHECK-FIXME: Currently LLVM produces inefficient code:
; CHECK: mul
; CHECK: fsub
; CHECK-FIXME: It should instead produce the following instruction:
; CHECK-FIXME: fmls    {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %mul = mul <8 x i16> %c, %b
  %m = bitcast <8 x i16> %mul to <8 x half>
  %sub = fsub fast <8 x half> %a, %m
  ret <8 x half> %sub
}
