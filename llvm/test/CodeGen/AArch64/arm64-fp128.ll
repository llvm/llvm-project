; RUN: llc -mtriple=arm64-linux-gnu -verify-machineinstrs -mcpu=cyclone -aarch64-enable-atomic-cfg-tidy=0 < %s | FileCheck -enable-var-scope %s

@lhs = global fp128 zeroinitializer, align 16
@rhs = global fp128 zeroinitializer, align 16

define fp128 @test_add() {
; CHECK-LABEL: test_add:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  %val = fadd fp128 %lhs, %rhs
; CHECK: bl __addtf3
  ret fp128 %val
}

define fp128 @test_sub() {
; CHECK-LABEL: test_sub:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  %val = fsub fp128 %lhs, %rhs
; CHECK: bl __subtf3
  ret fp128 %val
}

define fp128 @test_mul() {
; CHECK-LABEL: test_mul:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  %val = fmul fp128 %lhs, %rhs
; CHECK: bl __multf3
  ret fp128 %val
}

define fp128 @test_div() {
; CHECK-LABEL: test_div:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  %val = fdiv fp128 %lhs, %rhs
; CHECK: bl __divtf3
  ret fp128 %val
}

@var32 = global i32 0
@var64 = global i64 0

define void @test_fptosi() {
; CHECK-LABEL: test_fptosi:
  %val = load fp128, fp128* @lhs, align 16

  %val32 = fptosi fp128 %val to i32
  store i32 %val32, i32* @var32
; CHECK: bl __fixtfsi

  %val64 = fptosi fp128 %val to i64
  store i64 %val64, i64* @var64
; CHECK: bl __fixtfdi

  ret void
}

define void @test_fptoui() {
; CHECK-LABEL: test_fptoui:
  %val = load fp128, fp128* @lhs, align 16

  %val32 = fptoui fp128 %val to i32
  store i32 %val32, i32* @var32
; CHECK: bl __fixunstfsi

  %val64 = fptoui fp128 %val to i64
  store i64 %val64, i64* @var64
; CHECK: bl __fixunstfdi

  ret void
}

define void @test_sitofp() {
; CHECK-LABEL: test_sitofp:

  %src32 = load i32, i32* @var32
  %val32 = sitofp i32 %src32 to fp128
  store volatile fp128 %val32, fp128* @lhs
; CHECK: bl __floatsitf

  %src64 = load i64, i64* @var64
  %val64 = sitofp i64 %src64 to fp128
  store volatile fp128 %val64, fp128* @lhs
; CHECK: bl __floatditf

  ret void
}

define void @test_uitofp() {
; CHECK-LABEL: test_uitofp:

  %src32 = load i32, i32* @var32
  %val32 = uitofp i32 %src32 to fp128
  store volatile fp128 %val32, fp128* @lhs
; CHECK: bl __floatunsitf

  %src64 = load i64, i64* @var64
  %val64 = uitofp i64 %src64 to fp128
  store volatile fp128 %val64, fp128* @lhs
; CHECK: bl __floatunditf

  ret void
}

define i1 @test_setcc1() {
; CHECK-LABEL: test_setcc1:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

; Technically, everything after the call to __letf2 is redundant, but we'll let
; LLVM have its fun for now.
  %val = fcmp ole fp128 %lhs, %rhs
; CHECK: bl __letf2
; CHECK: cmp w0, #0
; CHECK: cset w0, le

  ret i1 %val
; CHECK: ret
}

define i1 @test_setcc2() {
; CHECK-LABEL: test_setcc2:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  %val = fcmp ugt fp128 %lhs, %rhs
; CHECK: bl      __letf2
; CHECK: cmp     w0, #0
; CHECK: cset    w0, gt

  ret i1 %val
; CHECK: ret
}

define i1 @test_setcc3() {
; CHECK-LABEL: test_setcc3:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  %val = fcmp ueq fp128 %lhs, %rhs
; CHECK: bl __eqtf2
; CHECK: cmp     w0, #0
; CHECK: cset    w19, eq
; CHECK: bl __unordtf2
; CHECK: cmp     w0, #0
; CHECK: cset    w8, ne
; CHECK: orr     w0, w8, w19

  ret i1 %val
; CHECK: ret
}


define i32 @test_br_cc() {
; CHECK-LABEL: test_br_cc:

  %lhs = load fp128, fp128* @lhs, align 16
  %rhs = load fp128, fp128* @rhs, align 16
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:lhs]
; CHECK: ldr q1, [{{x[0-9]+}}, :lo12:rhs]

  ; olt == !uge, which LLVM optimizes this to.
  %cond = fcmp olt fp128 %lhs, %rhs
; CHECK: bl      __lttf2
; CHECK-NEXT: cmp     w0, #0
; CHECK-NEXT: b.ge {{.LBB[0-9]+_[0-9]+}}
  br i1 %cond, label %iftrue, label %iffalse

iftrue:
  ret i32 42
; CHECK-NEXT: %bb.
; CHECK-NEXT: mov w0, #42
; CHECK: ret
iffalse:
  ret i32 29
; CHECK: mov w0, #29
; CHECK: ret
}

define void @test_select(i1 %cond, fp128 %lhs, fp128 %rhs) {
; CHECK-LABEL: test_select:

  %val = select i1 %cond, fp128 %lhs, fp128 %rhs
  store fp128 %val, fp128* @lhs, align 16
; CHECK: tst w0, #0x1
; CHECK-NEXT: b.eq [[IFFALSE:.LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: %bb.
; CHECK-NEXT: mov v[[VAL:[0-9]+]].16b, v0.16b
; CHECK-NEXT: [[IFFALSE]]:
; CHECK: str q[[VAL]], [{{x[0-9]+}}, :lo12:lhs]
  ret void
; CHECK: ret
}

@varfloat = global float 0.0, align 4
@vardouble = global double 0.0, align 8

define void @test_round() {
; CHECK-LABEL: test_round:

  %val = load fp128, fp128* @lhs, align 16

  %float = fptrunc fp128 %val to float
  store float %float, float* @varfloat, align 4
; CHECK: bl __trunctfsf2
; CHECK: str s0, [{{x[0-9]+}}, :lo12:varfloat]

  %double = fptrunc fp128 %val to double
  store double %double, double* @vardouble, align 8
; CHECK: bl __trunctfdf2
; CHECK: str d0, [{{x[0-9]+}}, :lo12:vardouble]

  ret void
}

define void @test_extend() {
; CHECK-LABEL: test_extend:

  %val = load fp128, fp128* @lhs, align 16

  %float = load float, float* @varfloat
  %fromfloat = fpext float %float to fp128
  store volatile fp128 %fromfloat, fp128* @lhs, align 16
; CHECK: bl __extendsftf2
; CHECK: str q0, [{{x[0-9]+}}, :lo12:lhs]

  %double = load double, double* @vardouble
  %fromdouble = fpext double %double to fp128
  store volatile fp128 %fromdouble, fp128* @lhs, align 16
; CHECK: bl __extenddftf2
; CHECK: str q0, [{{x[0-9]+}}, :lo12:lhs]

  ret void
; CHECK: ret
}

define fp128 @test_neg(fp128 %in) {
; CHECK: [[$MINUS0:.LCPI[0-9]+_0]]:
; Make sure the weird hex constant below *is* -0.0
; CHECK-NEXT: fp128 -0

; CHECK-LABEL: test_neg:

  ; Could in principle be optimized to fneg which we can't select, this makes
  ; sure that doesn't happen.
  %ret = fsub fp128 0xL00000000000000008000000000000000, %in
; CHECK: mov v1.16b, v0.16b
; CHECK: ldr q0, [{{x[0-9]+}}, :lo12:[[$MINUS0]]]
; CHECK: bl __subtf3

  ret fp128 %ret
; CHECK: ret
}
