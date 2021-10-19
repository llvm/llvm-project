; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

@var = dso_local global i1 0

define dso_local i32 @test_sextloadi32() {
; CHECK-LABEL: test_sextloadi32

  %val = load i1, i1* @var
  %ret = sext i1 %val to i32
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]
; CHECK: {{sbfx x[0-9]+, x[0-9]+, #0, #1|sbfx w[0-9]+, w[0-9]+, #0, #1}}

  ret i32 %ret
; CHECK: ret
}

define dso_local i64 @test_sextloadi64() {
; CHECK-LABEL: test_sextloadi64

  %val = load i1, i1* @var
  %ret = sext i1 %val to i64
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]
; CHECK: {{sbfx x[0-9]+, x[0-9]+, #0, #1}}

  ret i64 %ret
; CHECK: ret
}

define dso_local i32 @test_zextloadi32() {
; CHECK-LABEL: test_zextloadi32

; It's not actually necessary that "ret" is next, but as far as LLVM
; is concerned only 0 or 1 should be loadable so no extension is
; necessary.
  %val = load i1, i1* @var
  %ret = zext i1 %val to i32
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]

  ret i32 %ret
; CHECK-NEXT: ret
}

define dso_local i64 @test_zextloadi64() {
; CHECK-LABEL: test_zextloadi64

; It's not actually necessary that "ret" is next, but as far as LLVM
; is concerned only 0 or 1 should be loadable so no extension is
; necessary.
  %val = load i1, i1* @var
  %ret = zext i1 %val to i64
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var]

  ret i64 %ret
; CHECK-NEXT: ret
}
