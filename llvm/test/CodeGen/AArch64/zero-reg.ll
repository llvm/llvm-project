; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

@var32 = dso_local global i32 0
@var64 = dso_local global i64 0

define dso_local void @test_zr() {
; CHECK-LABEL: test_zr:

  store i32 0, ptr @var32
; CHECK: str wzr, [{{x[0-9]+}}, {{#?}}:lo12:var32]
  store i64 0, ptr @var64
; CHECK: str xzr, [{{x[0-9]+}}, {{#?}}:lo12:var64]

  ret void
; CHECK: ret
}

define dso_local void @test_sp(i32 %val) {
; CHECK-LABEL: test_sp:

; Important correctness point here is that LLVM doesn't try to use xzr
; as an addressing register: "str w0, [xzr]" is not a valid A64
; instruction (0b11111 in the Rn field would mean "sp").
  %addr = getelementptr i32, ptr null, i64 0
  store i32 %val, ptr %addr
; CHECK: str {{w[0-9]+}}, [{{x[0-9]+|sp}}]

  ret void
; CHECK: ret
}
