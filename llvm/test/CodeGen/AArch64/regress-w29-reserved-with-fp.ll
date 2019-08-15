; RUN: llc -mtriple=aarch64-none-linux-gnu -frame-pointer=all < %s | FileCheck %s
@var = global i32 0

declare void @bar()

define void @test_w29_reserved() {
; CHECK-LABEL: test_w29_reserved:
; CHECK: mov x29, sp

  %val1 = load volatile i32, i32* @var
  %val2 = load volatile i32, i32* @var
  %val3 = load volatile i32, i32* @var
  %val4 = load volatile i32, i32* @var
  %val5 = load volatile i32, i32* @var
  %val6 = load volatile i32, i32* @var
  %val7 = load volatile i32, i32* @var
  %val8 = load volatile i32, i32* @var
  %val9 = load volatile i32, i32* @var

; CHECK-NOT: ldr w29,

  ; Call to prevent fp-elim that occurs regardless in leaf functions.
  call void @bar()

  store volatile i32 %val1,  i32* @var
  store volatile i32 %val2,  i32* @var
  store volatile i32 %val3,  i32* @var
  store volatile i32 %val4,  i32* @var
  store volatile i32 %val5,  i32* @var
  store volatile i32 %val6,  i32* @var
  store volatile i32 %val7,  i32* @var
  store volatile i32 %val8,  i32* @var
  store volatile i32 %val9,  i32* @var

  ret void
; CHECK: ret
}
