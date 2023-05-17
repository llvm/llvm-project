; RUN: llc -mtriple=aarch64-none-linux-gnu -frame-pointer=all < %s | FileCheck %s
@var = global i32 0

declare void @bar()

define void @test_w29_reserved() {
; CHECK-LABEL: test_w29_reserved:
; CHECK: mov x29, sp

  %val1 = load volatile i32, ptr @var
  %val2 = load volatile i32, ptr @var
  %val3 = load volatile i32, ptr @var
  %val4 = load volatile i32, ptr @var
  %val5 = load volatile i32, ptr @var
  %val6 = load volatile i32, ptr @var
  %val7 = load volatile i32, ptr @var
  %val8 = load volatile i32, ptr @var
  %val9 = load volatile i32, ptr @var

; CHECK-NOT: ldr w29,

  ; Call to prevent fp-elim that occurs regardless in leaf functions.
  call void @bar()

  store volatile i32 %val1,  ptr @var
  store volatile i32 %val2,  ptr @var
  store volatile i32 %val3,  ptr @var
  store volatile i32 %val4,  ptr @var
  store volatile i32 %val5,  ptr @var
  store volatile i32 %val6,  ptr @var
  store volatile i32 %val7,  ptr @var
  store volatile i32 %val8,  ptr @var
  store volatile i32 %val9,  ptr @var

  ret void
; CHECK: ret
}
