; RUN: llc -mtriple=aarch64-none-linux-gnu -frame-pointer=none < %s | \
; RUN:    FileCheck %s --check-prefixes=CHECK,NONE
; RUN: llc -mtriple=aarch64-none-linux-gnu -frame-pointer=reserved < %s | \
; RUN:    FileCheck %s --check-prefixes=CHECK,RESERVED
; RUN: llc -mtriple=aarch64-none-linux-gnu -frame-pointer=all < %s | \
; RUN:    FileCheck %s --check-prefixes=CHECK,ALL

; By default, Darwin and Windows will reserve x29
; RUN: llc -mtriple=aarch64-darwin -frame-pointer=none < %s | \
; RUN:    FileCheck %s --check-prefixes=CHECK,RESERVED
; RUN: llc -mtriple=aarch64-darwin -frame-pointer=none < %s | \
; RUN:    FileCheck %s --check-prefixes=CHECK,RESERVED
@var = global i32 0

declare void @bar()

define void @test_w29_reserved() {
; CHECK-LABEL: test_w29_reserved:
; ALL: add x29, sp
; NONE-NOT: add x29
; NONE-NOT: mov x29
; RESERVED-NOT: add x29
; RESERVED-NOT: mov x29

  %val1 = load volatile i32, ptr @var
  %val2 = load volatile i32, ptr @var
  %val3 = load volatile i32, ptr @var
  %val4 = load volatile i32, ptr @var
  %val5 = load volatile i32, ptr @var
  %val6 = load volatile i32, ptr @var
  %val7 = load volatile i32, ptr @var
  %val8 = load volatile i32, ptr @var
  %val9 = load volatile i32, ptr @var
  %val10 = load volatile i32, ptr @var

; NONE: ldr w29,
; ALL-NOT: ldr w29,
; RESERVED-NOT: ldr w29,

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
  store volatile i32 %val10,  ptr @var

  ret void
; CHECK: ret
}
