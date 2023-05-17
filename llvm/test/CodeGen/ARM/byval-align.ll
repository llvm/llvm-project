; RUN: llc -mtriple=thumbv7-apple-ios8.0 %s -o - | FileCheck %s

; This checks that alignments greater than 4 are respected by APCS
; targets. Mostly here to make sure *some* correct code is created after some
; simplifying refactoring; at the time of writing there were no actual APCS
; users of byval alignments > 4, so no real calls for ABI stability.

; "byval align 16" can't fit in any regs with an ptr taking up r0.
define i32 @test_align16(ptr, ptr byval([4 x i32]) align 16 %b) {
; CHECK-LABEL: test_align16:
; CHECK-NOT: sub sp
; CHECK: push {r4, r7, lr}
; CHECK: add r7, sp, #4

; CHECK: ldr r0, [r7, #8]

  call void @bar()
  %val = load i32, ptr %b
  ret i32 %val
}

; byval align 8 can, but we used to incorrectly set r7 here (miscalculating the
; space taken up by arg regs).
define i32 @test_align8(ptr, ptr byval([4 x i32]) align 8 %b) {
; CHECK-LABEL: test_align8:
; CHECK: sub sp, #8
; CHECK: push {r4, r7, lr}
; CHECK: add r7, sp, #4

; CHECK: strd r2, r3, [r7, #8]

; CHECK: ldr r0, [r7, #8]

  call void @bar()
  %val = load i32, ptr %b
  ret i32 %val
}

; "byval align 32" can't fit in regs no matter what: it would be misaligned
; unless the incoming stack was deliberately misaligned.
define i32 @test_align32(ptr, ptr byval([4 x i32]) align 32 %b) {
; CHECK-LABEL: test_align32:
; CHECK-NOT: sub sp
; CHECK: push {r4, r7, lr}
; CHECK: add r7, sp, #4

; CHECK: ldr r0, [r7, #8]

  call void @bar()
  %val = load i32, ptr %b
  ret i32 %val
}

; When passing an object "byval align N", the stack must be at least N-aligned.
define void @test_call_align16() {
; CHECK-LABEL: test_call_align16:
; CHECK: push {r4, r7, lr}
; CHECK: add r7, sp, #4

; CHECK: mov [[TMP:r[0-9]+]], sp
; CHECK: bfc [[TMP]], #0, #4
; CHECK: mov sp, [[TMP]]

; While we're here, make sure the caller also puts it at sp
  ; CHECK: mov r[[BASE:[0-9]+]], sp
  ; CHECK: vst1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]
  call i32 @test_align16(ptr null, ptr byval([4 x i32]) align 16 @var)
  ret void
}

@var = global [4 x i32] zeroinitializer
declare void @bar()
