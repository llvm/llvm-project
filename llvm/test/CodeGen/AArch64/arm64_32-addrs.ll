; RUN: llc -mtriple=arm64_32-apple-ios %s -o - | FileCheck %s

; If %base < 96 then the sum will not wrap (in an unsigned sense), but "ldr w0,
; [x0, #-96]" would.
define i32 @test_valid_wrap(i32 %base) {
; CHECK-LABEL: test_valid_wrap:
; CHECK: sub w[[ADDR:[0-9]+]], w0, #96
; CHECK: ldr w0, [x[[ADDR]]]

  %newaddr = add nuw i32 %base, -96
  %ptr = inttoptr i32 %newaddr to i32*
  %val = load i32, i32* %ptr
  ret i32 %val
}

define i8 @test_valid_wrap_optimizable(i8* %base) {
; CHECK-LABEL: test_valid_wrap_optimizable:
; CHECK: ldurb w0, [x0, #-96]

  %newaddr = getelementptr inbounds i8, i8* %base, i32 -96
  %val = load i8, i8* %newaddr
  ret i8 %val
}

define i8 @test_valid_wrap_optimizable1(i8* %base, i32 %offset) {
; CHECK-LABEL: test_valid_wrap_optimizable1:
; CHECK: ldrb w0, [x0, w1, sxtw]

  %newaddr = getelementptr inbounds i8, i8* %base, i32 %offset
  %val = load i8, i8* %newaddr
  ret i8 %val
}

;
define i8 @test_valid_wrap_optimizable2(i8* %base, i32 %offset) {
; CHECK-LABEL: test_valid_wrap_optimizable2:
; CHECK: sxtw x[[OFFSET:[0-9]+]], w1
; CHECK: mov w[[BASE:[0-9]+]], #-100
; CHECK: ldrb w0, [x[[OFFSET]], x[[BASE]]]

  %newaddr = getelementptr inbounds i8, i8* inttoptr(i32 -100 to i8*), i32 %offset
  %val = load i8, i8* %newaddr
  ret i8 %val
}
