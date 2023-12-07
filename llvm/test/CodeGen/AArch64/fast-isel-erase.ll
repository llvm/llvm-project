; RUN: llc -mtriple=arm64-apple-ios -o - %s -fast-isel=1 -O0 | FileCheck %s

; The zext can be folded into the load and removed, but doing so can invalidate
; pointers internal to FastISel and cause a crash so it must be done carefully.
define i32 @test() {
; CHECK-LABEL: test:
; CHECK: ldrh
; CHECK: bl _callee
; CHECK-NOT: uxth

entry:
  store i32 undef, ptr undef, align 4
  %t81 = load i16, ptr undef, align 2
  call void @callee()
  %t82 = zext i16 %t81 to i32
  %t83 = shl i32 %t82, 16
  %t84 = or i32 undef, %t83
  br label %end

end:
  %val = phi i32 [%t84, %entry]
  ret i32 %val
}

declare void @callee()
