; RUN: llc -mtriple arm64-apple-ios15.0.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-ios15.0.0 -swift-async-fp=auto %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-ios15.0.0 -swift-async-fp=never %s -o - | FileCheck %s --check-prefix=CHECK-NEVER
; RUN: llc -mtriple arm64-apple-ios14.9.0 -swift-async-fp=always %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-ios14.9.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-ios14.9.0 -swift-async-fp=auto %s -o - | FileCheck %s --check-prefix=CHECK-DYNAMIC
; RUN: llc -mtriple arm64-apple-tvos15.0.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-tvos14.9.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-macosx12.0.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64-apple-macosx11.9.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64_32-apple-watchos8.0.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64_32-apple-watchos7.9.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple arm64_32-apple-watchos7.9.0 -swift-async-fp=auto %s -o - | FileCheck %s --check-prefix=CHECK-DYNAMIC-32

; CHECK-STATIC-LABEL: foo:
; CHECK-STATIC: orr x29, x29, #0x1000000000000000

; CHECK-NEVER-LABEL: foo:
; CHECK-NEVER-NOT: orr x29, x29, #0x1000000000000000
; CHECK-NEVER-NOT: _swift_async_extendedFramePointerFlags

; CHECK-DYNAMIC-LABEL: foo:
; CHECK-DYNAMIC: adrp x16, _swift_async_extendedFramePointerFlags@GOTPAGE
; CHECK-DYNAMIC: ldr x16, [x16, _swift_async_extendedFramePointerFlags@GOTPAGEOFF]
; CHECK-DYNAMIC: orr x29, x29, x16

; CHECK-DYNAMIC-32-LABEL: foo:
; CHECK-DYNAMIC-32: adrp x16, _swift_async_extendedFramePointerFlags@GOTPAGE
; CHECK-DYNAMIC-32: ldr w16, [x16, _swift_async_extendedFramePointerFlags@GOTPAGEOFF]
; CHECK-DYNAMIC-32: orr x29, x29, x16, lsl #32

define void @foo(i8* swiftasync) "frame-pointer"="all" {
  ret void
}
