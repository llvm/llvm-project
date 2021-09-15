; RUN: llc -mtriple x86_64-apple-macosx12.0.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx11.9.0 %s -o - | FileCheck %s --check-prefix=CHECK-DYNAMIC

; CHECK-STATIC-LABEL: foo:
; CHECK-STATIC: btsq $60, %rbp

; CHECK-DYNAMIC-LABEL: foo:
; CHECK-DYNAMIC: orq _swift_async_extendedFramePointerFlags@GOTPCREL(%rip), %rbp

define void @foo(i8* swiftasync) "frame-pointer"="all" {
  ret void
}
