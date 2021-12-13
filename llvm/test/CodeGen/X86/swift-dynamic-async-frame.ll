; RUN: llc -mtriple x86_64-apple-macosx12.0.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx12.0.0 -swift-async-fp=always %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx12.0.0 -swift-async-fp=auto %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx11.9.0 -swift-async-fp=always %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx11.9.0 %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx11.9.0 -swift-async-fp=auto %s -o - | FileCheck %s --check-prefix=CHECK-DYNAMIC
; RUN: llc -mtriple x86_64-apple-macosx11.9.0 -swift-async-fp=always %s -o - | FileCheck %s --check-prefix=CHECK-STATIC
; RUN: llc -mtriple x86_64-apple-macosx11.9.0 -swift-async-fp=never %s -o - | FileCheck %s --check-prefix=CHECK-NEVER

; CHECK-STATIC-LABEL: foo:
; CHECK-STATIC: btsq $60, %rbp

; CHECK-DYNAMIC-LABEL: foo:
; CHECK-DYNAMIC: orq _swift_async_extendedFramePointerFlags@GOTPCREL(%rip), %rbp
; CHECK-DYNAMIC: .weak_reference _swift_async_extendedFramePointerFlags

; CHECK-NEVER-LABEL: foo:
; CHECK-NEVER-NOT: btsq $60, %rbp
; CHECK-NEVER-NOT: _swift_async_extendedFramePointerFlags

define void @foo(i8* swiftasync) "frame-pointer"="all" {
  ret void
}
