; RUN: llc -verify-machineinstrs -enable-machine-outliner %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -aarch64-redzone %s -o - | FileCheck %s -check-prefix=REDZONE

; Ensure that the MachineOutliner does not fire on functions which use a
; redzone. We don't care about what's actually outlined here. We just want to
; force behaviour in the outliner to make sure that it never acts on anything
; that might have a redzone.
target triple = "arm64----"

@x = common global i32 0, align 4
declare void @baz() #0

; In AArch64FrameLowering, there are a couple special early exit cases where we
; *know* we don't use a redzone. The GHC calling convention is one of these
; cases. Make sure that we know we don't have a redzone even in these cases.
define cc 10 void @bar() #0 {
  ; CHECK-LABEL: bar
  ; CHECK: bl OUTLINED_FUNCTION
  ; REDZONE-LABEL: bar
  ; REDZONE: bl OUTLINED_FUNCTION
  %1 = load i32, ptr @x, align 4
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @x, align 4
  call void @baz()
  %3 = load i32, ptr @x, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @x, align 4
  call void @baz()
  %5 = load i32, ptr @x, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr @x, align 4
  ret void
}

; foo() should have a redzone when compiled with -aarch64-redzone, and no
; redzone otherwise.
define void @foo() #0 {
  ; CHECK-LABEL: foo
  ; CHECK: bl OUTLINED_FUNCTION
  ; REDZONE-LABEL: foo
  ; REDZONE-NOT: bl OUTLINED_FUNCTION
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %4, align 4
  %5 = load i32, ptr %1, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %1, align 4
  %7 = load i32, ptr %3, align 4
  %8 = add nsw i32 %7, 1
  store i32 %8, ptr %3, align 4
  %9 = load i32, ptr %4, align 4
  %10 = add nsw i32 %9, 1
  store i32 %10, ptr %4, align 4
  %11 = load i32, ptr %2, align 4
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr %2, align 4
  %13 = load i32, ptr %1, align 4
  %14 = add nsw i32 %13, 1
  store i32 %14, ptr %1, align 4
  %15 = load i32, ptr %3, align 4
  %16 = add nsw i32 %15, 1
  store i32 %16, ptr %3, align 4
  %17 = load i32, ptr %4, align 4
  %18 = add nsw i32 %17, 1
  store i32 %18, ptr %4, align 4
  %19 = load i32, ptr %2, align 4
  %20 = add nsw i32 %19, -1
  store i32 %20, ptr %2, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone }
