; RUN: llc -mtriple=arm-eabi -force-enable-global-merge %s -o - | FileCheck %s

@g_value1 = dso_local local_unnamed_addr global i32 0, align 4
@g_value2 = dso_local local_unnamed_addr global i32 0, align 4
@g_value3 = dso_local local_unnamed_addr global i32 0, align 4
@g_value4 = dso_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @foo1() local_unnamed_addr {
entry:
  %0 = load i32, ptr @g_value1, align 4
  %1 = load i32, ptr @g_value2, align 4
  %2 = load i32, ptr @g_value3, align 4
  %3 = load i32, ptr @g_value4, align 4
  %call = tail call i32 @foo(i32 %0, i32 %1, i32 %2, i32 %3)
  ret i32 %call
}

declare i32 @foo(i32, i32, i32, i32)

; CHECK:      ldr     [[BASE:r[0-9]+]], .LCPI0_0
; CHECK:      ldm     [[BASE]], {[[R0:r[0-9]+]], [[R1:r[0-9]+]], [[R2:r[0-9]+]], [[R3:r[0-9]+]]}
; CHECK:      .LCPI0_0:
; CHECK-NEXT: .long   .L_MergedGlobals
