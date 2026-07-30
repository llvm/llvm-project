; Test the order of global variables during llvm-link

; RUN: llvm-link %s %S/Inputs/globalorder-2.ll -o %t.bc
; RUN: llvm-dis  -o - %t.bc | FileCheck %s

@var1 = internal global i32 0, align 4
@var2 = internal global i32 0, align 4
@var3 = global ptr @var1, align 4
@var4 = global ptr @var2, align 4

define i32 @foo() {
entry:
  %0 = load ptr, ptr @var3, align 4
  %1 = load i32, ptr %0, align 4
  %2 = load ptr, ptr @var4, align 4
  %3 = load i32, ptr %2, align 4
  %add = add nsw i32 %3, %1
  ret i32 %add
}
; CHECK: @var1 =
; CHECK-NEXT: @var2 =
; CHECK-NEXT: @var3 =
; CHECK-NEXT: @var4 =
; CHECK-NEXT: @var5 =
; CHECK-NEXT: @var6 =
; CHECK-NEXT: @var7 =
; CHECK-NEXT: @var8 =
