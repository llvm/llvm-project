; RUN: llc -mtriple=thumbv4t-none--eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V4T
; RUN: llc -mtriple=thumbv6m-none--eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V6M

; CHECK-LABEL: foo
define i32 @foo(i32 %z, ...) #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca i32, align 4

  store i32 1, i32* %a, align 4
  store i32 2, i32* %b, align 4
  store i32 3, i32* %c, align 4
  store i32 4, i32* %d, align 4
  store i32 5, i32* %e, align 4
  store i32 6, i32* %f, align 4
  store i32 7, i32* %g, align 4
  store i32 8, i32* %h, align 4

  %0 = load i32* %a, align 4
  %1 = load i32* %b, align 4
  %2 = load i32* %c, align 4
  %3 = load i32* %d, align 4
  %4 = load i32* %e, align 4
  %5 = load i32* %f, align 4
  %6 = load i32* %g, align 4
  %7 = load i32* %h, align 4

  %add  = add nsw i32 %0, %1
  %add4 = add nsw i32 %add, %2
  %add5 = add nsw i32 %add4, %3
  %add6 = add nsw i32 %add5, %4
  %add7 = add nsw i32 %add6, %5
  %add8 = add nsw i32 %add7, %6
  %add9 = add nsw i32 %add8, %7

  %addz = add nsw i32 %add9, %z
  call void @llvm.va_start(i8* null)
  ret i32 %addz

; CHECK:      sub sp, #40
; CHECK-NEXT: add [[BASE:r[0-9]]], sp, #8

; CHECK-V4T:  movs [[NEWBASE:r[0-9]]], [[BASE]]
; CHECK-V6M:  mov [[NEWBASE:r[0-9]]], [[BASE]]
; CHECK-NEXT: adds [[NEWBASE]], #8
; CHECK-NEXT: ldm [[NEWBASE]],
}

declare void @llvm.va_start(i8*) nounwind
