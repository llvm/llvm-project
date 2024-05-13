; RUN: llc -mtriple=thumbv4t-none--eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V4T
; RUN: llc -mtriple=thumbv6m-none--eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V6M

; CHECK-LABEL: test1
define i32 @test1(ptr %p) {

; Offsets less than 8 can be generated in a single add
; CHECK: adds [[NEWBASE:r[0-9]]], r0, #4
  %1 = getelementptr inbounds i32, ptr %p, i32 1
  %2 = getelementptr inbounds i32, ptr %p, i32 2
  %3 = getelementptr inbounds i32, ptr %p, i32 3
  %4 = getelementptr inbounds i32, ptr %p, i32 4

; CHECK-NEXT: ldm [[NEWBASE]],
  %5 = load i32, ptr %1, align 4
  %6 = load i32, ptr %2, align 4
  %7 = load i32, ptr %3, align 4
  %8 = load i32, ptr %4, align 4

  %9 = add nsw i32 %5, %6
  %10 = add nsw i32 %9, %7
  %11 = add nsw i32 %10, %8
  ret i32 %11
}

; CHECK-LABEL: test2
define i32 @test2(ptr %p) {

; Offsets >=8 require a mov and an add
; CHECK-V4T:  movs [[NEWBASE:r[0-9]]], r0
; CHECK-V6M:  mov [[NEWBASE:r[0-9]]], r0
; CHECK-NEXT: adds [[NEWBASE]], #8
  %1 = getelementptr inbounds i32, ptr %p, i32 2
  %2 = getelementptr inbounds i32, ptr %p, i32 3
  %3 = getelementptr inbounds i32, ptr %p, i32 4
  %4 = getelementptr inbounds i32, ptr %p, i32 5

; CHECK-NEXT: ldm [[NEWBASE]],
  %5 = load i32, ptr %1, align 4
  %6 = load i32, ptr %2, align 4
  %7 = load i32, ptr %3, align 4
  %8 = load i32, ptr %4, align 4

  %9 = add nsw i32 %5, %6
  %10 = add nsw i32 %9, %7
  %11 = add nsw i32 %10, %8
  ret i32 %11
}
