; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK

%struct.small_struct = type { i16 }

@a = global %struct.small_struct zeroinitializer, align 2 #0
@b = global [2 x i16] zeroinitializer, align 2 #0

; Function Attrs: noinline
define i16 @foo() #1 {
entry:
  %0 = load i16, ptr @a, align 2
  %1 = load i16, ptr @b, align 2
  %add = add nsw i16 %0, %1
  ret i16 %add
}

attributes #0 = { "toc-data" }
attributes #1 = { noinline }

; CHECK:      .toc
; CHECK-NEXT: .csect a[TD],2
; CHECK-NEXT: .globl    a[TD]                           # @a
; CHECK-NEXT: .align    1
; CHECK-NEXT: .space    2
; CHECK-NEXT: .csect b[TD],2
; CHECK-NEXT: .globl    b[TD]                           # @b
; CHECK-NEXT: .align    1
; CHECK-NEXT: .space    4
