; RUN: llc < %s -mtriple=s390x-linux-gnu -stack-size-section | FileCheck %s

; CHECK-LABEL: func1:
; CHECK: .section .stack_sizes,"",@progbits
; CHECK-NEXT: .quad func1
; CHECK-NEXT: .byte 0
define void @func1(i32, i32) #0 {
  ret void
}

; CHECK-LABEL: func2:
; CHECK: .section .stack_sizes,"",@progbits
; CHECK-NEXT: .quad func2
; CHECK-NEXT: .ascii  "\250\001"
define void @func2(i32, i32) #0 {
  alloca i32, align 4
  alloca i32, align 4
  ret void
}

; CHECK-LABEL: func3:
; CHECK: .section .stack_sizes,"",@progbits
; CHECK-NEXT: .quad func3
; CHECK-NEXT: .ascii  "\250\001"
define void @func3() #0 {
  alloca i32, align 4
  call void @func1(i32 1, i32 2)
  ret void
}

; CHECK-LABEL: dynalloc:
; CHECK-NOT: .section .stack_sizes
define void @dynalloc(i32 %N) #0 {
  alloca i32, i32 %N
  ret void
}

attributes #0 = { "no-frame-pointer-elim"="true" }
