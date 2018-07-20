; RUN: llc -enable-machine-outliner -mtriple=x86_64-apple-darwin < %s | FileCheck %s

@x = global i32 0, align 4

define i32 @check_boundaries() #0 {
  ; CHECK-LABEL: _check_boundaries:
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 0, i32* %2, align 4
  %6 = load i32, i32* %2, align 4
  %7 = icmp ne i32 %6, 0
  br i1 %7, label %9, label %8

  ; CHECK: callq 
  ; CHECK-SAME: [[OFUNC1:OUTLINED_FUNCTION_[0-9]+]]
  ; CHECK: cmpl  $0, -{{[0-9]+}}(%rbp)
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  store i32 4, i32* %5, align 4
  br label %10

  store i32 1, i32* %4, align 4
  br label %10

  %11 = load i32, i32* %2, align 4
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %14, label %13

  ; CHECK: callq
  ; CHECK-SAME: [[OFUNC1]]
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  store i32 4, i32* %5, align 4
  br label %15

  store i32 1, i32* %4, align 4
  br label %15

  ret i32 0
}

define i32 @empty_1() #0 {
  ; CHECK-LABEL: _empty_1:
  ; CHECK-NOT: OUTLINED_FUNCTION
  ret i32 1
}

define i32 @empty_2() #0 {
  ; CHECK-LABEL: _empty_2
  ; CHECK-NOT: OUTLINED_FUNCTION
  ret i32 1
}

define i32 @no_empty_outlining() #0 {
  ; CHECK-LABEL: _no_empty_outlining:
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  ; CHECK-NOT: OUTLINED_FUNCTION
  %2 = call i32 @empty_1() #1
  %3 = call i32 @empty_2() #1
  %4 = call i32 @empty_1() #1
  %5 = call i32 @empty_2() #1
  %6 = call i32 @empty_1() #1
  %7 = call i32 @empty_2() #1
  ret i32 0
}

define i32 @main() #0 {
  ; CHECK-LABEL: _main:
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4

  store i32 0, i32* %1, align 4
  store i32 0, i32* @x, align 4
  ; CHECK: callq
  ; CHECK-SAME: [[OFUNC2:OUTLINED_FUNCTION_[0-9]+]]
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  store i32 4, i32* %5, align 4
  store i32 1, i32* @x, align 4
  call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: callq
  ; CHECK-SAME: [[OFUNC2]]
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  store i32 4, i32* %5, align 4
  ret i32 0
}

attributes #0 = { noredzone nounwind ssp uwtable "no-frame-pointer-elim"="true" }

; CHECK: OUTLINED_FUNCTION_{{[0-9]+}}:
; CHECK-DAG:      movl  $1, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $2, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $3, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $4, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: retq

; CHECK: OUTLINED_FUNCTION_{{[0-9]+}}:
; CHECK-DAG:      movl  $1, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $2, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $3, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $4, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: retq
