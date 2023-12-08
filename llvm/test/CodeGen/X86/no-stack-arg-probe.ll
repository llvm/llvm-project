; This test is attempting to detect that the compiler disables stack
; probe calls when the corresponding option is specified.
;
; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

define i32 @test1() "no-stack-arg-probe" {
  %buffer = alloca [4095 x i8]

  ret i32 0

; CHECK-LABEL: _test1:
; CHECK-NOT: movl $4095, %eax
; CHECK: subl $4095, %esp
; CHECK-NOT: calll __chkstk
}

define i32 @test2() "no-stack-arg-probe" {
  %buffer = alloca [4096 x i8]

  ret i32 0

; CHECK-LABEL: _test2:
; CHECK-NOT: movl $4096, %eax
; CHECK: subl $4096, %esp
; CHECK-NOT: calll __chkstk
}

define i32 @test3(i32 %size) "no-stack-arg-probe" {
  %buffer = alloca i8, i32 %size

  ret i32 0

; CHECK-LABEL: _test3:
; CHECK: subl {{.*}}, %esp
; CHECK-NOT: calll __chkstk
}
