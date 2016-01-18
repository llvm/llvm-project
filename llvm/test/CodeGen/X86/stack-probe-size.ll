; This test is attempting to detect that the compiler correctly generates stack
; probe calls when the size of the local variables exceeds the specified stack
; probe size.
;
; Testing the default value of 4096 bytes makes sense, because the default
; stack probe size equals the page size (4096 bytes for all x86 targets), and
; this is unlikely to change in the future.
;
; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=i686-windows-coreclr < %s | FileCheck %s

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

define i32 @test1() "stack-probe-size"="0" {
  %buffer = alloca [4095 x i8]

  ret i32 0

; CHECK-LABEL: _test1:
; CHECK-NOT: subl $4095, %esp
; CHECK: movl $4095, %eax
; CHECK: calll __chkstk
}

define i32 @test2() {
  %buffer = alloca [4095 x i8]

  ret i32 0

; CHECK-LABEL: _test2:
; CHECK-NOT: movl $4095, %eax
; CHECK: subl $4095, %esp
; CHECK-NOT: calll __chkstk
}

define i32 @test3() "stack-probe-size"="8192" {
  %buffer = alloca [4095 x i8]

  ret i32 0

; CHECK-LABEL: _test3:
; CHECK-NOT: movl $4095, %eax
; CHECK: subl $4095, %esp
; CHECK-NOT: calll __chkstk
}

define i32 @test4() "stack-probe-size"="0" {
  %buffer = alloca [4096 x i8]

  ret i32 0

; CHECK-LABEL: _test4:
; CHECK-NOT: subl $4096, %esp
; CHECK: movl $4096, %eax
; CHECK: calll __chkstk
}

define i32 @test5() {
  %buffer = alloca [4096 x i8]

  ret i32 0

; CHECK-LABEL: _test5:
; CHECK-NOT: subl $4096, %esp
; CHECK: movl $4096, %eax
; CHECK: calll __chkstk
}

define i32 @test6() "stack-probe-size"="8192" {
  %buffer = alloca [4096 x i8]

  ret i32 0

; CGECK-LABEL: _test6:
; CGECK-NOT: movl $4096, %eax
; CGECK: subl $4096, %esp
; CGECK-NOT: calll __chkstk
}
