; RUN: llc < %s -mcpu=atom -mtriple=i686-linux  | FileCheck -check-prefix=ATOM %s
; RUN: llc < %s -mcpu=core2 -mtriple=i686-linux | FileCheck %s

declare void @use_arr(i8*)
declare void @many_params(i32, i32, i32, i32, i32, i32)

define void @test1() nounwind {
; ATOM: test1:
; ATOM: leal -1052(%esp), %esp
; ATOM-NOT: sub
; ATOM: call
; ATOM: leal 1052(%esp), %esp

; CHECK: test1:
; CHECK: subl
; CHECK: call
; CHECK-NOT: lea
  %arr = alloca [1024 x i8], align 16
  %arr_ptr = getelementptr inbounds [1024 x i8]* %arr, i8 0, i8 0
  call void @use_arr(i8* %arr_ptr)
  ret void
}

define void @test2() nounwind {
; ATOM: test2:
; ATOM: leal -28(%esp), %esp
; ATOM: call
; ATOM: leal 28(%esp), %esp

; CHECK: test2:
; CHECK-NOT: lea
  call void @many_params(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6)
  ret void
}

define void @test3() nounwind {
; ATOM: test3:
; ATOM: leal -8(%esp), %esp
; ATOM: leal 8(%esp), %esp

; CHECK: test3:
; CHECK-NOT: lea
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %x, align 4
  ret void
}

