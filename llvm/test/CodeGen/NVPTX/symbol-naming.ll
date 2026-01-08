; RUN: llc < %s -mtriple=nvptx -mattr=+ptx60 -mcpu=sm_30 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | FileCheck %s
; RUN: %if ptxas-isa-6.0 && ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}
; RUN: %if ptxas-isa-6.0 %{ llc < %s -mtriple=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}

; Verify that the NVPTX target removes invalid symbol names prior to emitting
; PTX.

; CHECK-NOT: .str
; CHECK-NOT: <str>
; CHECK-NOT: another-str
; CHECK-NOT: .function.

; CHECK-DAG: _$_str
; CHECK-DAG: _$_str_$_
; CHECK-DAG: _$_str1
; CHECK-DAG: another_$_str

; CHECK-DAG: _$_function_$_
; CHECK-DAG: _$_function_$_2

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"


@.str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1
@"<str>" = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1
@_$_str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1
@another-str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1


; Function Attrs: nounwind
define internal void @.function.() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  %call2 = call i32 (ptr, ...) @printf(ptr @"<str>")
  ret void
}

; Function Attrs: nounwind
define internal void @_$_function_$_() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @_$_str)
  %call2 = call i32 (ptr, ...) @printf(ptr @another-str)
  ret void
}

; Function Attrs: nounwind
define void @global_function() {
entry:
  call void @.function.()
  call void @_$_function_$_()
  ret void
}

declare i32 @printf(ptr, ...)
