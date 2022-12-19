; RUN: llc < %s -march=nvptx -mattr=+ptx60 -mcpu=sm_30 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}

; Verify that the NVPTX target removes invalid symbol names prior to emitting
; PTX.

; CHECK-NOT: .str
; CHECK-NOT: .function.

; CHECK-DAG: _$_str
; CHECK-DAG: _$_str1

; CHECK-DAG: _$_function_$_
; CHECK-DAG: _$_function_$_2

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"


@.str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1
@_$_str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1


; Function Attrs: nounwind
define internal void @.function.() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  ret void
}

; Function Attrs: nounwind
define internal void @_$_function_$_() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @_$_str)
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
