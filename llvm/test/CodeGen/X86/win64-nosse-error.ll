; RUN: not --crash llc %s -mattr="-sse" 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-macho"

; Function Attrs: noimplicitfloat noinline noredzone nounwind optnone
define void @crash() #0 {
  call void (i32*, ...) @func(i32* null, double undef)
  ret void
}
; CHECK: in function crash void (): Win64 ABI varargs functions require SSE to be enabled
; Function Attrs: noimplicitfloat noredzone
declare void @func(i32*, ...)

attributes #0 = { "target-cpu"="x86-64" "target-features"="-sse"}


