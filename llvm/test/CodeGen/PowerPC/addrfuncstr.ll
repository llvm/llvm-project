; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s

; Verify that a constant with an initializer that may turn into a dynamic
; relocation is not placed in .rodata, but rather in .data.rel.ro.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.x = type { ptr }
%struct._IO_FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }
%struct._IO_marker = type { ptr, ptr, i32 }

@_ZL1y = internal constant %struct.x { ptr @fread }, align 8

; Function Attrs: nounwind
define ptr @_Z3foov() #0 {
entry:
  ret ptr @_ZL1y
}

declare i64 @fread(ptr, i64, i64, ptr) #1

; CHECK: .section .data.rel.ro
; CHECK: .quad fread

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
