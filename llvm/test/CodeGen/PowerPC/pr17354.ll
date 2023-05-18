; RUN: llc -verify-machineinstrs -mcpu=pwr7 -relocation-model=pic <%s | FileCheck %s

; Test that PR17354 is fixed.  We must generate a nop following even
; local calls when generating code for shared libraries, to permit
; TOC fixup.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.CS = type { i32 }

@_ZL3glb = internal global [1 x %struct.CS] zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_a, ptr null }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_Z4funcv(ptr sret(%struct.CS) @_ZL3glb)
  ret void
}

; CHECK-LABEL: __cxx_global_var_init:
; CHECK: bl _Z4funcv
; CHECK-NEXT: nop

; Function Attrs: nounwind
define void @_Z4funcv(ptr noalias sret(%struct.CS) %agg.result) #0 {
entry:
  store i32 0, ptr %agg.result, align 4
  ret void
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
