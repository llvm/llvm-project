; RUN: not opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxilv1.3-pc-shadermodel6.3-library"

; All entry functions of a library shader need to either have optnone
; or not have the attribute
; CHECK: error:
; CHECK-SAME: in function entry_two
; CHECK-SAME:  Inconsistent optnone attribute
; Function Attrs: convergent noinline norecurse optnone
define void @entry_one() #0 {
entry:
  ret void
}

; Function Attrs: convergent noinline norecurse
define void @entry_two() #1 {
entry:
  ret void
}

attributes #0 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!dx.valver = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 1, i32 8}
