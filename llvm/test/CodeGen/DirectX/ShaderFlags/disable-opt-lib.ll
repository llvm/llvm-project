; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s


; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000001

; CHECK: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;       Disable shader optimizations

; CHECK: ; Shader Flags for Module Functions
; CHECK: ; Function main : 0x00000001
; The test source in this file generated from the following command:
; clang -cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -O0 -o - <<EOF

; [numthreads(1,1,1)]
; [shader("compute")]
; void main() {}

; int foo() {return 0;}
; EOF

target triple = "dxilv1.3-pc-shadermodel6.3-library"

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define internal void @_Z4mainv() #0 {
entry:
  ret void
}

; Function Attrs: convergent noinline norecurse
define void @main() #1 {
entry:
  call void @_Z4mainv()
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define noundef i32 @_Z3foov() #0 {
entry:
  ret i32 0
}

attributes #0 = { convergent mustprogress noinline norecurse nounwind "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }


!llvm.module.flags = !{!0}

!0 = !{i32 4, !"dx.disable_optimizations", i32 1}
