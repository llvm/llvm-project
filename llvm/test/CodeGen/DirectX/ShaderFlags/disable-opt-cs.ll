; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s


; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000001

; CHECK: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;       Disable shader optimizations

; CHECK: ; Shader Flags for Module Functions
; CHECK: ; Function main : 0x00000001
; The test source in this file generated from the following command:
; clang -cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -O0 -o - <<EOF
; [numthreads(1,1,1)]
; [shader("compute")]
; void main() {}
; EOF

target triple = "dxilv1.0-pc-shadermodel6.0-compute"

; Function Attrs: convergent noinline norecurse optnone
define void @main() #0 {
entry:
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define noundef i32 @_Z3foov() #1 {
entry:
  ret i32 0
}

attributes #0 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { alwaysinline convergent mustprogress norecurse nounwind "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
