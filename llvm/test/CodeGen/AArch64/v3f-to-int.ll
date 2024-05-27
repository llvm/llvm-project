; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s

; CHECK-LABEL: convert_v3f32
; CHECK: strb
; CHECK: strh
define void @convert_v3f32() {
entry:
  br label %bb

bb:
  %0 = shufflevector <4 x float> zeroinitializer, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %1 = fmul reassoc nnan ninf nsz contract afn <3 x float> %0, <float 2.550000e+02, float 2.550000e+02, float 2.550000e+02>
  %2 = fptoui <3 x float> %1 to <3 x i8>
  %3 = bitcast ptr undef to ptr
  store <3 x i8> %2, ptr %3, align 1
  ret void
}
