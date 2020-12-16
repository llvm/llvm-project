; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure we generate 3 aligned stores.
; CHECK: vmem({{.*}}) =
; CHECK: vmem({{.*}}) =
; CHECK: vmem({{.*}}) =
; CHECK-NOT: vmem

define void @f0(i16* %a0, i32 %a11, <64 x i16> %a22, <64 x i16> %a3) #0 {
b0:
  %v0 = add i32 %a11, 64
  %v1 = getelementptr i16, i16* %a0, i32 %v0
  %v2 = bitcast i16* %v1 to <64 x i16>*
  store <64 x i16> %a22, <64 x i16>* %v2, align 2
  %v33 = add i32 %a11, 128
  %v44 = getelementptr i16, i16* %a0, i32 %v33
  %v5 = bitcast i16* %v44 to <64 x i16>*
  store <64 x i16> %a3, <64 x i16>* %v5, align 2
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+hvx,+hvx-length128b" }
