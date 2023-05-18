; RUN: llc -march=hexagon -hvc-va-full-stores < %s | FileCheck %s

; Make sure we generate 3 aligned stores.
; CHECK: vmem({{.*}}) =
; CHECK: vmem({{.*}}) =
; CHECK: vmem({{.*}}) =
; CHECK-NOT: vmem

define void @f0(ptr %a0, i32 %a11, <64 x i16> %a22, <64 x i16> %a3) #0 {
b0:
  %v0 = add i32 %a11, 64
  %v1 = getelementptr i16, ptr %a0, i32 %v0
  store <64 x i16> %a22, ptr %v1, align 2
  %v33 = add i32 %a11, 128
  %v44 = getelementptr i16, ptr %a0, i32 %v33
  store <64 x i16> %a3, ptr %v44, align 2
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+hvxv66,+hvx-length128b" }
