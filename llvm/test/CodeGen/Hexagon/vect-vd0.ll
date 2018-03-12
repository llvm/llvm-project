; RUN: llc -march=hexagon < %s | FileCheck %s

; Verify __builtin_HEXAGON_V6_vd0 maps to vxor
; CHECK: v{{[0-9]*}} = vxor(v{{[0-9]*}},v{{[0-9]*}})

@g0 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 %a0, i32* %v0, align 4
  %v1 = call <16 x i32> @llvm.hexagon.V6.vd0()
  store <16 x i32> %v1, <16 x i32>* @g0, align 64
  ret i32 ptrtoint (<16 x i32>* @g0 to i32)
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
