; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: = vmem(r{{[0-9]+}}++#1)

target triple = "hexagon-unknown--elf"

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vzb.128B(<32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32>, <64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32>, <64 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32>, <32 x i32>) #0

define void @fred() #1 {
entry:
  br i1 undef, label %b1, label %call_destructor.exit

b1:                                               ; preds = %entry
  br label %b2

b2:                                               ; preds = %b1, %b2
  %c2.host32.sroa.3.0 = phi <128 x i8> [ %5, %b2 ], [ undef, %b1 ]
  %sobel_halide.s0.x.x = phi i32 [ %17, %b2 ], [ 0, %b1 ]
  %0 = add nsw i32 %sobel_halide.s0.x.x, undef
  %1 = shl i32 %0, 7
  %2 = add nsw i32 %1, 128
  %3 = getelementptr inbounds i8, i8* undef, i32 %2
  %4 = bitcast i8* %3 to <128 x i8>*
  %5 = load <128 x i8>, <128 x i8>* %4, align 128
  %6 = bitcast <128 x i8> %c2.host32.sroa.3.0 to <32 x i32>
  %7 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> %6, i32 1)
  %8 = tail call <64 x i32> @llvm.hexagon.V6.vzb.128B(<32 x i32> %7) #1
  %9 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> undef, <64 x i32> %8) #1
  %10 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> %9, <64 x i32> undef) #1
  %11 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %10)
  %12 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32> undef, <32 x i32> %11) #1
  %13 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %12, <32 x i32> undef)
  %14 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %13) #1
  %15 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %14) #1
  %16 = tail call <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32> %15, <32 x i32> undef) #1
  store <32 x i32> %16, <32 x i32>* undef, align 128
  %17 = add nuw nsw i32 %sobel_halide.s0.x.x, 1
  br label %b2

call_destructor.exit:                             ; preds = %entry
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-double" }
