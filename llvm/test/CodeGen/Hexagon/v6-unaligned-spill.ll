; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we generate an unaligned vector store for a spill when a function
; has an alloca. Also, make sure the addressing mode for unaligned store does
; is not a base+offset with a non-zero offset that is not a multiple of 128.

; CHECK: vmemu(r{{[0-9]+}}+#0)

%s.0 = type { [5 x [4 x i8]], i32, i32, i32, i32 }

; Function Attrs: nounwind
define i32 @f0(i8* nocapture readonly %a0, i8* nocapture %a1, i8* nocapture readonly %a2, i8* nocapture readonly %a3, i32 %a4, i32 %a5, i32 %a6, %s.0* nocapture readonly %a7) #0 {
b0:
  %v0 = alloca i8, i32 %a4, align 128
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  %v1 = icmp sgt i32 %a5, 2
  br label %b2

b2:                                               ; preds = %b3, %b2, %b1
  br i1 undef, label %b3, label %b2

b3:                                               ; preds = %b2
  call void @f1(i8* undef, i8* undef, i8* nonnull %v0, i32 %a4, i32 %a5, %s.0* %a7)
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vd0.128B() #2
  br i1 %v1, label %b4, label %b2

b4:                                               ; preds = %b4, %b3
  %v3 = phi <32 x i32> [ %v5, %b4 ], [ undef, %b3 ]
  %v4 = tail call <32 x i32> @llvm.hexagon.V6.vsubhnq.128B(<1024 x i1> undef, <32 x i32> undef, <32 x i32> %v3) #2
  %v5 = tail call <32 x i32> @llvm.hexagon.V6.vavguh.128B(<32 x i32> %v3, <32 x i32> %v2) #2
  br label %b4

b5:                                               ; preds = %b0
  ret i32 0
}

; Function Attrs: nounwind
declare void @f1(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture, i32, i32, %s.0* nocapture readonly) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vd0.128B() #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubhnq.128B(<1024 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vavguh.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
