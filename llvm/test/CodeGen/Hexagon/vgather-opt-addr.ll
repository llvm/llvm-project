; RUN: llc -march=hexagon -O3 -disable-hexagon-amodeopt < %s | FileCheck %s --check-prefix=CHECK-NO-AMODE
; RUN: llc -march=hexagon -O3 < %s | FileCheck %s --check-prefix=CHECK-AMODE

; CHECK-NO-AMODE: [[REG1:(r[0-9]+)]] = add({{r[0-9]+}},#0)

; CHECK-NO-AMODE: [[REG2:(r[0-9]+)]] = add([[REG1]],#128)
; CHECK-NO-AMODE: [[REG3:(r[0-9]+)]] = add([[REG1]],#256)
; CHECK-NO-AMODE: [[REG4:(r[0-9]+)]] = add([[REG1]],#384)
; CHECK-NO-AMODE: [[REG5:(r[0-9]+)]] = add([[REG1]],#512)
; CHECK-NO-AMODE: [[REG6:(r[0-9]+)]] = add([[REG1]],#640)
; CHECK-NO-AMODE: vmem([[REG1]]+#0) = vtmp.new
; CHECK-NO-AMODE: vmem([[REG2]]+#0) = vtmp.new
; CHECK-NO-AMODE: vmem([[REG3]]+#0) = vtmp.new
; CHECK-NO-AMODE: vmem([[REG4]]+#0) = vtmp.new
; CHECK-NO-AMODE: vmem([[REG5]]+#0) = vtmp.new
; CHECK-NO-AMODE: vmem([[REG6]]+#0) = vtmp.new


; CHECK-AMODE: [[REG1:(r[0-9]+)]] = add({{r[0-9]+}},#0)
; CHECK-AMODE-NOT: {{r[0-9]+}} = add([[REG1]],{{[0-9]+}})
; CHECK-AMODE: vmem([[REG1]]+#0) = vtmp.new
; CHECK-AMODE: vmem([[REG1]]+#1) = vtmp.new
; CHECK-AMODE: vmem([[REG1]]+#2) = vtmp.new
; CHECK-AMODE: vmem([[REG1]]+#3) = vtmp.new
; CHECK-AMODE: vmem([[REG1]]+#4) = vtmp.new
; CHECK-AMODE: vmem([[REG1]]+#5) = vtmp.new

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nounwind readnone
define dso_local void @contiguos_vgather_test(i32 %Rb, i32 %mu, i32 %nloops, <32 x i32> %Vv, <64 x i32> %Vvv, <32 x i32> %Qs) local_unnamed_addr #0 {
entry:
  %Vout1 = alloca <32 x i32>, align 128
  %0 = bitcast <32 x i32>* %Vout1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 128, i8* nonnull %0) #2
  %cmp23 = icmp sgt i32 %nloops, 0
  br i1 %cmp23, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr = getelementptr inbounds <32 x i32>, <32 x i32>* %Vout1, i32 1
  %1 = bitcast <32 x i32>* %add.ptr to i8*
  %add.ptr1 = getelementptr inbounds <32 x i32>, <32 x i32>* %Vout1, i32 2
  %2 = bitcast <32 x i32>* %add.ptr1 to i8*
  %add.ptr2 = getelementptr inbounds <32 x i32>, <32 x i32>* %Vout1, i32 3
  %3 = bitcast <32 x i32>* %add.ptr2 to i8*
  %4 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %Qs, i32 -1)
  %add.ptr3 = getelementptr inbounds <32 x i32>, <32 x i32>* %Vout1, i32 4
  %5 = bitcast <32 x i32>* %add.ptr3 to i8*
  %add.ptr4 = getelementptr inbounds <32 x i32>, <32 x i32>* %Vout1, i32 5
  %6 = bitcast <32 x i32>* %add.ptr4 to i8*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  call void @llvm.lifetime.end.p0i8(i64 128, i8* nonnull %0) #2
  ret void

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.024 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  call void @llvm.hexagon.V6.vgathermh.128B(i8* nonnull %0, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermw.128B(i8* nonnull %1, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermhw.128B(i8* nonnull %2, i32 %Rb, i32 %mu, <64 x i32> %Vvv)
  call void @llvm.hexagon.V6.vgathermhq.128B(i8* nonnull %3, <128 x i1> %4, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermwq.128B(i8* nonnull %5, <128 x i1> %4, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermhwq.128B(i8* nonnull %6, <128 x i1> %4, i32 %Rb, i32 %mu, <64 x i32> %Vvv)
  %inc = add nuw nsw i32 %i.024, 1
  %exitcond = icmp eq i32 %inc, %nloops
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermh.128B(i8*, i32, i32, <32 x i32>) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermw.128B(i8*, i32, i32, <32 x i32>) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermhw.128B(i8*, i32, i32, <64 x i32>) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermhq.128B(i8*, <128 x i1>, i32, i32, <32 x i32>) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermwq.128B(i8*, <128 x i1>, i32, i32, <32 x i32>) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermhwq.128B(i8*, <128 x i1>, i32, i32, <64 x i32>) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv65" "target-features"="+hvx-length128b,+hvxv65,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
