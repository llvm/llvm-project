; RUN: llc -march=hexagon < %s | FileCheck %s

; After register allocation it is possible to have a spill of a register
; that is only partially defined. That in itself it fine, but creates a
; problem for double vector registers. Stores of such registers are pseudo
; instructions that are expanded into pairs of individual vector stores,
; and in case of a partially defined source, one of the stores may use
; an entirely undefined register.
;
; This testcase used to crash. Make sure we can handle it, and that we
; do generate a store for the defined part of W0:

; CHECK-LABEL: fred:
; CHECK: v[[REG:[0-9]+]] = vsplat
; CHECK: vmem(r29+#6) = v[[REG]]


target triple = "hexagon"

declare void @danny() local_unnamed_addr #0
declare void @sammy() local_unnamed_addr #0
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vlsrh.128B(<32 x i32>, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32>, <64 x i32>) #1

define hidden void @fred() #2 {
b0:
  %v1 = load i32, i32* null, align 4
  %v2 = icmp ult i64 0, 2147483648
  br i1 %v2, label %b3, label %b5

b3:                                               ; preds = %b0
  %v4 = icmp sgt i32 0, -1
  br i1 %v4, label %b6, label %b5

b5:                                               ; preds = %b3, %b0
  ret void

b6:                                               ; preds = %b3
  tail call void @danny()
  br label %b7

b7:                                               ; preds = %b21, %b6
  %v8 = icmp sgt i32 %v1, 0
  %v9 = select i1 %v8, i32 %v1, i32 0
  %v10 = select i1 false, i32 0, i32 %v9
  %v11 = icmp slt i32 %v10, 0
  %v12 = select i1 %v11, i32 %v10, i32 0
  %v13 = icmp slt i32 0, %v12
  br i1 %v13, label %b14, label %b18

b14:                                              ; preds = %b16, %b7
  br i1 false, label %b15, label %b16

b15:                                              ; preds = %b14
  br label %b16

b16:                                              ; preds = %b15, %b14
  %v17 = icmp eq i32 0, %v12
  br i1 %v17, label %b18, label %b14

b18:                                              ; preds = %b16, %b7
  tail call void @danny()
  %v19 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 524296) #0
  %v20 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v19, <32 x i32> %v19)
  br label %b22

b21:                                              ; preds = %b22
  tail call void @sammy() #3
  br label %b7

b22:                                              ; preds = %b22, %b18
  %v23 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> zeroinitializer, <64 x i32> %v20) #0
  %v24 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v23)
  %v25 = tail call <32 x i32> @llvm.hexagon.V6.vlsrh.128B(<32 x i32> %v24, i32 4) #0
  %v26 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> zeroinitializer, <32 x i32> %v25)
  %v27 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer) #0
  %v28 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v26) #0
  %v29 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> zeroinitializer, <32 x i32> %v28) #0
  store <32 x i32> %v27, <32 x i32>* null, align 128
  %v30 = add nsw i32 0, 128
  %v31 = getelementptr inbounds i8, i8* null, i32 %v30
  %v32 = bitcast i8* %v31 to <32 x i32>*
  store <32 x i32> %v29, <32 x i32>* %v32, align 128
  %v33 = icmp eq i32 0, 0
  br i1 %v33, label %b21, label %b22
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "reciprocal-estimates"="none" "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #3 = { nobuiltin nounwind }
