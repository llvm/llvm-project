; There should not be any mismatch for xqf with this testcase.

; RUN: llc -O3 -mv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp -enable-xqf-gen=true \
; RUN: -mtriple=hexagon -hexagon-qfloat-mode=lossy -enable-postra-xqf-check < %s -o - | FileCheck %s

; CHECK-NOT: Mismatch

declare void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1>, ptr, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1

define tailcc void @widget(ptr %arg) {
bb:
  %load = load i32, ptr %arg, align 4
  %getelementptr = getelementptr i8, ptr null, i32 %load
  br label %bb1

bb1:                                              ; preds = %bb67, %bb
  %phi = phi i32 [ 0, %bb ], [ 1, %bb67 ]
  %call = tail call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>)
  br label %bb50

bb2:                                              ; preds = %bb50
  %call3 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1056964608)
  %call4 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1060439283)
  %call5 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -2139095041)
  %call6 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1065353216)
  %call7 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 255)
  %call8 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -2147483648)
  %call9 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 2147483647)
  %call10 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 0)
  %call11 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 2139095040)
  %call12 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %call59, <32 x i32> %call9)
  %call13 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %call10, <32 x i32> zeroinitializer)
  %call14 = tail call <128 x i1> @llvm.hexagon.V6.veqw.128B(<32 x i32> %call12, <32 x i32> %call10)
  %call15 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <32 x i32> zeroinitializer)
  %call16 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %call14, <32 x i32> %call11, <32 x i32> %call15)
  %call17 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %call6, <32 x i32> %call16)
  %call18 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>)
  %call19 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %call17, <32 x i32> %call18)
  %call20 = tail call <128 x i1> @llvm.hexagon.V6.pred.or.128B(<128 x i1> zeroinitializer, <128 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %call21 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %call58, <32 x i32> %call5)
  %call22 = tail call <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32> %call21, <32 x i32> %call3)
  %call23 = tail call <128 x i1> @llvm.hexagon.V6.vgtsf.128B(<32 x i32> %call4, <32 x i32> %call22)
  %call24 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %call23, <32 x i32> zeroinitializer, <32 x i32> %call7)
  %call25 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call26 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %call24, <32 x i32> %call8)
  %call27 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> %call25, <32 x i32> %call26)
  %call28 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %call27, <32 x i32> %call13)
  %call29 = tail call <32 x i32> @llvm.hexagon.V6.vasrw.128B(<32 x i32> %call28, i32 31)
  %call30 = tail call <128 x i1> @llvm.hexagon.V6.veqw.128B(<32 x i32> %call29, <32 x i32> zeroinitializer)
  %call31 = tail call <32 x i32> @llvm.hexagon.V6.vasrw.128B(<32 x i32> zeroinitializer, i32 0)
  %call32 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %call31, <32 x i32> zeroinitializer)
  %call33 = tail call <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32> %call32, <32 x i32> zeroinitializer)
  %call34 = tail call <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32> zeroinitializer, <32 x i32> %call33)
  %call35 = tail call <32 x i32> @llvm.hexagon.V6.vmaxw.128B(<32 x i32> %call34, <32 x i32> zeroinitializer)
  %call36 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call37 = tail call <32 x i32> @llvm.hexagon.V6.vasrwv.128B(<32 x i32> %call36, <32 x i32> %call35)
  %call38 = tail call <32 x i32> @llvm.hexagon.V6.vaslwv.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call39 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %call38)
  %call40 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %call39, <32 x i32> %call37)
  %call41 = tail call <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call42 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %call30, <32 x i32> %call40, <32 x i32> %call41)
  %call43 = tail call <128 x i1> @llvm.hexagon.V6.vgtsf.128B(<32 x i32> %call58, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>)
  %call44 = tail call <32 x i32> @llvm.hexagon.V6.vaddwq.128B(<128 x i1> %call43, <32 x i32> %call42, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>)
  %call45 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call46 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %call20, <32 x i32> %call45, <32 x i32> %call44)
  %call47 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> %call19, <32 x i32> %call46)
  %icmp = icmp sgt i32 0, 0
  br i1 %icmp, label %bb48, label %bb67

bb48:                                             ; preds = %bb2
  %load49 = load i32, ptr null, align 4
  br label %bb67

bb50:                                             ; preds = %bb50, %bb1
  %phi51 = phi <64 x i32> [ %call, %bb1 ], [ %call63, %bb50 ]
  %phi52 = phi i32 [ 0, %bb1 ], [ %add64, %bb50 ]
  %load53 = load i32, ptr %arg, align 4
  %add = add i32 %phi, %phi52
  %mul = mul i32 %add, %load53
  %getelementptr54 = getelementptr i16, ptr %getelementptr, i32 %mul
  %load55 = load <32 x i32>, ptr %getelementptr54, align 1
  %call56 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(<32 x i32> %load55, <32 x i32> zeroinitializer)
  %call57 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %phi51)
  %call58 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call56)
  %call59 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> %call57, <32 x i32> %call58)
  %call60 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %phi51)
  %call61 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %call56)
  %call62 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> %call60, <32 x i32> %call61)
  %call63 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %call62, <32 x i32> %call59)
  %add64 = add i32 %phi52, 1
  %load65 = load i32, ptr %arg, align 4
  %icmp66 = icmp slt i32 %phi52, %load65
  br i1 %icmp66, label %bb50, label %bb2

bb67:                                             ; preds = %bb48, %bb2
  tail call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1> zeroinitializer, ptr null, <32 x i32> %call47)
  %call68 = tail call <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1> zeroinitializer, <128 x i1> zeroinitializer)
  %call69 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %call68, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call70 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %call69)
  %call71 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %call70)
  %call72 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %call71)
  %call73 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %call72, <32 x i32> zeroinitializer)
  %call74 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call73)
  tail call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1> zeroinitializer, ptr null, <32 x i32> %call74)
  br label %bb1
}

declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.vgtsf.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.veqw.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaslwv.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.or.128B(<128 x i1>, <128 x i1>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1>, <128 x i1>) #1
declare <32 x i32> @llvm.hexagon.V6.vasrw.128B(<32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vaddwq.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vasrwv.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmaxw.128B(<32 x i32>, <32 x i32>) #1
