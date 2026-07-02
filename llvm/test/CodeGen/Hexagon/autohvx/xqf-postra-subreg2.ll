; Test passes if we don't generate conversion for both
; of the subregisters since only one is live at the use.
;
; UNSUPPORTED: asserts

; REQUIRES: asserts
; RUN: llc -O2 -mtriple=hexagon -mattr=+hvxv81,+hvx-length128B \
; RUN: -enable-xqf-gen=true -hexagon-qfloat-mode=lossy \
; RUN: -debug-only=handle-qfp -enable-postra-xqf-check < %s 2>&1 -o - | FileCheck %s

; CHECK: [HandleConvertToQfCopies]       Processing Copy:  renamable [[V0:\$v[0-9]+]] = COPY
; CHECK: Inserting new instruction:   [[V0]] = V6_vconv_qf32_sf killed renamable [[V0]]
; CHECK: [HandleConvertToQfCopies]       Processing Copy:  renamable [[V1:\$v[0-9]+]] = COPY
; CHECK: Inserting new instruction:   [[V1]] = V6_vconv_qf32_sf killed renamable [[V1]]
; CHECK: [HandleConvertToQfCopies]       Processing Copy:  renamable [[V2:\$v[0-9]+]] = COPY
; CHECK: Inserting new instruction:   [[V2]] = V6_vconv_qf32_sf killed renamable [[V2]]
; CHECK: [HandleConvertToQfCopies]       Processing Copy:  renamable [[V3:\$v[0-9]+]] = COPY
; CHECK: Inserting new instruction:   [[V3]] = V6_vconv_qf32_sf killed renamable [[V3]]


declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vshufoeh.128B(<32 x i32>, <32 x i32>) #0

define tailcc void @hoge(ptr %arg, ptr %arg1, i1 %arg2, <32 x i32> %arg3, <32 x i32> %arg4, <32 x i32> %arg5) {
bb:
  br label %bb6

bb6:                                              ; preds = %bb51, %bb
  br i1 %arg2, label %bb7, label %bb14

bb7:                                              ; preds = %bb6
  %call = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <32 x i32> zeroinitializer)
  %call8 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %call, <32 x i32> zeroinitializer)
  %call9 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call8)
  %call10 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> %call9, <32 x i32> zeroinitializer)
  %call11 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> %call10)
  %call12 = tail call <64 x i32> @llvm.hexagon.V6.vshufoeh.128B(<32 x i32> %call11, <32 x i32> zeroinitializer)
  %call13 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call12)
  br label %bb51

bb14:                                             ; preds = %bb6
  %load = load ptr, ptr %arg, align 16
  tail call void (i32, i32, ptr, ...) %load(i32 0, i32 0, ptr null, ptr null, i32 0, ptr null, ptr null)
  %call15 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>)
  %call16 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call15)
  %call17 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> %call16)
  %call18 = tail call <64 x i32> @llvm.hexagon.V6.vshufoeh.128B(<32 x i32> %call17, <32 x i32> zeroinitializer)
  %call19 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call18)
  store <32 x i32> %call19, ptr %arg1, align 128
  %call20 = tail call <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32> %call19, <32 x i32> zeroinitializer, i32 0)
  %call21 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call20)
  %load22 = load <32 x i32>, ptr %arg, align 128
  %call23 = tail call <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32> %load22, <32 x i32> zeroinitializer, i32 0)
  %call24 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call23)
  %call25 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32> %call21)
  %call26 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call25)
  %call27 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32> %call24)
  %call28 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call27)
  %call29 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32> %call28, <32 x i32> %call26)
  %call30 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %call29, <32 x i32> zeroinitializer)
  %call31 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call30)
  %call32 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32> %call31, <32 x i32> zeroinitializer)
  %call33 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %call32, <32 x i32> zeroinitializer)
  %call34 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call33)
  %call35 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> %call34, <32 x i32> zeroinitializer)
  %call36 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> %call35)
  %call37 = tail call <64 x i32> @llvm.hexagon.V6.vshufoeh.128B(<32 x i32> %call36, <32 x i32> %arg5)
  %call38 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call37)
  store <32 x i32> zeroinitializer, ptr %arg1, align 128
  %call39 = tail call <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %call38, i32 0)
  %call40 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call39)
  %call41 = tail call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32> %call40)
  %call42 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %call41)
  %call43 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> %call42, <32 x i32> zeroinitializer)
  %call44 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32> %call43, <32 x i32> zeroinitializer)
  %call45 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %call44, <32 x i32> zeroinitializer)
  %call46 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call45)
  %call47 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call46)
  %call48 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> %call47)
  %call49 = tail call <64 x i32> @llvm.hexagon.V6.vshufoeh.128B(<32 x i32> %call48, <32 x i32> zeroinitializer)
  %call50 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %call49)
  br label %bb51

bb51:                                             ; preds = %bb14, %bb7
  %phi = phi <32 x i32> [ %call50, %bb14 ], [ %call13, %bb7 ]
  store <32 x i32> %phi, ptr %arg, align 128
  br label %bb6
}

declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(<32 x i32>) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
