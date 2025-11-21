; RUN: llc -march=hexagon -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b -enable-xqf-gen=true -hexagon-commgep=false -hexagon-qfloat-mode=lossy -hexagon-noopt < %s | FileCheck %s

; CHECK: v{{[0-9]+}}:{{[0-9]+}}.qf16 = v{{[0-9]+}}.f8
; CHECK: v{{[0-9]+}}.hf = v{{[0-9]+}}.qf16
; CHECK: v{{[0-9]+}}.hf = v{{[0-9]+}}.qf16
; CHECK: vmem(r{{[0-9]+}}+#{{[0-9]+}}) = v{{[0-9]+}}
; CHECK: vmem(r{{[0-9]+}}+#{{[0-9]+}}) = v{{[0-9]+}}
; CHECK: v{{[0-9]+}}.qf16 = v{{[0-9]+}}.hf
; CHECK: v{{[0-9]+}}.hf = v{{[0-9]+}}.qf16
; CHECK: v{{[0-9]+}} = v{{[0-9]+}}
; CHECK: v{{[0-9]+}}.qf16 = v{{[0-9]+}}.hf
; CHECK: v{{[0-9]+}}.hf = v{{[0-9]+}}.qf16

@__const.main.input = private unnamed_addr constant <{ [8 x i8], [120 x i8] }> <{ [8 x i8] c"\00\01\02\03\04\05\06\07", [120 x i8] zeroinitializer }>, align 128

define dso_local noundef i32 @main() {
entry:
  %result.i3 = alloca [64 x i16], align 128
  %result.i = alloca [64 x i16], align 128
  %0 = load <32 x i32>, ptr @__const.main.input, align 128
  %1 = tail call <64 x i32> @llvm.hexagon.V6.vconv.qf16.f8.128B(<32 x i32> %0)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  call void @llvm.lifetime.start.p0(i64 128, ptr nonnull %result.i)
  store <32 x i32> %3, ptr %result.i, align 128
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %arrayidx1.phi.i = phi ptr [ %result.i, %entry ], [ %arrayidx1.inc.i, %for.body.i ]
  %i.06.i = phi i32 [ 0, %entry ], [ %inc.i, %for.body.i ]
  %4 = load i16, ptr %arrayidx1.phi.i, align 2
  %conv.i = zext i16 %4 to i32
  %inc.i = add nuw nsw i32 %i.06.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, 64
  %arrayidx1.inc.i = getelementptr i8, ptr %arrayidx1.phi.i, i32 2
  br i1 %exitcond.not.i, label %_Z5printItEvDv32_lPc.exit, label %for.body.i

_Z5printItEvDv32_lPc.exit:                        ; preds = %for.body.i
  %5 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %1)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %5)
  call void @llvm.lifetime.end.p0(i64 128, ptr nonnull %result.i)
  call void @llvm.lifetime.start.p0(i64 128, ptr nonnull %result.i3)
  store <32 x i32> %6, ptr %result.i3, align 128
  br label %for.body.i5

for.body.i5:                                      ; preds = %for.body.i5, %_Z5printItEvDv32_lPc.exit
  %arrayidx1.phi.i6 = phi ptr [ %result.i3, %_Z5printItEvDv32_lPc.exit ], [ %arrayidx1.inc.i12, %for.body.i5 ]
  %i.06.i7 = phi i32 [ 0, %_Z5printItEvDv32_lPc.exit ], [ %inc.i10, %for.body.i5 ]
  %7 = load i16, ptr %arrayidx1.phi.i6, align 2
  %conv.i8 = zext i16 %7 to i32
  %inc.i10 = add nuw nsw i32 %i.06.i7, 1
  %exitcond.not.i11 = icmp eq i32 %inc.i10, 64
  %arrayidx1.inc.i12 = getelementptr i8, ptr %arrayidx1.phi.i6, i32 2
  br i1 %exitcond.not.i11, label %_Z5printItEvDv32_lPc.exit14, label %for.body.i5

_Z5printItEvDv32_lPc.exit14:                      ; preds = %for.body.i5
  call void @llvm.lifetime.end.p0(i64 128, ptr nonnull %result.i3)
  ret i32 0
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none))
declare <64 x i32> @llvm.hexagon.V6.vconv.qf16.f8.128B(<32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>)
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none))
