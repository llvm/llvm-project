; Tests the case when an argument of vmpy(qf,qf) after spills/fills
; is an IEEE-754 type and another is qf type. The qf type is converted
; to IEEE type and the opcode is changed to handle two IEEE types
; The converted IEEE types are converted back to qf if there are used
; after the instruction.
; XFAIL: *
; NOTE: XFAIL until Hexagon HVX IEEE→QFloat isel translation is upstreamed; remove XFAIL when that lands.
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true -hexagon-qfloat-mode=strict-ieee -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp -o /dev/null  < %s 2>&1 | FileCheck %s

; CHECK: Instruction:   renamable [[V5:\$v[0-9]+]] = V6_vmpy_qf32 killed renamable [[V14:\$v[0-9]+]], killed renamable [[V4:\$v[0-9]+]]
; CHECK: Property: 1 ,0
; CHECK: Instruction:   renamable [[V9:\$v[0-9]+]] = V6_vmpy_qf32 killed renamable [[V13:\$v[0-9]+]], killed renamable [[V8:\$v[0-9]+]]
; CHECK: Property: 1 ,0
; CHECK: Inserting new instruction before:   [[V4]] = V6_vconv_sf_qf32 killed renamable [[V4]]
; CHECK: Inserting new instruction:   [[V5]] = V6_vmpy_qf32_sf killed renamable [[V14]], killed renamable [[V4]]
; CHECK: Inserting new instruction before:   [[V8]] = V6_vconv_sf_qf32 killed renamable [[V8]]
; CHECK: Inserting new instruction:   [[V9]] = V6_vmpy_qf32_sf killed renamable [[V13]], killed renamable [[V8]]


@.str = private unnamed_addr constant [16 x i8] c"Vector[%d]= %x\0A\00", align 1
@VectorResult = common dso_local global <32 x i32> zeroinitializer, align 128
@ptr = common dso_local local_unnamed_addr global [32768 x i8] zeroinitializer, align 8
@str = private unnamed_addr constant [65 x i8] c"HVX_Vector :  Q6_Vqf32_vmpy_VsfRsf(Q6_V_vsplat_R(0+1),INT32_MIN)\00", align 1
@str.3 = private unnamed_addr constant [58 x i8] c"HVX_Vector :  Q6_Vqf32_vmpy_VsfRsf(Q6_V_vsplat_R(0+1),-1)\00", align 1

declare dso_local void @print_vector(i32 noundef, ptr nocapture noundef readonly) local_unnamed_addr #0

declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0

define dso_local i32 @main() local_unnamed_addr #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.rt.sf.128B(<32 x i32> %0, i32 -2147483648)
  store <32 x i32> %1, ptr @VectorResult, align 128
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %counter.06.i = phi i32 [ %inc.i, %for.body.i ], [ 0, %entry ]
  %pointer.05.i = phi ptr [ %incdec.ptr.i, %for.body.i ], [ @VectorResult, %entry ]
  %incdec.ptr.i = getelementptr inbounds i16, ptr %pointer.05.i, i32 1
  %2 = load i16, ptr %pointer.05.i, align 2
  %conv.i = sext i16 %2 to i32
  %call.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %counter.06.i, i32 noundef %conv.i) #3
  %inc.i = add nuw nsw i32 %counter.06.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, 64
  br i1 %exitcond.not.i, label %print_vector.exit, label %for.body.i

print_vector.exit:                                ; preds = %for.body.i
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.rt.sf.128B(<32 x i32> %0, i32 -1)
  store <32 x i32> %3, ptr @VectorResult, align 128
  %puts2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.3)
  tail call void @print_vector(i32 noundef 128, ptr noundef nonnull @VectorResult)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.rt.sf.128B(<32 x i32>, i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #2

attributes #0 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls,-small-data" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nofree nounwind }
attributes #3 = { nounwind }
