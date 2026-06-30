; Tests whether convert instruction hf=qf is handled correctly.
; The live range of qf register goes beyond the convert and is used
; by a qf instruction.

; REQUIRES: asserts
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 \
; RUN: -debug-only=handle-qfp < %s 2>&1 -o /dev/null | FileCheck %s --check-prefix=V79
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 \
; RUN: -debug-only=handle-qfp < %s 2>&1 -o /dev/null | FileCheck %s --check-prefix=V81

; V79: Analyzing convert instruction:   renamable [[VREG1:\$v[0-9]+]] = V6_vconv_hf_qf16 renamable [[VREG2:\$v[0-9]+]]
; V79: Inserting new instruction:  [[VREG3:\$v[0-9]+]] = V6_vd0
; V79: Inserting new instruction:  [[VREG2]] = V6_vadd_hf killed renamable [[VREG2]], killed [[VREG3]]
; V79: Inserting after conv:  [[VREG2]] = V6_vconv_hf_qf16 killed renamable [[VREG2]]

; V81: Analyzing convert instruction:   renamable [[VREG1:\$v[0-9]+]] = V6_vconv_hf_qf16 renamable [[VREG2:\$v[0-9]+]]
; V81: Inserting new instruction:  [[VREG2]] = V6_vconv_qf16_hf killed renamable [[VREG2]]
; V81: Inserting after conv:  [[VREG2]] = V6_vconv_hf_qf16 killed renamable [[VREG2]]

@.str.1 = private unnamed_addr constant [9 x i8] c"0x%08lx \00", align 1
@.str.3 = private unnamed_addr constant [62 x i8] c"qfloat_test.c:135 0 && \22ERROR: Failed to acquire HVX unit.\\n\22\00", align 1
@__func__.main = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.4 = private unnamed_addr constant [44 x i8] c"The sum of hf   %.3f and hf   %.3f is %.3f\0A\00", align 1
@.str.5 = private unnamed_addr constant [44 x i8] c"The sum of qf16 %.3f and qf16 %.3f is %.3f\0A\00", align 1
@.str.6 = private unnamed_addr constant [44 x i8] c"The sum of qf16 %.3f and hf   %.3f is %.3f\0A\00", align 1
@.str.7 = private unnamed_addr constant [45 x i8] c"The sum of hf   %.3f and hf   -%.3f is %.3f\0A\00", align 1
@.str.8 = private unnamed_addr constant [45 x i8] c"The sum of qf16 %.3f and qf16 -%.3f is %.3f\0A\00", align 1
@.str.9 = private unnamed_addr constant [45 x i8] c"The sum of qf16 %.3f and hf   -%.3f is %.3f\0A\00", align 1
@str = private unnamed_addr constant [35 x i8] c"ERROR: Failed to acquire HVX unit.\00", align 1

; Function Attrs: nofree nounwind
define dso_local void @print_vector_words(<32 x i32> noundef %x) local_unnamed_addr #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end
  %putchar = tail call i32 @putchar(i32 10)
  ret void

for.body:                                         ; preds = %entry, %if.end
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %rem = and i32 %i.07, 7
  %tobool.not = icmp eq i32 %rem, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %putchar6 = tail call i32 @putchar(i32 10)
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %vecext = extractelement <32 x i32> %x, i32 %i.07
  %call1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %vecext) #6
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond.not = icmp eq i32 %inc, 32
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0

; Function Attrs: nounwind
define dso_local i32 @main(i32 noundef %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #1 {
entry:
  %call = tail call i32 @acquire_vector_unit(i8 noundef zeroext 0) #6
  %tobool.not = icmp eq i32 %call, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @_Assert(ptr noundef nonnull @.str.3, ptr noundef nonnull @__func__.main) #7
  unreachable

if.end:                                           ; preds = %entry
  tail call void @set_double_vector_mode() #6
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 14336)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 13312)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 0)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %2, <32 x i32> %0)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %2, <32 x i32> %1)
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %bc.i = bitcast <32 x i32> %0 to <64 x half>
  %6 = extractelement <64 x half> %bc.i, i64 0
  %conv = fpext half %6 to double
  %bc.i71 = bitcast <32 x i32> %1 to <64 x half>
  %7 = extractelement <64 x half> %bc.i71, i64 0
  %conv12 = fpext half %7 to double
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %5)
  %bc.i.i = bitcast <32 x i32> %8 to <64 x half>
  %9 = extractelement <64 x half> %bc.i.i, i64 0
  %conv14 = fpext half %9 to double
  %call15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %conv, double noundef %conv12, double noundef %conv14) #6
  %10 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.128B(<32 x i32> %3, <32 x i32> %4)
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %3)
  %bc.i.i73 = bitcast <32 x i32> %11 to <64 x half>
  %12 = extractelement <64 x half> %bc.i.i73, i64 0
  %conv17 = fpext half %12 to double
  %13 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %4)
  %bc.i.i75 = bitcast <32 x i32> %13 to <64 x half>
  %14 = extractelement <64 x half> %bc.i.i75, i64 0
  %conv19 = fpext half %14 to double
  %15 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %10)
  %bc.i.i77 = bitcast <32 x i32> %15 to <64 x half>
  %16 = extractelement <64 x half> %bc.i.i77, i64 0
  %conv21 = fpext half %16 to double
  %call22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, double noundef %conv17, double noundef %conv19, double noundef %conv21) #6
  %17 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %3, <32 x i32> %1)
  %18 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %17)
  %bc.i.i83 = bitcast <32 x i32> %18 to <64 x half>
  %19 = extractelement <64 x half> %bc.i.i83, i64 0
  %conv28 = fpext half %19 to double
  %call29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, double noundef %conv17, double noundef %conv12, double noundef %conv28) #6
  %20 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %21 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %20)
  %bc.i.i89 = bitcast <32 x i32> %21 to <64 x half>
  %22 = extractelement <64 x half> %bc.i.i89, i64 0
  %conv35 = fpext half %22 to double
  %call36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, double noundef %conv, double noundef %conv12, double noundef %conv35) #6
  %23 = tail call <32 x i32> @llvm.hexagon.V6.vsub.qf16.128B(<32 x i32> %3, <32 x i32> %4)
  %24 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %23)
  %bc.i.i95 = bitcast <32 x i32> %24 to <64 x half>
  %25 = extractelement <64 x half> %bc.i.i95, i64 0
  %conv42 = fpext half %25 to double
  %call43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, double noundef %conv17, double noundef %conv19, double noundef %conv42) #6
  %26 = tail call <32 x i32> @llvm.hexagon.V6.vsub.qf16.mix.128B(<32 x i32> %3, <32 x i32> %1)
  %27 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %26)
  %bc.i.i101 = bitcast <32 x i32> %27 to <64 x half>
  %28 = extractelement <64 x half> %bc.i.i101, i64 0
  %conv49 = fpext half %28 to double
  %call50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, double noundef %conv17, double noundef %conv12, double noundef %conv49) #6
  ret i32 0
}

declare dso_local i32 @acquire_vector_unit(i8 noundef zeroext) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare dso_local void @_Assert(ptr noundef, ptr noundef) local_unnamed_addr #3

declare dso_local void @set_double_vector_mode(...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.qf16.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vsub.qf16.mix.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32>) #4

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5

attributes #0 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-qfloat,-long-calls,-packets" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-qfloat,-long-calls,-packets" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-qfloat,-long-calls,-packets" }
attributes #3 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-qfloat,-long-calls,-packets" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nofree nounwind }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }
