; Tests whether a register is found from the pool of non-live
; registers. This reg is used to store zeroes for using in converts.

; REQUIRES: asserts
; RUN: llc --mtriple=hexagon-- -O0 -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -debug-only=handle-qfp -o /dev/null < %s 2>&1 | FileCheck %s

; CHECK: Analyzing convert instruction:   renamable [[VREG0:\$v[0-9]+]] = V6_vconv_hf_qf16 killed renamable [[VREG0]]
; CHECK: Using V30 register to store a vector of zeroes
; CHECK: Inserting new instruction:   [[VREG1:\$v[0-9]+]] = V6_vd0
; CHECK: Inserting new instruction:   [[VREG0]] = V6_vadd_hf killed renamable [[VREG0]], killed [[VREG1]]

@.str.1 = private unnamed_addr constant [9 x i8] c"0x%08lx \00", align 1
@.str.3 = private unnamed_addr constant [62 x i8] c"qfloat_test.c:135 0 && \22ERROR: Failed to acquire HVX unit.\\n\22\00", align 1
@__func__.main = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.4 = private unnamed_addr constant [44 x i8] c"The sum of hf   %.3f and hf   %.3f is %.3f\0A\00", align 1
@str = private unnamed_addr constant [35 x i8] c"ERROR: Failed to acquire HVX unit.\00", align 1

; Function Attrs: nofree nounwind optsize
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

; Function Attrs: nofree nounwind optsize
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0

; Function Attrs: nounwind optsize
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
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %bc.i = bitcast <32 x i32> %0 to <64 x half>
  %3 = extractelement <64 x half> %bc.i, i64 0
  %conv = fpext half %3 to double
  %bc.i18 = bitcast <32 x i32> %1 to <64 x half>
  %4 = extractelement <64 x half> %bc.i18, i64 0
  %conv12 = fpext half %4 to double
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  %bc.i.i = bitcast <32 x i32> %5 to <64 x half>
  %6 = extractelement <64 x half> %bc.i.i, i64 0
  %conv14 = fpext half %6 to double
  %call15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %conv, double noundef %conv12, double noundef %conv14) #6
  ret i32 0
}

; Function Attrs: optsize
declare dso_local i32 @acquire_vector_unit(i8 noundef zeroext) local_unnamed_addr #2

; Function Attrs: noreturn nounwind optsize
declare dso_local void @_Assert(ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: optsize
declare dso_local void @set_double_vector_mode(...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32>) #4

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5

attributes #0 = { nofree nounwind optsize "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls" }
attributes #1 = { nounwind optsize "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls" }
attributes #2 = { optsize "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls" }
attributes #3 = { noreturn nounwind optsize "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nofree nounwind }
attributes #6 = { nounwind optsize }
attributes #7 = { noreturn nounwind optsize }
