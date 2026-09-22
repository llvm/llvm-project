; Test that the correct vadd(qf32, sf) is generated instead of
; vadd(sf,sf).
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=strict-ieee -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s

; CHECK-LABEL: main:
; CHECK: [[VREG:v[0-9]+]].qf32 = vmpy({{.*}}.sf,{{.*}}.sf)
; CHECK: v{{.*}}.qf32 = vadd([[VREG]].qf32,v{{.*}}.sf)
; CHECK: v{{.*}}.qf32 = vadd([[VREG]].qf32,v{{.*}}.sf)

@.str.1 = private unnamed_addr constant [9 x i8] c"0x%08lx \00", align 1
@.str.3 = private unnamed_addr constant [99 x i8] c"/prj/qct/llvm/devops/test/users/sgundapa/del/test.c:54 0 && \22ERROR: Failed to acquire HVX unit.\\n\22\00", align 1
@__func__.main = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.5 = private unnamed_addr constant [31 x i8] c"sf mpy of 0.5 and -0.25  = %f\0A\00", align 1
@str = private unnamed_addr constant [35 x i8] c"ERROR: Failed to acquire HVX unit.\00", align 1
@str.6 = private unnamed_addr constant [24 x i8] c"\0Amultiply instructions\0A\00", align 1

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
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1056964608)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -1098907648)
  %puts7 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %bc.i = bitcast <32 x i32> %2 to <32 x float>
  %3 = extractelement <32 x float> %bc.i, i64 0
  %conv = fpext float %3 to double
  %call6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, double noundef %conv) #6
  ret i32 0
}

declare dso_local i32 @acquire_vector_unit(i8 noundef zeroext) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare dso_local void @_Assert(ptr noundef, ptr noundef) local_unnamed_addr #3

declare dso_local void @set_double_vector_mode(...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(<32 x i32>, <32 x i32>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #4

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5

attributes #0 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,-long-calls,-small-data" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,-long-calls,-small-data" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,-long-calls,-small-data" }
attributes #3 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-ieee-fp,+hvx-length128b,+hvx-qfloat,-long-calls,-small-data" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nofree nounwind }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }
