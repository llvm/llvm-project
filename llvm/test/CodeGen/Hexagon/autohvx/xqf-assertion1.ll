; On v79 and above, checks for Assertion `isImm() && "Wrong MachineOperand accessor"' failed

; RUN: llc -march=hexagon -enable-xqf-gen=true -enable-rem-conv=true \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -o /dev/null < %s
; RUN: llc -march=hexagon -enable-xqf-gen=true -enable-rem-conv=true \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -o /dev/null < %s


@.str.1 = private unnamed_addr constant [66 x i8] c"hvx_ieee_fp_test.c:39 0 && \22ERROR: Failed to acquire HVX unit.\\n\22\00", align 1
@__func__.main = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.2 = private unnamed_addr constant [33 x i8] c"half -3 converted to vhf = %.2f\0A\00", align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"uhalf 32k converted to vhf = %.2f\0A\00", align 1
@str = private unnamed_addr constant [35 x i8] c"ERROR: Failed to acquire HVX unit.\00", align 1

; Function Attrs: nounwind
define dso_local i32 @main(i32 noundef %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #0 {
entry:
  %call = tail call i32 @acquire_vector_unit(i8 noundef zeroext 0) #6
  %tobool.not = icmp eq i32 %call, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @_Assert(ptr noundef nonnull @.str.1, ptr noundef nonnull @__func__.main) #7
  unreachable

if.end:                                           ; preds = %entry
  tail call void @set_double_vector_mode() #6
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 -3)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 32768)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(<32 x i32> %0)
  %bc.i = bitcast <32 x i32> %2 to <64 x half>
  %3 = extractelement <64 x half> %bc.i, i64 0
  %conv = fpext half %3 to double
  %call5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %conv) #6
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.uh.128B(<32 x i32> %1)
  %bc.i9 = bitcast <32 x i32> %4 to <64 x half>
  %5 = extractelement <64 x half> %bc.i9, i64 0
  %conv7 = fpext half %5 to double
  %call8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, double noundef %conv7) #6
  ret i32 0
}

declare dso_local i32 @acquire_vector_unit(i8 noundef zeroext) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare dso_local void @_Assert(ptr noundef, ptr noundef) local_unnamed_addr #3

declare dso_local void @set_double_vector_mode(...) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(<32 x i32>) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.uh.128B(<32 x i32>) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #4

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #5

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-length128b,-long-calls" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-length128b,-long-calls" }
attributes #2 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-length128b,+hvxv79,+v79,-long-calls" }
attributes #3 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-length128b,+hvxv79,+v79,-long-calls" }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nofree nounwind }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 5, !"CG MDInfo", !3}
!3 = !{!4, !5}
!4 = !{!"F", !"no_filename_available", !"", !"", i1 false, !""}
!5 = !{!"C", !"set_double_vector_mode", !"(void)", !"(...)", i1 true, !""}
!6 = !{!"QuIC LLVM Hexagon Clang version 8.8-alpha3 Engineering Release: hexagon-clang-88"}
