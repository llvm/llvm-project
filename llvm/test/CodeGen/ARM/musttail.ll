; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

; The repro example from https://github.com/llvm/llvm-project/issues/57069#issuecomment-1212754850
; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind sspstrong willreturn memory(none)
define hidden noundef i32 @many_args_callee(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = add nsw i32 %1, %0
  %8 = add nsw i32 %7, %2
  %9 = add nsw i32 %8, %3
  %10 = add nsw i32 %9, %4
  %11 = add nsw i32 %10, %5
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind sspstrong willreturn memory(none)
define hidden noundef i32 @many_args(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #1 {
; CHECK: 	b	many_args_callee
  %7 = musttail call noundef i32 @many_args_callee(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6)
  ret i32 %7
}

; Test with sret
; Function Attrs: optsize
declare dso_local void @sret_callee(ptr dead_on_unwind writable sret({ double, double }) align 8, i16 noundef signext) local_unnamed_addr #1

; Function Attrs: mustprogress optsize
define dso_local void @sret_caller(ptr dead_on_unwind noalias writable sret({ double, double }) align 8 %agg.result, i16 noundef signext %P0) local_unnamed_addr #0 {
entry:
; CHECK: 	b	sret_callee
  musttail call void @sret_callee(ptr dead_on_unwind writable sret({ double, double }) align 8 %agg.result, i16 noundef signext 20391) #2
  ret void
}

%struct.Large = type { [60 x i32] }

; Function Attrs: mustprogress noinline optnone
define dso_local void @large_caller(i64 noundef %0, i64 noundef %1, %struct.Large* noundef byval(%struct.Large) align 4 %2, %struct.Large* noundef byval(%struct.Large) align 4 %3) #0 {
entry:
; CHECK: 	b	large_callee
  musttail call void @large_callee(i64 noundef %0, i64 noundef %1, %struct.Large* noundef byval(%struct.Large) align 4 %2, %struct.Large* noundef byval(%struct.Large) align 4 %3)
  ret void
}

declare dso_local void @large_callee(i64 noundef, i64 noundef, %struct.Large* noundef byval(%struct.Large) align 4, %struct.Large* noundef byval(%struct.Large) align 4) #1