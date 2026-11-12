; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8

; rdar://9117613

%struct.mo = type { i32, ptr }
%struct.mo_pops = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.ui = type { ptr, ptr, i32, ptr, ptr, i64, ptr, ptr, ptr }


define internal fastcc i32 @t(ptr %vp, i32 %withfsize, i64 %filesize) nounwind {
entry:
  br i1 undef, label %bb1, label %bb

bb:                                               ; preds = %entry
  unreachable

bb1:                                              ; preds = %entry
  %0 = call ptr @vn_pp_to_ui(ptr undef) nounwind
  call void @llvm.memset.p0.i32(ptr align 4 undef, i8 0, i32 40, i1 false)
  store ptr undef, ptr %0, align 4
  %1 = getelementptr inbounds %struct.ui, ptr %0, i32 0, i32 5
  %2 = load i64, ptr %1, align 4
  %3 = call i32 @mo_create_nnm(ptr undef, i64 %2, ptr undef) nounwind
  br i1 undef, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  unreachable

bb3:                                              ; preds = %bb1
  br i1 undef, label %bb4, label %bb6

bb4:                                              ; preds = %bb3
  %4 = call i32 @vn_size(ptr %vp, ptr %1, ptr undef) nounwind
  unreachable

bb6:                                              ; preds = %bb3
  ret i32 0
}

declare ptr @vn_pp_to_ui(ptr)

declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) nounwind

declare i32 @mo_create_nnm(ptr, i64, ptr)

declare i32 @vn_size(ptr, ptr, ptr)
