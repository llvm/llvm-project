; RUN: opt < %s -indvars -S | FileCheck %s
; PR13371: indvars pass incorrectly substitutes 'undef' values
;
; LFTR should not user %undef as the loop counter.
; CHECK: @test
; CHECK-NOT: icmp{{.*}}undef
@.str3 = private constant [6 x i8] c"%lld\0A\00", align 1
declare i32 @printf(i8* noalias nocapture, ...) nounwind
define i64 @test() nounwind {
func_start:
  br label %block9
block9:                                           ; preds = %block9,%func_start
  %undef = phi i64 [ %next_undef, %block9 ], [ undef, %func_start ]
  %iter = phi i64 [ %next_iter, %block9 ], [ 1, %func_start ]
  %next_iter = add nsw i64 %iter, 1
  %0 = tail call i32 (i8*, ...)* @printf(i8* noalias nocapture getelementptr inbounds ([6 x i8]* @.str3, i64 0, i64 0), i64 %next_iter, i64 %undef)
  %next_undef = add nsw i64 %undef, 1
  %_tmp_3 = icmp slt i64 %next_iter, 100
  br i1 %_tmp_3, label %block9, label %exit
exit:                                             ; preds = %block9
  ret i64 0
}
