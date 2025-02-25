; RUN: llc -mtriple=hexagon < %s | FileCheck %s

%struct.RESULTS_S.A = type { i16, i16, i16, [4 x ptr], i32, i32, i32, ptr, %struct.MAT_PARAMS_S.D, i16, i16, i16, i16, i16, %struct.CORE_PORTABLE_S.E }
%struct.list_head_s.B = type { ptr, ptr }
%struct.list_data_s.C = type { i16, i16 }
%struct.MAT_PARAMS_S.D = type { i32, ptr, ptr, ptr }
%struct.CORE_PORTABLE_S.E = type { i8 }

; Test that we don't generate a zero extend in this case. Instead we generate
; a single sign extend instead of two zero extends.

; CHECK-NOT: zxth

; Function Attrs: nounwind
define void @core_bench_list(ptr %res) #0 {
entry:
  %seed3 = getelementptr inbounds %struct.RESULTS_S.A, ptr %res, i32 0, i32 2
  %0 = load i16, ptr %seed3, align 2
  %cmp364 = icmp sgt i16 %0, 0
  br i1 %cmp364, label %for.body, label %while.body19.i160

for.body:
  %i.0370 = phi i16 [ %inc50, %if.then ], [ 0, %entry ]
  br i1 undef, label %if.then, label %while.body.i273

while.body.i273:
  %tobool.i272 = icmp eq ptr undef, null
  br i1 %tobool.i272, label %if.then, label %while.body.i273

if.then:
  %inc50 = add i16 %i.0370, 1
  %exitcond = icmp eq i16 %inc50, %0
  br i1 %exitcond, label %while.body19.i160, label %for.body

while.body19.i160:
  br label %while.body19.i160
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

