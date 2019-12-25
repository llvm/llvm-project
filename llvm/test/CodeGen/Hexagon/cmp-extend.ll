; RUN: llc -march=hexagon < %s | FileCheck %s

%struct.RESULTS_S.A = type { i16, i16, i16, [4 x i8*], i32, i32, i32, %struct.list_head_s.B*, %struct.MAT_PARAMS_S.D, i16, i16, i16, i16, i16, %struct.CORE_PORTABLE_S.E }
%struct.list_head_s.B = type { %struct.list_head_s.B*, %struct.list_data_s.C* }
%struct.list_data_s.C = type { i16, i16 }
%struct.MAT_PARAMS_S.D = type { i32, i16*, i16*, i32* }
%struct.CORE_PORTABLE_S.E = type { i8 }

; Test that we don't generate a zero extend in this case. Instead we generate
; a single sign extend instead of two zero extends.

; CHECK-NOT: zxth

; Function Attrs: nounwind
define void @core_bench_list(%struct.RESULTS_S.A* %res) #0 {
entry:
  %seed3 = getelementptr inbounds %struct.RESULTS_S.A, %struct.RESULTS_S.A* %res, i32 0, i32 2
  %0 = load i16, i16* %seed3, align 2
  %cmp364 = icmp sgt i16 %0, 0
  br i1 %cmp364, label %for.body, label %while.body19.i160

for.body:
  %i.0370 = phi i16 [ %inc50, %if.then ], [ 0, %entry ]
  br i1 undef, label %if.then, label %while.body.i273

while.body.i273:
  %tobool.i272 = icmp eq %struct.list_head_s.B* undef, null
  br i1 %tobool.i272, label %if.then, label %while.body.i273

if.then:
  %inc50 = add i16 %i.0370, 1
  %exitcond = icmp eq i16 %inc50, %0
  br i1 %exitcond, label %while.body19.i160, label %for.body

while.body19.i160:
  br label %while.body19.i160
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

