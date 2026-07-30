; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -systemz-subreg-liveness < %s | FileCheck %s

; Check for successful compilation.
; CHECK: aghi %r15, -160

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

%0 = type { ptr, i32, i32 }

declare ptr @Perl_sv_grow(ptr, i64) #0

; Function Attrs: nounwind
define signext i32 @Perl_yylex() #1 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %tmp = phi ptr [ %tmp8, %bb3 ], [ undef, %bb ]
  %tmp2 = icmp eq i8 undef, 0
  br i1 %tmp2, label %bb9, label %bb3

bb3:                                              ; preds = %bb1
  %tmp4 = ptrtoint ptr %tmp to i64
  %tmp5 = sub i64 %tmp4, 0
  %tmp6 = shl i64 %tmp5, 32
  %tmp7 = ashr exact i64 %tmp6, 32
  %tmp8 = getelementptr inbounds i8, ptr null, i64 %tmp7
  br label %bb1

bb9:                                              ; preds = %bb1
  br i1 undef, label %bb10, label %bb15

bb10:                                             ; preds = %bb9
  %tmp11 = ptrtoint ptr %tmp to i64
  %tmp12 = sub i64 %tmp11, 0
  %tmp13 = call ptr @Perl_sv_grow(ptr nonnull undef, i64 undef) #2
  %tmp14 = getelementptr inbounds i8, ptr %tmp13, i64 %tmp12
  br label %bb15

bb15:                                             ; preds = %bb10, %bb9
  %tmp16 = phi ptr [ %tmp14, %bb10 ], [ %tmp, %bb9 ]
  %tmp17 = call ptr @Perl_uvuni_to_utf8(ptr %tmp16, i64 undef) #2
  unreachable
}

declare ptr @Perl_uvuni_to_utf8(ptr, i64) #0

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
