; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -systemz-subreg-liveness < %s | FileCheck %s

; Check for successful compilation.
; CHECK: aghi %r15, -160

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

%0 = type { i8*, i32, i32 }

declare i8* @Perl_sv_grow(%0*, i64) #0

; Function Attrs: nounwind
define signext i32 @Perl_yylex() #1 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %tmp = phi i8* [ %tmp8, %bb3 ], [ undef, %bb ]
  %tmp2 = icmp eq i8 undef, 0
  br i1 %tmp2, label %bb9, label %bb3

bb3:                                              ; preds = %bb1
  %tmp4 = ptrtoint i8* %tmp to i64
  %tmp5 = sub i64 %tmp4, 0
  %tmp6 = shl i64 %tmp5, 32
  %tmp7 = ashr exact i64 %tmp6, 32
  %tmp8 = getelementptr inbounds i8, i8* null, i64 %tmp7
  br label %bb1

bb9:                                              ; preds = %bb1
  br i1 undef, label %bb10, label %bb15

bb10:                                             ; preds = %bb9
  %tmp11 = ptrtoint i8* %tmp to i64
  %tmp12 = sub i64 %tmp11, 0
  %tmp13 = call i8* @Perl_sv_grow(%0* nonnull undef, i64 undef) #2
  %tmp14 = getelementptr inbounds i8, i8* %tmp13, i64 %tmp12
  br label %bb15

bb15:                                             ; preds = %bb10, %bb9
  %tmp16 = phi i8* [ %tmp14, %bb10 ], [ %tmp, %bb9 ]
  %tmp17 = call i8* @Perl_uvuni_to_utf8(i8* %tmp16, i64 undef) #2
  unreachable
}

declare i8* @Perl_uvuni_to_utf8(i8*, i64) #0

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
