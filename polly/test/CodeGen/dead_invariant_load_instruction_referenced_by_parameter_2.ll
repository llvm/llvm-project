; RUN: opt %loadNPMPolly -passes=polly-codegen < %s
;
; Check we do not crash even though there is a dead load that is referenced by
; a parameter and we do not pre-load it (as it is dead).
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@REGISTER = external global [10 x i32], align 16

; Function Attrs: nounwind uwtable
define void @FORMAT3_4() #0 {
entry:
  %INSTR = alloca [32 x i32], align 16
  br label %entry.split

entry.split:                                      ; preds = %entry
  %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @REGISTER, i64 0, i64 8), align 16
  %add = add nsw i32 %0, 2
  %cmp = icmp sgt i32 %add, 1048575
  br i1 %cmp, label %if.end.36, label %if.else

if.else:                                          ; preds = %entry.split
  call void (i32, i32, ptr, ...) @BYTES_TO_BITS(i32 undef, i32 1, ptr undef) #2
  %1 = load i32, ptr undef, align 4
  %cmp14 = icmp eq i32 %1, 1
  br i1 %cmp14, label %land.lhs.true, label %if.end.36

land.lhs.true:                                    ; preds = %if.else
  %arrayidx16 = getelementptr inbounds [32 x i32], ptr %INSTR, i64 0, i64 6
  br i1 false, label %land.lhs.true.19, label %if.then.23

land.lhs.true.19:                                 ; preds = %land.lhs.true
  %arrayidx20 = getelementptr inbounds [32 x i32], ptr %INSTR, i64 0, i64 7
  br i1 false, label %if.end.36, label %if.then.23

if.then.23:                                       ; preds = %land.lhs.true.19, %land.lhs.true
  br i1 false, label %if.end.36, label %if.else.28

if.else.28:                                       ; preds = %if.then.23
  br label %if.end.36

if.end.36:                                        ; preds = %if.else.28, %if.then.23, %land.lhs.true.19, %if.else, %entry.split
  %RANGE_ERROR.0 = phi i1 [ false, %land.lhs.true.19 ], [ false, %if.else.28 ], [ false, %if.else ], [ true, %entry.split ], [ true, %if.then.23 ]
  br i1 %RANGE_ERROR.0, label %if.then.37, label %if.end.38

if.then.37:                                       ; preds = %if.end.36
  br label %return

if.end.38:                                        ; preds = %if.end.36
  br i1 undef, label %land.lhs.true.43, label %if.else.50

land.lhs.true.43:                                 ; preds = %if.end.38
  br i1 undef, label %if.then.47, label %if.else.50

if.then.47:                                       ; preds = %land.lhs.true.43
  br label %if.end.107

if.else.50:                                       ; preds = %land.lhs.true.43, %if.end.38
  br i1 undef, label %if.then.53, label %if.else.89

if.then.53:                                       ; preds = %if.else.50
  br i1 undef, label %land.lhs.true.59, label %if.end.64

land.lhs.true.59:                                 ; preds = %if.then.53
  br i1 undef, label %if.then.63, label %if.end.64

if.then.63:                                       ; preds = %land.lhs.true.59
  br label %return

if.end.64:                                        ; preds = %land.lhs.true.59, %if.then.53
  br i1 undef, label %if.then.80, label %if.end.107

if.then.80:                                       ; preds = %if.end.64
  br i1 undef, label %if.then.83, label %if.else.85

if.then.83:                                       ; preds = %if.then.80
  br label %if.end.107

if.else.85:                                       ; preds = %if.then.80
  br label %if.end.107

if.else.89:                                       ; preds = %if.else.50
  br i1 undef, label %if.then.96, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.else.89
  br i1 undef, label %if.then.96, label %if.end.97

if.then.96:                                       ; preds = %lor.lhs.false, %if.else.89
  br label %return

if.end.97:                                        ; preds = %lor.lhs.false
  br i1 undef, label %if.then.103, label %if.end.107

if.then.103:                                      ; preds = %if.end.97
  br label %if.end.107

if.end.107:                                       ; preds = %if.then.103, %if.end.97, %if.else.85, %if.then.83, %if.end.64, %if.then.47
  br i1 undef, label %land.lhs.true.111, label %if.end.142

land.lhs.true.111:                                ; preds = %if.end.107
  br i1 undef, label %if.then.115, label %if.end.142

if.then.115:                                      ; preds = %land.lhs.true.111
  br i1 undef, label %if.then.118, label %return

if.then.118:                                      ; preds = %if.then.115
  br i1 undef, label %if.then.125, label %for.cond.preheader

for.cond.preheader:                               ; preds = %if.then.118
  br i1 undef, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  br label %for.body

if.then.125:                                      ; preds = %if.then.118
  br label %return

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  br i1 undef, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %for.cond.preheader
  br label %return

if.end.142:                                       ; preds = %land.lhs.true.111, %if.end.107
  br i1 undef, label %land.lhs.true.146, label %if.end.206

land.lhs.true.146:                                ; preds = %if.end.142
  br i1 undef, label %if.then.150, label %if.end.206

if.then.150:                                      ; preds = %land.lhs.true.146
  br i1 undef, label %if.then.157, label %lor.lhs.false.153

lor.lhs.false.153:                                ; preds = %if.then.150
  br i1 undef, label %if.then.157, label %if.end.158

if.then.157:                                      ; preds = %lor.lhs.false.153, %if.then.150
  br label %return

if.end.158:                                       ; preds = %lor.lhs.false.153
  br i1 undef, label %if.then.179, label %return

if.then.179:                                      ; preds = %if.end.158
  br i1 undef, label %if.then.183, label %for.cond.185.preheader

for.cond.185.preheader:                           ; preds = %if.then.179
  br i1 undef, label %for.body.188.lr.ph, label %for.end.198

for.body.188.lr.ph:                               ; preds = %for.cond.185.preheader
  br label %for.body.188

if.then.183:                                      ; preds = %if.then.179
  br label %return

for.body.188:                                     ; preds = %for.body.188, %for.body.188.lr.ph
  br i1 undef, label %for.body.188, label %for.cond.185.for.end.198_crit_edge

for.cond.185.for.end.198_crit_edge:               ; preds = %for.body.188
  br label %for.end.198

for.end.198:                                      ; preds = %for.cond.185.for.end.198_crit_edge, %for.cond.185.preheader
  br label %return

if.end.206:                                       ; preds = %land.lhs.true.146, %if.end.142
  br i1 undef, label %land.lhs.true.210, label %return

land.lhs.true.210:                                ; preds = %if.end.206
  br i1 undef, label %if.then.214, label %return

if.then.214:                                      ; preds = %land.lhs.true.210
  br i1 undef, label %if.then.219, label %return

if.then.219:                                      ; preds = %if.then.214
  br label %return

return:                                           ; preds = %if.then.219, %if.then.214, %land.lhs.true.210, %if.end.206, %for.end.198, %if.then.183, %if.end.158, %if.then.157, %for.end, %if.then.125, %if.then.115, %if.then.96, %if.then.63, %if.then.37
  ret void
}

declare void @BYTES_TO_BITS(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512pf,-avx512vl,-fma4,-prfchw,-rdseed,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512pf,-avx512vl,-fma4,-prfchw,-rdseed,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
