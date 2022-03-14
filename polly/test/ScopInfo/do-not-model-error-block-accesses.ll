; RUN: opt %loadPolly -polly-print-scops -disable-output < %s

; Check that we do not crash on this input. Earlier this indeed crashed as
; we tried to model the access functions in an error block.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @FORMAT3_4() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br i1 false, label %if.end.38, label %if.else

if.else:                                          ; preds = %entry.split
  call void (i32, i32, i32*, ...) bitcast (void (...)* @BYTES_TO_BITS to void (i32, i32, i32*, ...)*)(i32 undef, i32 1, i32* undef) #2
  %0 = load i32, i32* null, align 4
  br label %if.end.38

if.end.38:                                        ; preds = %if.else, %entry.split
  unreachable
}

declare void @BYTES_TO_BITS(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512pf,-avx512vl,-fma4,-prfchw,-rdseed,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512pf,-avx512vl,-fma4,-prfchw,-rdseed,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0"}
