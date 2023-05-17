; RUN: llc -O3 -mcpu=swift -mtriple=armv7s-apple-ios6.0.0 %s -o /dev/null
; rdar://14811848

; Make sure that we do not emit the BMOVPCB_CALL instruction for now or if we
; fix the assumptions in its implementation that we do not crash when doing it.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "armv7s-apple-ios6.0.0"

@main.title = private unnamed_addr constant [15 x i8] c"foo and stuff\0A\00", align 1
@.str = private unnamed_addr constant [3 x i8] c"%s\00", align 1

; Function Attrs: nounwind optsize ssp
define i32 @main() #0 {
entry:
  %title = alloca [15 x i8], align 1
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %title, ptr align 1 @main.title, i32 15, i1 false)
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %title) #3
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #1

; Function Attrs: nounwind optsize
declare i32 @printf(ptr nocapture readonly, ...) #2

attributes #0 = { nounwind optsize ssp "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind optsize }
