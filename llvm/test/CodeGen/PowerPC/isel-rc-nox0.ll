; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@g_62 = external global [1 x [9 x i32]], align 4

; Function Attrs: nounwind
define void @main() #0 {
entry:
  br i1 undef, label %cond.true, label %for.cond1.preheader.i

cond.true:                                        ; preds = %entry
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %for.cond1.preheader.i, %cond.true, %entry
  br i1 undef, label %crc32_gentab.exit, label %for.cond1.preheader.i

crc32_gentab.exit:                                ; preds = %for.cond1.preheader.i
  %tobool.i19.i.i = icmp eq i32 undef, 0
  %retval.0.i.i.i = select i1 %tobool.i19.i.i, i32* getelementptr inbounds ([1 x [9 x i32]], [1 x [9 x i32]]* @g_62, i64 0, i64 0, i64 6), i32* getelementptr inbounds ([1 x [9 x i32]], [1 x [9 x i32]]* @g_62, i64 0, i64 0, i64 8)
  br label %for.cond1.preheader.i2961.i

for.cond1.preheader.i2961.i:                      ; preds = %for.inc44.i2977.i, %crc32_gentab.exit
  call void @llvm.memset.p0i8.i64(i8* bitcast ([1 x [9 x i32]]* @g_62 to i8*), i8 -1, i64 36, i32 4, i1 false) #1
  %0 = load i32, i32* %retval.0.i.i.i, align 4
  %tobool.i2967.i = icmp eq i32 %0, 0
  br label %for.body21.i2968.i

for.body21.i2968.i:                               ; preds = %safe_mod_func_int32_t_s_s.exit.i2974.i, %for.cond1.preheader.i2961.i
  br i1 %tobool.i2967.i, label %safe_mod_func_int32_t_s_s.exit.i2974.i, label %for.inc44.i2977.i

safe_mod_func_int32_t_s_s.exit.i2974.i:           ; preds = %for.body21.i2968.i
  br i1 undef, label %for.body21.i2968.i, label %for.inc44.i2977.i

for.inc44.i2977.i:                                ; preds = %safe_mod_func_int32_t_s_s.exit.i2974.i, %for.body21.i2968.i
  br i1 undef, label %func_80.exit2978.i, label %for.cond1.preheader.i2961.i

func_80.exit2978.i:                               ; preds = %for.inc44.i2977.i
  unreachable
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "ssp-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
