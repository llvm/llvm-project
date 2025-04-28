; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; CHECK-LABEL: @test(
; CHECK:       call void asm sideeffect
; CHECK-NEXT:  call void asm sideeffect
; CHECK-NOT:   unreachable
; CHECK-NEXT:  br label
define dso_local i64 @test(i64 %arg.coerce) local_unnamed_addr #0 align 16 {
entry:
  %tobool.not = icmp eq i64 %arg.coerce, 0
  br i1 %tobool.not, label %if.then.i.i, label %if.end

if.end:                                           ; preds = %entry
  %cond9.i = call i64 @llvm.abs.i64(i64 %arg.coerce, i1 false)
  br label %bar.1.exit.i

if.then.i.i:                                      ; preds = %entry
  call void asm sideeffect "# test", "i,r,i,~{dirflag},~{fpsr},~{flags}"(i32 199, i32 2305, i64 12)
  call void asm sideeffect "# bar.1", "i,r,i,~{dirflag},~{fpsr},~{flags}"(i32 32, i32 2305, i64 12)
  br label %bar.1.exit.i

bar.1.exit.i:         ; preds = %if.then.i.i, %if.end
  %cond9.i4 = phi i64 [ 0, %if.then.i.i ], [ %cond9.i, %if.end ]
  %rem.i.i.i.i = urem i64 4294967296, %cond9.i4
  %div.i.i.i.i = udiv i64 4294967296, %cond9.i4
  br label %do.body.i

do.body.i:                                        ; preds = %do.body.i, %bar.1.exit.i
  %remainder.0.i = phi i64 [ %rem.i.i.i.i, %bar.1.exit.i ], [ %storemerge.i, %do.body.i ]
  %i.0.i = phi i32 [ 32, %bar.1.exit.i ], [ %dec.i, %do.body.i ]
  %res_value.0.i = phi i64 [ %div.i.i.i.i, %bar.1.exit.i ], [ %res_value.1.i, %do.body.i ]
  %shl.i = shl i64 %remainder.0.i, 1
  %shl25.i = shl i64 %res_value.0.i, 1
  %cmp26.not.i = icmp uge i64 %shl.i, %cond9.i4
  %sub29.i = select i1 %cmp26.not.i, i64 %cond9.i4, i64 0
  %storemerge.i = sub nuw i64 %shl.i, %sub29.i
  %or.i = zext i1 %cmp26.not.i to i64
  %res_value.1.i = or disjoint i64 %shl25.i, %or.i
  %dec.i = add nsw i32 %i.0.i, -1
  %cmp31.not.i = icmp eq i32 %dec.i, 0
  br i1 %cmp31.not.i, label %do.end.i, label %do.body.i

do.end.i:                                         ; preds = %do.body.i
  %shl33.i = shl i64 %storemerge.i, 1
  %cmp34.i = icmp uge i64 %shl33.i, %cond9.i4
  %sub38.i = select i1 %cmp34.i, i64 9223372036854775806, i64 9223372036854775807
  %cmp39.not.i = icmp ugt i64 %res_value.1.i, %sub38.i
  br i1 %cmp39.not.i, label %if.then55.i, label %bar.2.exit

if.then55.i:                                      ; preds = %do.end.i
  call void asm sideeffect "# bar.2", "i,r,i,~{dirflag},~{fpsr},~{flags}"(i32 88, i32 2305, i64 12)
  br label %bar.2.exit

bar.2.exit:                     ; preds = %if.then55.i, %do.end.i
  %conv36.i = zext i1 %cmp34.i to i64
  %add.i = add i64 %res_value.1.i, %conv36.i
  %numerator.lobit20.i = xor i64 %arg.coerce, 4294967296
  %sub75.i = sub i64 0, %add.i
  %tobool72.not22.i = icmp slt i64 %numerator.lobit20.i, 0
  %retval.sroa.0.0.i = select i1 %tobool72.not22.i, i64 %sub75.i, i64 %add.i
  ret i64 %retval.sroa.0.0.i
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.abs.i64(i64, i1 immarg)

attributes #0 = { fn_ret_thunk_extern noredzone nounwind null_pointer_is_valid sanitize_address sspstrong "min-legal-vector-width"="0" "no-builtin-wcslen" "no-jump-tables"="true" "no-trapping-math"="true" "patchable-function-entry"="0" "patchable-function-prefix"="16" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+retpoline-external-thunk,+retpoline-indirect-branches,+retpoline-indirect-calls,-aes,-avx,-avx10.1-256,-avx10.1-512,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512fp16,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-f16c,-fma,-fma4,-gfni,-kl,-mmx,-pclmul,-sha,-sha512,-sm3,-sm4,-sse,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-x87,-xop" "tune-cpu"="generic" }
