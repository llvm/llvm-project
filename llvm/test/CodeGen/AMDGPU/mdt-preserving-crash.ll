; RUN: llc < %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@_RSENC_gDcd_______________________________ = external protected addrspace(1) externally_initialized global [4096 x i8], align 16

define protected amdgpu_kernel void @_RSENC_PRInit__________________________________(i1 %c0) local_unnamed_addr #0 {
entry:
  %runtimeVersionCopy = alloca [128 x i8], align 16, addrspace(5)
  %licenseVersionCopy = alloca [128 x i8], align 16, addrspace(5)
  %pD10 = alloca [128 x i8], align 16, addrspace(5)
  br label %if.end

if.end:                                           ; preds = %entry
  %0 = load i32, ptr undef, align 4
  %mul = mul i32 %0, 3
  %cmp13 = icmp eq i32 %mul, 989619
  br i1 %cmp13, label %cleanup.cont, label %if.end15

if.end15:                                         ; preds = %if.end
  br i1 %c0, label %cleanup.cont, label %lor.lhs.false17

lor.lhs.false17:                                  ; preds = %if.end15
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i, %lor.lhs.false17
  switch i32 undef, label %if.end60 [
    i32 0, label %while.cond.i
    i32 3, label %if.end60.loopexit857
  ]

if.end60.loopexit857:                             ; preds = %while.cond.i
  br label %if.end60

if.end60:                                         ; preds = %if.end60.loopexit857, %while.cond.i
  %1 = load i8, ptr addrspace(1) getelementptr inbounds ([4096 x i8], ptr addrspace(1) @_RSENC_gDcd_______________________________, i64 0, i64 655), align 1
  %arrayidx144260.5 = getelementptr inbounds [128 x i8], ptr addrspace(5) %runtimeVersionCopy, i32 0, i32 5
  %arrayidx156258.5 = getelementptr inbounds [128 x i8], ptr addrspace(5) %licenseVersionCopy, i32 0, i32 5
  switch i8 0, label %if.end5.i [
    i8 45, label %if.then.i
    i8 43, label %if.then3.i
  ]

if.then.i:                                        ; preds = %if.end60
  unreachable

if.then3.i:                                       ; preds = %if.end60
  br label %if.end5.i

if.end5.i:                                        ; preds = %if.then3.i, %if.end60
  %pS.addr.0.i = phi ptr addrspace(5) [ undef, %if.then3.i ], [ %runtimeVersionCopy, %if.end60 ]
  %2 = load i8, ptr addrspace(5) %pS.addr.0.i, align 1
  %conv612.i = sext i8 %2 to i32
  %sub13.i = add nsw i32 %conv612.i, -48
  %cmp714.i = icmp ugt i32 %sub13.i, 9
  switch i8 undef, label %if.end5.i314 [
    i8 45, label %if.then.i306
    i8 43, label %if.then3.i308
  ]

if.then.i306:                                     ; preds = %if.end5.i
  unreachable

if.then3.i308:                                    ; preds = %if.end5.i
  br label %if.end5.i314

if.end5.i314:                                     ; preds = %if.then3.i308, %if.end5.i
  %pS.addr.0.i309 = phi ptr addrspace(5) [ undef, %if.then3.i308 ], [ %licenseVersionCopy, %if.end5.i ]
  %3 = load i8, ptr addrspace(5) %pS.addr.0.i309, align 1
  %conv612.i311 = sext i8 %3 to i32
  %sub13.i312 = add nsw i32 %conv612.i311, -48
  %cmp714.i313 = icmp ugt i32 %sub13.i312, 9
  switch i8 undef, label %if.end5.i338 [
    i8 45, label %if.then.i330
    i8 43, label %if.then3.i332
  ]

if.then.i330:                                     ; preds = %if.end5.i314
  unreachable

if.then3.i332:                                    ; preds = %if.end5.i314
  br label %if.end5.i338

if.end5.i338:                                     ; preds = %if.then3.i332, %if.end5.i314
  %pS.addr.0.i333 = phi ptr addrspace(5) [ undef, %if.then3.i332 ], [ %arrayidx144260.5, %if.end5.i314 ]
  %4 = load i8, ptr addrspace(5) %pS.addr.0.i333, align 1
  %conv612.i335 = sext i8 %4 to i32
  %sub13.i336 = add nsw i32 %conv612.i335, -48
  %cmp714.i337 = icmp ugt i32 %sub13.i336, 9
  switch i8 undef, label %if.end5.i362 [
    i8 45, label %if.then.i354
    i8 43, label %if.then3.i356
  ]

if.then.i354:                                     ; preds = %if.end5.i338
  unreachable

if.then3.i356:                                    ; preds = %if.end5.i338
  br label %if.end5.i362

if.end5.i362:                                     ; preds = %if.then3.i356, %if.end5.i338
  %pS.addr.0.i357 = phi ptr addrspace(5) [ undef, %if.then3.i356 ], [ %arrayidx156258.5, %if.end5.i338 ]
  %5 = load i8, ptr addrspace(5) %pS.addr.0.i357, align 1
  %conv612.i359 = sext i8 %5 to i32
  %sub13.i360 = add nsw i32 %conv612.i359, -48
  %cmp714.i361 = icmp ugt i32 %sub13.i360, 9
  store i8 0, ptr addrspace(5) undef, align 16
  %6 = load i8, ptr addrspace(1) getelementptr inbounds ([4096 x i8], ptr addrspace(1) @_RSENC_gDcd_______________________________, i64 0, i64 1153), align 1
  %arrayidx232250.1 = getelementptr inbounds [128 x i8], ptr addrspace(5) %pD10, i32 0, i32 1
  store i8 %6, ptr addrspace(5) %arrayidx232250.1, align 1
  switch i8 undef, label %if.end5.i400 [
    i8 45, label %if.then.i392
    i8 43, label %if.then3.i394
  ]

if.then.i392:                                     ; preds = %if.end5.i362
  unreachable

if.then3.i394:                                    ; preds = %if.end5.i362
  br label %if.end5.i400

if.end5.i400:                                     ; preds = %if.then3.i394, %if.end5.i362
  %pS.addr.0.i395 = phi ptr addrspace(5) [ %arrayidx232250.1, %if.then3.i394 ], [ undef, %if.end5.i362 ]
  %7 = load i8, ptr addrspace(5) %pS.addr.0.i395, align 1
  %conv612.i397 = sext i8 %7 to i32
  %sub13.i398 = add nsw i32 %conv612.i397, -48
  %cmp714.i399 = icmp ugt i32 %sub13.i398, 9
  %8 = load i8, ptr undef, align 1
  %cmp9.not.i500 = icmp eq i8 0, %8
  br label %land.lhs.true402.critedge

land.lhs.true402.critedge:                        ; preds = %if.end5.i400
  br i1 %cmp9.not.i500, label %if.then404, label %if.else407

if.then404:                                       ; preds = %land.lhs.true402.critedge
  br label %for.body564

if.else407:                                       ; preds = %land.lhs.true402.critedge
  br label %if.end570

for.body564:                                      ; preds = %for.body564, %if.then404
  %i560.0801 = phi i32 [ 0, %if.then404 ], [ %inc568.31, %for.body564 ]
  %inc568.31 = add nuw nsw i32 %i560.0801, 32
  %exitcond839.not.31 = icmp eq i32 %inc568.31, 4096
  br i1 %exitcond839.not.31, label %if.end570, label %for.body564

if.end570:                                        ; preds = %for.body564, %if.else407
  unreachable

cleanup.cont:                                     ; preds = %if.end15, %if.end
  ret void
}

attributes #0 = { "uniform-work-group-size"="true" }
