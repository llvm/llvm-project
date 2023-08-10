; RUN: opt < %s -S -p='newgvn' | FileCheck %s
; MemorySSA should be able to handle a clobber query with an empty MemoryLocation.

; CHECK: @userread
define ptr @userread(ptr %p) {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
; 2 = MemoryPhi({entry,liveOnEntry},{loop,1})
  %pos = phi i64 [ 1, %entry ], [ %diff, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i64 %pos
; MemoryUse(2)
  %ld = load ptr, ptr %gep, align 8
; 1 = MemoryDef(2)->2
  %readval = call i64 @fread(ptr noundef nonnull %gep, i64 noundef 1, i64 noundef %pos, ptr noundef %ld)
  %readvalispos = icmp eq i64 %readval, %pos
  call void @llvm.assume(i1 %readvalispos)
  %diff = sub i64 0, %pos
  br label %loop
}

declare noundef i64 @fread(ptr nocapture noundef %0, i64 noundef %1, i64 noundef %2, ptr nocapture noundef %3) local_unnamed_addr #0
declare void @llvm.assume(i1 %cond)

attributes #0 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "prefer-vector-width"="128" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+aes,+cmov,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="generic" }
