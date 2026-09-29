; RUN: llc < %s -mtriple=x86_64-apple-darwin
; rdar://7886733

%struct.CMTime = type <{ i64, i32, i32, i64 }>
%struct.CMTimeMapping = type { %struct.CMTimeRange, %struct.CMTimeRange }
%struct.CMTimeRange = type { %struct.CMTime, %struct.CMTime }

define void @t(ptr noalias nocapture sret(%struct.CMTimeMapping) %agg.result) nounwind optsize ssp {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.result, ptr align 4 null, i64 96, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
