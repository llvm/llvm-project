; RUN: llc < %s -mtriple=x86_64-apple-darwin
; rdar://7886733

%struct.CMTime = type <{ i64, i32, i32, i64 }>
%struct.CMTimeMapping = type { %struct.CMTimeRange, %struct.CMTimeRange }
%struct.CMTimeRange = type { %struct.CMTime, %struct.CMTime }

define void @t(%struct.CMTimeMapping* noalias nocapture sret %agg.result) nounwind optsize ssp {
entry:
  %agg.result1 = bitcast %struct.CMTimeMapping* %agg.result to i8* ; <i8*> [#uses=1]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.result1, i8* null, i64 96, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
