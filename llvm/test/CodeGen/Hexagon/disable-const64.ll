; RUN: llc -march=hexagon -mcpu=hexagonv67t < %s | FileCheck %s

target triple = "hexagon"

; Disable CONST64 for tiny core since it is a memory operation and tiny core has
; only one memory resource per packet.
; CHECK: ##
; CHECK: ##

define void @analyze(ptr nocapture %in) local_unnamed_addr {
entry:
  %0 = tail call i64 @llvm.hexagon.M2.mmachs.s1(i64 10230955697128160, i64 10230955697128160, i64 0)
  store i64 %0, ptr %in, align 8
  ret void
}

; CHECK-NOT: CONST64
define dso_local void @analyze2(ptr nocapture %in) local_unnamed_addr {
entry:
  %arrayidx = getelementptr inbounds i16, ptr %in, i32 3
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i64
  %1 = tail call i64 @llvm.hexagon.M2.mmacls.s1(i64 undef, i64 30432282833584128, i64 %conv)
  %arrayidx4 = getelementptr inbounds i16, ptr %in, i32 4
  store i64 %1, ptr %arrayidx4, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.mmachs.s1(i64, i64, i64)
declare i64 @llvm.hexagon.M2.mmacls.s1(i64, i64, i64)
