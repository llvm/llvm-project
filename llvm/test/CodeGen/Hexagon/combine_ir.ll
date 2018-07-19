; RUN: llc -march=hexagon -mcpu=hexagonv4 -disable-hsdr < %s | FileCheck %s

declare void @bar(i64)

; CHECK-LABEL: halfword:
; CHECK: combine(#0

define void @halfword(i16* nocapture %a) nounwind {
entry:
  %0 = load i16, i16* %a, align 2
  %1 = zext i16 %0 to i64
  %add.ptr = getelementptr inbounds i16, i16* %a, i32 1
  %2 = load i16, i16* %add.ptr, align 2
  %3 = zext i16 %2 to i64
  %4 = shl nuw nsw i64 %3, 16
  %ins = or i64 %4, %1
  tail call void @bar(i64 %ins) nounwind
  ret void
}

; CHECK-LABEL: byte:
; CHECK: combine(#0

define void @byte(i8* nocapture %a) nounwind {
entry:
  %0 = load i8, i8* %a, align 1
  %1 = zext i8 %0 to i64
  %add.ptr = getelementptr inbounds i8, i8* %a, i32 1
  %2 = load i8, i8* %add.ptr, align 1
  %3 = zext i8 %2 to i64
  %4 = shl nuw nsw i64 %3, 8
  %ins = or i64 %4, %1
  tail call void @bar(i64 %ins) nounwind
  ret void
}
