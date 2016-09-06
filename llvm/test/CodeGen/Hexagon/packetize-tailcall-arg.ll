; RUN: llc -march=hexagon < %s | FileCheck %s
; There should only be one packet:
; {
;   jump free
;   r0 = memw(r0 + #-4)
; }
;
; CHECK: {
; CHECK-NOT: {

define void @fred(i8* %p) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 -4
  %t0 = bitcast i8* %arrayidx to i8**
  %t1 = load i8*, i8** %t0, align 4
  tail call void @free(i8* %t1)
  ret void
}

; Function Attrs: nounwind
declare void @free(i8* nocapture) nounwind

