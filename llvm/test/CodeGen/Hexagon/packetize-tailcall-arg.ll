; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; There should only be one packet:
; {
;   jump free
;   r0 = memw(r0 + #-4)
; }
;
; CHECK: {
; CHECK-NOT: {

define void @fred(ptr %p) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 -4
  %t1 = load ptr, ptr %arrayidx, align 4
  tail call void @free(ptr %t1)
  ret void
}

; Function Attrs: nounwind
declare void @free(ptr nocapture) nounwind

