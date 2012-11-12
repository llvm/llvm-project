; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: foo_empty
; CHECK-NOT: allocframe
; CHECK-NOT: memd(r29
; CHECK: jump bar_empty

define void @foo_empty(i32 %h) nounwind {
entry:
  %add = add nsw i32 %h, 3
  %call = tail call i32 bitcast (i32 (...)* @bar_empty to i32 (i32)*)(i32 %add) nounwind
  ret void
}

declare i32 @bar_empty(...)
