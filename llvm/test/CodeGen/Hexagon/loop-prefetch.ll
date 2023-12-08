; RUN: llc -march=hexagon -hexagon-loop-prefetch < %s | FileCheck %s
; CHECK: dcfetch

target triple = "hexagon"

define void @copy(ptr nocapture %d, ptr nocapture readonly %s, i32 %n) local_unnamed_addr #0 {
entry:
  %tobool2 = icmp eq i32 %n, 0
  br i1 %tobool2, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %n.addr.05 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %s.addr.04 = phi ptr [ %incdec.ptr, %while.body ], [ %s, %entry ]
  %d.addr.03 = phi ptr [ %incdec.ptr1, %while.body ], [ %d, %entry ]
  %dec = add i32 %n.addr.05, -1
  %incdec.ptr = getelementptr inbounds i32, ptr %s.addr.04, i32 1
  %0 = load i32, ptr %s.addr.04, align 4
  %incdec.ptr1 = getelementptr inbounds i32, ptr %d.addr.03, i32 1
  store i32 %0, ptr %d.addr.03, align 4
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="-hvx" }
