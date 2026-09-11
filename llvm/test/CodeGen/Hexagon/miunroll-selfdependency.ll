; RUN: llc -O3 -march=hexagon -debug-only=machine-unroller < %s 2>&1 |\
; RUN:  FileCheck %s
; The test checks that we don't unroll the loop if we detect there is a self
; dependency between instructions across loop iterations that cannot be removed
; and ResMII=1
; CHECK: Self Dependencies Found. Using unroll factor = 1

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @fac(i32 %n) local_unnamed_addr {
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %while.body, label %while.end

while.body:                                       ; preds = %entry, %while.body
  %f.07 = phi i32 [ %mul, %while.body ], [ 1, %entry ]
  %n.addr.06 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %mul = mul nsw i32 %f.07, %n.addr.06
  %dec = add nsw i32 %n.addr.06, -1
  %cmp = icmp sgt i32 %dec, 0
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %entry
  %f.0.lcssa = phi i32 [ 1, %entry ], [ %mul, %while.body ]
  ret i32 %f.0.lcssa
}
