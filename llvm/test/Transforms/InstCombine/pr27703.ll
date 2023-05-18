; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define void @mem() {
bb:
  br label %bb6

bb6:
  %.0 = phi ptr [ undef, %bb ], [ %t2, %bb6 ]
  %tmp = load ptr, ptr %.0, align 8
  %bc = bitcast ptr %tmp to ptr
  %t1 = load ptr, ptr %bc, align 8
  %t2 = bitcast ptr %t1 to ptr
  br label %bb6

bb206:
  ret void
; CHECK: phi
; CHECK-NEXT: load
; CHECK-NEXT: load
; CHECK-NEXT: br

}
