; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define void @mem() {
bb:
  br label %bb6

bb6:
  %.0 = phi i8** [ undef, %bb ], [ %t2, %bb6 ]
  %tmp = load i8*, i8** %.0, align 8
  %bc = bitcast i8* %tmp to i8**
  %t1 = load i8*, i8** %bc, align 8
  %t2 = bitcast i8* %t1 to i8**
  br label %bb6

bb206:
  ret void
; CHECK: phi
; CHECK: bitcast
; CHECK: load
}
