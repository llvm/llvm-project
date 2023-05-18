; RUN: llc < %s

define void @f() {
entry:
  br label %bb1

bb1:
  %0 = phi [0 x { ptr, i64, i64 }] [ %load, %bb2 ], [ undef, %entry ]
  store [0 x { ptr, i64, i64 }] %0, ptr undef, align 8
  %1 = icmp eq i64 undef, 0
  br i1 %1, label %bb2, label %bb3

bb2:
  %load = load [0 x { ptr, i64, i64 }], ptr undef, align 8
  br label %bb1

bb3:
  ret void
}
