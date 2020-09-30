; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; CHECK: Printing analysis {{.*}} for function 'branch'
; CHECK: edge  -> return probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge  -> return2 probability is 0x40000000 / 0x80000000 = 50.00%
define void @branch(i1 %x) {
  br i1 %x, label %return, label %return2, !prof !1
return:
  ret void
return2:
  ret void
}

!1 = !{!"branch_weights",
       i64 -4611686018427387904,
       i64 -4611686018427387904}

define void @switch(i32 %x) {
  switch i32 %x, label %return [
    i32 0, label %return2
    i32 3, label %return2
    i32 6, label %return2
    i32 1, label %return2
    i32 4, label %return2
    i32 7, label %return2
    i32 2, label %return2
    i32 5, label %return2
    i32 8, label %return2
    i32 9, label %return2
  ], !prof !2
return:
  ret void
return2:
  ret void
}

!2 = !{!"branch_weights",
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904,
       i64 -4611686018427387904}
