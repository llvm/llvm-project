; RUN: opt -scalar-evolution -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt "-passes=print<scalar-evolution>" -disable-output < %s 2>&1 | FileCheck %s

; CHECK: %1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* null, i32 3
; CHECK: -->  (3 * sizeof(<vscale x 4 x i32>)) U: [0,-15) S: [-9223372036854775808,9223372036854775793)
; CHECK: %2 = getelementptr <vscale x 1 x i64>, <vscale x 1 x i64>* %p, i32 1
; CHECK: -->  (sizeof(<vscale x 1 x i64>) + %p) U: full-set S: full-set
define void @a(<vscale x 1 x i64> *%p) {
  getelementptr <vscale x 4 x i32>, <vscale x 4 x i32> *null, i32 3
  getelementptr <vscale x 1 x i64>, <vscale x 1 x i64> *%p, i32 1
  ret void
}
