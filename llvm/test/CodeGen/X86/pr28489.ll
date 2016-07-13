; ; RUN: llc < %s -mtriple=i686-pc-linux -O0 | FileCheck %s
declare void @g(i32, i1)

;CHECK-LABEL: f:
;CHECK: cmpxchg8b
;CHECK: sete %cl
;CHECK: movzbl %cl
define void @f(i64* %arg, i64 %arg1) {
entry:
  %tmp5 = cmpxchg i64* %arg, i64 %arg1, i64 %arg1 seq_cst seq_cst
  %tmp7 = extractvalue { i64, i1 } %tmp5, 1
  %tmp9 = zext i1 %tmp7 to i32
  call void @g(i32 %tmp9, i1 %tmp7)
  ret void
}
