; RUN: opt < %s -mtriple=amdgcn -passes=early-cse -S | FileCheck %s
;
; Test type mismatch in ConstantFolding for vector types.

define internal void @f() {
  ret void
}

define void @test() {
  %1 = ptrtoint ptr @f to i64
  %2 = bitcast i64 %1 to <4 x i16>
  %3 = ptrtoint ptr @f to i64
  %4 = bitcast i64 %3 to <4 x i16>
  %sub = sub <4 x i16> %2, %4
  store <4 x i16> %sub, ptr @f, align 8
  ret void
}