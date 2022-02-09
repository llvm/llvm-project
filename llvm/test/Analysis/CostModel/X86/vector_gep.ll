; RUN: opt < %s  -passes='print<cost-model>' 2>&1 -disable-output -mtriple=x86_64-linux-unknown-unknown -mattr=+avx512f | FileCheck %s

%struct.S = type { [1000 x i32] }


declare <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)

define <4 x i32> @foov(<4 x %struct.S*> %s, i64 %base){
  %temp = insertelement <4 x i64> undef, i64 %base, i32 0
  %vector = shufflevector <4 x i64> %temp, <4 x i64> undef, <4 x i32> zeroinitializer
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds %struct.S
  %B = getelementptr inbounds %struct.S, <4 x %struct.S*> %s, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds [1000 x i32]
  %arrayidx = getelementptr inbounds [1000 x i32], <4 x [1000 x i32]*> %B, <4 x i64> zeroinitializer, <4 x i64> %vector
  %res = call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*> %arrayidx, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> undef)
  ret <4 x i32> %res
}
