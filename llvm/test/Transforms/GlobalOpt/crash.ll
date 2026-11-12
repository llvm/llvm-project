; RUN: opt -passes=globalopt -disable-output < %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.8"

%0 = type { i32, ptr, ptr }
%struct.btSimdScalar = type { %"union.btSimdScalar::$_14" }
%"union.btSimdScalar::$_14" = type { <4 x float> }

@_ZL6vTwist =  global %struct.btSimdScalar zeroinitializer ; <ptr> [#uses=1]
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, ptr @_GLOBAL__I__ZN21btConeTwistConstraintC2Ev, ptr null }] ; <ptr> [#uses=0]

define internal void @_GLOBAL__I__ZN21btConeTwistConstraintC2Ev() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  store float 1.0, ptr getelementptr inbounds (%struct.btSimdScalar, ptr @_ZL6vTwist, i32 0, i32 0, i32 0, i32 3), align 4
  ret void
}


; PR6760
%T = type { [5 x i32] }

@switch_inf = internal global ptr null

define void @test(ptr %arch_file, i32 %route_type) {
entry:
  %A = sext i32 1 to i64
  %B = mul i64 %A, 20
  %C = call noalias ptr @malloc(i64 %B) nounwind
  store ptr %C, ptr @switch_inf, align 8
  unreachable

bb.nph.i: 
  %scevgep.i539 = getelementptr i8, ptr %C, i64 4
  unreachable

xx:
  %E = load ptr, ptr @switch_inf, align 8 
  unreachable
}

declare noalias ptr @malloc(i64) nounwind


; PR8063
@permute_bitrev.bitrev = internal global ptr null, align 8
define void @permute_bitrev() nounwind {
entry:
  %tmp = load ptr, ptr @permute_bitrev.bitrev, align 8
  %conv = sext i32 0 to i64
  %mul = mul i64 %conv, 4
  %call = call ptr @malloc(i64 %mul)
  store ptr %call, ptr @permute_bitrev.bitrev, align 8
  ret void
}




@data8 = internal global [8000 x i8] zeroinitializer, align 16
define void @memset_with_strange_user() ssp {
  call void @llvm.memset.p0.i64(ptr align 16 @data8, i8 undef, i64 ptrtoint (ptr getelementptr ([8000 x i8], ptr @data8, i64 1, i64 sub (i64 0, i64 ptrtoint (ptr @data8 to i64))) to i64), i1 false)
  ret void
}
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind


; PR9856
@g_52 = internal global ptr null, align 8
@g_90 = external global ptr, align 8

define void @icmp_user_of_stored_once() nounwind ssp {
entry:
  %tmp4 = load ptr, ptr @g_52, align 8
  store ptr @g_90, ptr @g_52
  %cmp17 = icmp ne ptr undef, @g_52
  ret void
}

