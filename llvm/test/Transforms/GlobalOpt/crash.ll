; RUN: opt -globalopt -disable-output %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.8"

%0 = type { i32, void ()* }
%struct.btSimdScalar = type { %"union.btSimdScalar::$_14" }
%"union.btSimdScalar::$_14" = type { <4 x float> }

@_ZL6vTwist =  global %struct.btSimdScalar zeroinitializer ; <%struct.btSimdScalar*> [#uses=1]
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, void ()* @_GLOBAL__I__ZN21btConeTwistConstraintC2Ev }] ; <[12 x %0]*> [#uses=0]

define internal void @_GLOBAL__I__ZN21btConeTwistConstraintC2Ev() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  store float 1.0, float* getelementptr inbounds (%struct.btSimdScalar* @_ZL6vTwist, i32 0, i32 0, i32 0, i32 3), align 4
  ret void
}


; PR6760
%T = type { [5 x i32] }

@switch_inf = internal global %T* null

define void @test(i8* %arch_file, i32 %route_type) {
entry:
  %A = sext i32 1 to i64
  %B = mul i64 %A, 20
  %C = call noalias i8* @malloc(i64 %B) nounwind
  %D = bitcast i8* %C to %T*
  store %T* %D, %T** @switch_inf, align 8
  unreachable

bb.nph.i: 
  %scevgep.i539 = getelementptr i8* %C, i64 4
  unreachable

xx:
  %E = load %T** @switch_inf, align 8 
  unreachable
}

declare noalias i8* @malloc(i64) nounwind
