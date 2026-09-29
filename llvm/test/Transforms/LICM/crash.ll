; RUN: opt -passes=licm -disable-output < %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<target-ir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


; PR8068
@g_12 = external global i8, align 1
define void @test1() nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond, %bb.nph
  store i8 0, ptr @g_12, align 1
  %tmp6 = load i8, ptr @g_12, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.body
  store i8 %tmp6, ptr @g_12, align 1
  br i1 false, label %for.cond.for.end10_crit_edge, label %for.body

for.cond.for.end10_crit_edge:                     ; preds = %for.cond
  br label %for.end10

for.end10:                                        ; preds = %for.cond.for.end10_crit_edge, %entry
  ret void
}

; PR8067
@g_8 = external global i32, align 4

define void @test2() noreturn nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %tmp7 = load i32, ptr @g_8, align 4
  store ptr @g_8, ptr undef, align 16
  store i32 undef, ptr @g_8, align 4
  br label %for.body
}

; PR8102
define void @test3(i1 %arg) {
entry:
  %__first = alloca { ptr }
  br i1 %arg, label %for.cond, label %for.end

for.cond:                                         ; preds = %for.cond, %entry
  %tmp2 = load ptr, ptr %__first, align 4
  %call = tail call ptr @test3helper(ptr %tmp2)
  store ptr %call, ptr %__first, align 4
  br i1 false, label %for.cond, label %for.end

for.end:                                          ; preds = %for.cond, %entry
  ret void
}

declare ptr @test3helper(ptr)


; PR8602
@g_47 = external global i32, align 4

define void @test4() noreturn nounwind {
  br label %1

; <label>:1                                       ; preds = %1, %0
  store volatile ptr @g_47, ptr undef, align 8
  store i32 undef, ptr @g_47, align 4
  br label %1
}

; OSS-Fuzz #29050
; https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=29050
define <2 x i177> @ossfuzz_29050(<2 x i177> %X) {
bb:
  br label %BB
BB:
  %I3 = insertelement <2 x i177> undef, i177 95780971304118053647396689196894323976171195136475135, i177 95780971304118053647396689196894323976171195136475135
  br i1 true, label %BB, label %BB1
BB1:
  ret <2 x i177> %I3
}
