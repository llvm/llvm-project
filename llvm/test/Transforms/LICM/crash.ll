; RUN: opt -licm -disable-output < %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,loop(licm)' -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


; PR8068
@g_12 = external global i8, align 1
define void @test1() nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond, %bb.nph
  store i8 0, i8* @g_12, align 1
  %tmp6 = load i8, i8* @g_12, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.body
  store i8 %tmp6, i8* @g_12, align 1
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
  %tmp7 = load i32, i32* @g_8, align 4
  store i32* @g_8, i32** undef, align 16
  store i32 undef, i32* @g_8, align 4
  br label %for.body
}

; PR8102
define void @test3() {
entry:
  %__first = alloca { i32* }
  br i1 undef, label %for.cond, label %for.end

for.cond:                                         ; preds = %for.cond, %entry
  %tmp1 = getelementptr { i32*}, { i32*}* %__first, i32 0, i32 0
  %tmp2 = load i32*, i32** %tmp1, align 4
  %call = tail call i32* @test3helper(i32* %tmp2)
  %tmp3 = getelementptr { i32*}, { i32*}* %__first, i32 0, i32 0
  store i32* %call, i32** %tmp3, align 4
  br i1 false, label %for.cond, label %for.end

for.end:                                          ; preds = %for.cond, %entry
  ret void
}

declare i32* @test3helper(i32*)


; PR8602
@g_47 = external global i32, align 4

define void @test4() noreturn nounwind {
  br label %1

; <label>:1                                       ; preds = %1, %0
  store volatile i32* @g_47, i32** undef, align 8
  store i32 undef, i32* @g_47, align 4
  br label %1
}
