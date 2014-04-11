; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -domtree -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s --check-prefix=DOM
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; rdar://7282591

@X = common global i32 0
@Y = common global i32 0
@Z = common global i32 0

; CHECK-LABEL: foo
; CHECK:  NoAlias: i32* %P, i32* @Z

; DOM-LABEL: foo
; DOM:  NoAlias: i32* %P, i32* @Z

define void @foo(i32 %cond) nounwind {
entry:
  %"alloca point" = bitcast i32 0 to i32
  %tmp = icmp ne i32 %cond, 0
  br i1 %tmp, label %bb, label %bb1

bb:
  br label %bb2

bb1:
  br label %bb2

bb2:
  %P = phi i32* [ @X, %bb ], [ @Y, %bb1 ]
  %tmp1 = load i32* @Z, align 4
  store i32 123, i32* %P, align 4
  %tmp2 = load i32* @Z, align 4
  br label %return

return:
  ret void
}

; Pointers can vary in between iterations of loops.
; PR18068

; CHECK-LABEL: pr18068
; CHECK: MayAlias: i32* %0, i32* %arrayidx5

; DOM-LABEL: pr18068
; DOM: MayAlias: i32* %0, i32* %arrayidx5

define i32 @pr18068(i32* %jj7, i32* %j) {
entry:
  %oa5 = alloca [100 x i32], align 16
  br label %codeRepl

codeRepl:
  %0 = phi i32* [ %arrayidx13, %for.body ], [ %j, %entry ]
  %targetBlock = call i1 @cond(i32* %jj7)
  br i1 %targetBlock, label %for.body, label %bye

for.body:
  %1 = load i32* %jj7, align 4
  %idxprom4 = zext i32 %1 to i64
  %arrayidx5 = getelementptr inbounds [100 x i32]* %oa5, i64 0, i64 %idxprom4
  %2 = load i32* %arrayidx5, align 4
  %sub6 = sub i32 %2, 6
  store i32 %sub6, i32* %arrayidx5, align 4
  ; %0 and %arrayidx5 can alias! It is not safe to DSE the above store.
  %3 = load i32* %0, align 4
  store i32 %3, i32* %arrayidx5, align 4
  %sub11 = add i32 %1, -1
  %idxprom12 = zext i32 %sub11 to i64
  %arrayidx13 = getelementptr inbounds [100 x i32]* %oa5, i64 0, i64 %idxprom12
  call void @inc(i32* %jj7)
  br label %codeRepl

bye:
  %.reload = load i32* %jj7, align 4
  ret i32 %.reload
}

declare i1 @cond(i32*)

declare void @inc(i32*)


