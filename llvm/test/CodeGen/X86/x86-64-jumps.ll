; RUN: llc < %s 
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

define i8 @test1() nounwind ssp {
entry:
  %0 = select i1 undef, ptr blockaddress(@test1, %bb), ptr blockaddress(@test1, %bb6) ; <ptr> [#uses=1]
  indirectbr ptr %0, [label %bb, label %bb6]

bb:                                               ; preds = %entry
  ret i8 1

bb6:                                              ; preds = %entry
  ret i8 2
}


; PR5930 - Trunc of block address differences.
@test.array = internal constant [3 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr blockaddress(@test2, %foo) to i64), i64 ptrtoint (ptr blockaddress(@test2, %foo) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr blockaddress(@test2, %bar) to i64), i64 ptrtoint (ptr blockaddress(@test2, %foo) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr blockaddress(@test2, %hack) to i64), i64 ptrtoint (ptr blockaddress(@test2, %foo) to i64)) to i32)] ; <ptr> [#uses=1]

define void @test2(i32 %i) nounwind ssp {
entry:
  %i.addr = alloca i32                            ; <ptr> [#uses=2]
  store i32 %i, ptr %i.addr
  %tmp = load i32, ptr %i.addr                        ; <i32> [#uses=1]
  %idxprom = sext i32 %tmp to i64                 ; <i64> [#uses=1]
  %arrayidx = getelementptr inbounds i32, ptr @test.array, i64 %idxprom ; <ptr> [#uses=1]
  %tmp1 = load i32, ptr %arrayidx                     ; <i32> [#uses=1]
  %idx.ext = sext i32 %tmp1 to i64                ; <i64> [#uses=1]
  %add.ptr = getelementptr i8, ptr blockaddress(@test2, %foo), i64 %idx.ext ; <ptr> [#uses=1]
  br label %indirectgoto

foo:                                              ; preds = %indirectgoto, %indirectgoto, %indirectgoto, %indirectgoto, %indirectgoto
  br label %bar

bar:                                              ; preds = %foo, %indirectgoto
  br label %hack

hack:                                             ; preds = %bar, %indirectgoto
  ret void

indirectgoto:                                     ; preds = %entry
  %indirect.goto.dest = phi ptr [ %add.ptr, %entry ] ; <ptr> [#uses=1]
  indirectbr ptr %indirect.goto.dest, [label %foo, label %foo, label %bar, label %foo, label %hack, label %foo, label %foo]
}
