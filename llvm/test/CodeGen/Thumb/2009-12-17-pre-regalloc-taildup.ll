; RUN: llc -O3 < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This test should not produce any spills, even when tail duplication creates lots of phi nodes.
; CHECK-NOT: push
; CHECK-NOT: pop
; CHECK: bx lr

@codetable.2928 = internal constant [5 x ptr] [ptr blockaddress(@interpret_threaded, %RETURN), ptr blockaddress(@interpret_threaded, %INCREMENT), ptr blockaddress(@interpret_threaded, %DECREMENT), ptr blockaddress(@interpret_threaded, %DOUBLE), ptr blockaddress(@interpret_threaded, %SWAPWORD)] ; <ptr> [#uses=5]
@llvm.used = appending global [1 x ptr] [ptr @interpret_threaded], section "llvm.metadata" ; <ptr> [#uses=0]

define i32 @interpret_threaded(ptr nocapture %opcodes) nounwind readonly optsize {
entry:
  %0 = load i8, ptr %opcodes, align 1                 ; <i8> [#uses=1]
  %1 = zext i8 %0 to i32                          ; <i32> [#uses=1]
  %2 = getelementptr inbounds [5 x ptr], ptr @codetable.2928, i32 0, i32 %1 ; <ptr> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb.backedge, %entry
  %indvar = phi i32 [ %phitmp, %bb.backedge ], [ 1, %entry ] ; <i32> [#uses=2]
  %gotovar.22.0.in = phi ptr [ %gotovar.22.0.in.be, %bb.backedge ], [ %2, %entry ] ; <ptr> [#uses=1]
  %result.0 = phi i32 [ %result.0.be, %bb.backedge ], [ 0, %entry ] ; <i32> [#uses=6]
  %opcodes_addr.0 = getelementptr i8, ptr %opcodes, i32 %indvar ; <ptr> [#uses=4]
  %gotovar.22.0 = load ptr, ptr %gotovar.22.0.in, align 4 ; <ptr> [#uses=1]
  indirectbr ptr %gotovar.22.0, [label %RETURN, label %INCREMENT, label %DECREMENT, label %DOUBLE, label %SWAPWORD]

RETURN:                                           ; preds = %bb
  ret i32 %result.0

INCREMENT:                                        ; preds = %bb
  %3 = add nsw i32 %result.0, 1                   ; <i32> [#uses=1]
  %4 = load i8, ptr %opcodes_addr.0, align 1          ; <i8> [#uses=1]
  %5 = zext i8 %4 to i32                          ; <i32> [#uses=1]
  %6 = getelementptr inbounds [5 x ptr], ptr @codetable.2928, i32 0, i32 %5 ; <ptr> [#uses=1]
  br label %bb.backedge

bb.backedge:                                      ; preds = %SWAPWORD, %DOUBLE, %DECREMENT, %INCREMENT
  %gotovar.22.0.in.be = phi ptr [ %20, %SWAPWORD ], [ %14, %DOUBLE ], [ %10, %DECREMENT ], [ %6, %INCREMENT ] ; <ptr> [#uses=1]
  %result.0.be = phi i32 [ %17, %SWAPWORD ], [ %11, %DOUBLE ], [ %7, %DECREMENT ], [ %3, %INCREMENT ] ; <i32> [#uses=1]
  %phitmp = add i32 %indvar, 1                    ; <i32> [#uses=1]
  br label %bb

DECREMENT:                                        ; preds = %bb
  %7 = add i32 %result.0, -1                      ; <i32> [#uses=1]
  %8 = load i8, ptr %opcodes_addr.0, align 1          ; <i8> [#uses=1]
  %9 = zext i8 %8 to i32                          ; <i32> [#uses=1]
  %10 = getelementptr inbounds [5 x ptr], ptr @codetable.2928, i32 0, i32 %9 ; <ptr> [#uses=1]
  br label %bb.backedge

DOUBLE:                                           ; preds = %bb
  %11 = shl i32 %result.0, 1                      ; <i32> [#uses=1]
  %12 = load i8, ptr %opcodes_addr.0, align 1         ; <i8> [#uses=1]
  %13 = zext i8 %12 to i32                        ; <i32> [#uses=1]
  %14 = getelementptr inbounds [5 x ptr], ptr @codetable.2928, i32 0, i32 %13 ; <ptr> [#uses=1]
  br label %bb.backedge

SWAPWORD:                                         ; preds = %bb
  %15 = shl i32 %result.0, 16                     ; <i32> [#uses=1]
  %16 = ashr i32 %result.0, 16                    ; <i32> [#uses=1]
  %17 = or i32 %15, %16                           ; <i32> [#uses=1]
  %18 = load i8, ptr %opcodes_addr.0, align 1         ; <i8> [#uses=1]
  %19 = zext i8 %18 to i32                        ; <i32> [#uses=1]
  %20 = getelementptr inbounds [5 x ptr], ptr @codetable.2928, i32 0, i32 %19 ; <ptr> [#uses=1]
  br label %bb.backedge
}
