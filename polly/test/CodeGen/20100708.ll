; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define fastcc void @execute() nounwind {
entry:
  br i1 undef, label %check_stack.exit456.thread, label %bb.i451.preheader

bb.i451.preheader:                                ; preds = %bb116
  br label %bb.i451

bb.i451:                                          ; preds = %bb.i451, %bb.i451.preheader
  br label %bb.i451

check_stack.exit456.thread:                       ; preds = %bb116
  unreachable

}
