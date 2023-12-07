; RUN: opt %loadPolly -polly-scops -disable-output < %s

; Bug description: Alias Analysis thinks IntToPtrInst aliases with alloca instructions created by IndependentBlocks Pass.
;                  This will trigger the assertion when we are verifying the SCoP after IndependentBlocks.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.precisionType = type { i16, i16, i16, i8, [1 x i16] }

define void @main() nounwind {
entry:
 br label %bb1.i198.i

bb1.i198.i:                                       ; preds = %bb.i197.i, %psetq.exit196.i
  %tmp51.i = inttoptr i64 0 to ptr
  br i1 undef, label %bb1.i210.i, label %bb.i209.i

bb.i209.i:                                        ; preds = %bb1.i198.i
  br label %bb1.i210.i

bb1.i210.i:                                       ; preds = %bb.i209.i, %bb1.i198.i
  %0 = icmp eq i64 0, 0
  br i1 %0, label %bb1.i216.i, label %bb.i215.i

bb.i215.i:                                        ; preds = %bb1.i210.i
  store i16 undef, ptr %tmp51.i, align 2
  br label %bb1.i216.i

bb1.i216.i:                                       ; preds = %bb.i215.i, %bb1.i210.i
  br i1 undef, label %psetq.exit220.i, label %bb2.i217.i

bb2.i217.i:                                       ; preds = %bb1.i216.i
  br i1 undef, label %bb3.i218.i, label %psetq.exit220.i

bb3.i218.i:                                       ; preds = %bb2.i217.i
  br label %psetq.exit220.i

psetq.exit220.i:                                  ; preds = %bb3.i218.i, %bb2.i217.i, %bb1.i216.i
  br i1 undef, label %bb14.i76, label %bb15.i77

bb14.i76:                                         ; preds = %psetq.exit220.i
  unreachable

bb15.i77:                                         ; preds = %psetq.exit220.i
  br i1 %0, label %psetq.exit238.i, label %bb2.i235.i

bb2.i235.i:                                       ; preds = %bb15.i77
  br i1 undef, label %bb3.i236.i, label %psetq.exit238.i

bb3.i236.i:                                       ; preds = %bb2.i235.i
  unreachable

psetq.exit238.i:                                  ; preds = %bb2.i235.i, %bb15.i77
  unreachable

bb56.i.loopexit:                                  ; preds = %psetq.exit172.i
  unreachable
}
