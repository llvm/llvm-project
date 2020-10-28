; RUN: opt < %s -early-cse-memssa -earlycse-debug-hash -gvn-hoist -S | FileCheck %s

; Make sure opt doesn't crash. On top of that, the instructions
; of the side blocks should be hoisted to the entry block.

%s = type { i32, i64 }
%S = type { %s, i32 }

;CHECK-LABEL: @foo

define void @foo(i32* %arg) {
bb0:
  %0 = bitcast i32* %arg to %S*
  %call.idx.i = getelementptr %S, %S* %0, i64 0, i32 0, i32 0
  %call.idx.val.i = load i32, i32* %call.idx.i
  br label %bb1

;CHECK: bb1:
;CHECK:   %call264 = call zeroext i1 @bar
;CHECK:   store i32 %call.idx.val.i, i32* %call.idx.i
;CHECK:   %1 = getelementptr inbounds %S, %S* %0, i64 0, i32 0, i32 1
;CHECK:   store i64 undef, i64* %1
;CHECK:   br i1 %call264, label %bb2, label %bb3

bb1:
  %call264 = call zeroext i1 @bar()
  br i1 %call264, label %bb2, label %bb3

;CHECK:     bb2:
;CHECK-NOT:   store i32 %call.idx.val.i, i32* %call.idx.i
;CHECK-NOT:   store i64 undef, i64* %{.*}

bb2:
  store i32 %call.idx.val.i, i32* %call.idx.i
  %1 = getelementptr inbounds %S, %S* %0, i64 0, i32 0, i32 1
  store i64 undef, i64* %1
  ret void

;CHECK:     bb3:
;CHECK-NOT:   store i32 %call.idx.val.i, i32* %call.idx.i
;CHECK-NOT:   store i64 undef, i64* %{.*}

bb3:
  store i32 %call.idx.val.i, i32* %call.idx.i
  %2 = getelementptr inbounds %S, %S* %0, i64 0, i32 0, i32 1
  store i64 undef, i64* %2
  ret void
}

declare zeroext i1 @bar()
