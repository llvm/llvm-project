; RUN: opt < %s -passes='early-cse<memssa>,gvn-hoist' -earlycse-debug-hash -S | FileCheck %s

; Make sure opt doesn't crash. On top of that, the instructions
; of the side blocks should be hoisted to the entry block.

%s = type { i32, i64 }
%S = type { %s, i32 }

;CHECK-LABEL: @foo

define void @foo(ptr %arg) {
bb0:
  %call.idx.val.i = load i32, ptr %arg
  br label %bb1

;CHECK: bb1:
;CHECK:   %call264 = call zeroext i1 @bar
;CHECK:   store i32 %call.idx.val.i, ptr %arg
;CHECK:   %0 = getelementptr inbounds %S, ptr %arg, i64 0, i32 0, i32 1
;CHECK:   store i64 undef, ptr %0
;CHECK:   br i1 %call264, label %bb2, label %bb3

bb1:
  %call264 = call zeroext i1 @bar()
  br i1 %call264, label %bb2, label %bb3

;CHECK:     bb2:
;CHECK-NOT:   store i32 %call.idx.val.i, ptr %arg
;CHECK-NOT:   store i64 undef, ptr %{.*}

bb2:
  store i32 %call.idx.val.i, ptr %arg
  %0 = getelementptr inbounds %S, ptr %arg, i64 0, i32 0, i32 1
  store i64 undef, ptr %0
  ret void

;CHECK:     bb3:
;CHECK-NOT:   store i32 %call.idx.val.i, ptr %arg
;CHECK-NOT:   store i64 undef, ptr %{.*}

bb3:
  store i32 %call.idx.val.i, ptr %arg
  %1 = getelementptr inbounds %S, ptr %arg, i64 0, i32 0, i32 1
  store i64 undef, ptr %1
  ret void
}

declare zeroext i1 @bar()
