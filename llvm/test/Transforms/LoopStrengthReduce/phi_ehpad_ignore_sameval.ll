; This tests that LoopStrengthReduce ignores possible optimizations that are
; not realizable because they would require rewriting EHPad-class instructions.
; If this type of optimization is attempted it will hit the
; "Insertion point must be a normal instruction" assertion.
;
; See also https://bugs.llvm.org/show_bug.cgi?id=48708
;
; RUN: opt -loop-reduce -S %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:6:2:64-S128"

%"class.std::allocator" = type { i8 }
%"class.absl::Storage" = type {}

define void @0() personality i8* undef {
init1:
  %i14 = invoke i8* undef(i8* null, i8 0)
          to label %init2 unwind label %unwind

init2:                                            ; preds = %init1
  %i19 = select i1 undef, %"class.std::allocator"* null, %"class.std::allocator"* null
  br label %loop

loop:                                             ; preds = %loop.increment, %init2
  %i21 = phi i64 [ %i24, %loop.increment ], [ 0, %init2 ]
  %i22 = getelementptr %"class.std::allocator", %"class.std::allocator"* %i19, i64 %i21
  invoke void undef(i8* null, %"class.std::allocator"* null, %"class.std::allocator"* %i22)
          to label %loop.increment unwind label %loop.unwind

loop.increment:                                   ; preds = %loop
  %i24 = add i64 %i21, 1
  br label %loop

loop.unwind:                                      ; preds = %loop
  %i26 = catchswitch within none [label %loop.catch] unwind label %unwind

loop.catch:                                       ; preds = %loop.unwind
  %i28 = catchpad within %i26 []
  catchret from %i28 to label %caught

caught:                                           ; preds = %loop.catch
  invoke void undef(%"class.absl::Storage"* null)
          to label %unreach unwind label %unwind

unreach:                                          ; preds = %caught
  unreachable

unwind:                                           ; preds = %caught, %loop.unwind, %init1
  ; This phi node triggers the issue in combination with the optimizable loop
  ; above. It contains %i19 twice, once from %caught (which doesn't have an
  ; EHPad) and once from %loop.unwind, which does have one.
  %i32 = phi %"class.std::allocator"* [ %i19, %loop.unwind ], [ %i19, %caught ], [ null, %init1 ]
  %i33 = cleanuppad within none []
  cleanupret from %i33 unwind to caller
}
