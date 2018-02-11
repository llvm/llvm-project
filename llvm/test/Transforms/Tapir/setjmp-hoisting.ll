; Test to verify that optimizations performed after Tapir lowering
; will not hoist the setjmp and comparison inserted by Tapir lowering
; above a branch on a different condition.  Although the IR for the
; transformed code is valid, X86 machine-code generation doesn't
; correctly handle the comparison with the setjmp return value and
; other intermingled comparisons.
;
; Credit to I-Ting Angelina Lee for the original source code for this
; test.
;
; RUN: opt < %s -tapir2target -tapir-target=cilk -O3 -S | FileCheck %s

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

declare fastcc void @simple_sort(i64* noalias %result, i64* noalias %array, i64 %size, i32 %reverse) unnamed_addr #0

declare fastcc void @parallel_merge(i64* noalias %result, i64* noalias %A, i64* noalias %B, i64 %size_a, i64 %size_b) unnamed_addr #0

; Function Attrs: nounwind uwtable
define fastcc void @cilk_sort_routine(i64* noalias %result, i64* noalias %array, i64 %size, i32 %reverse, i32 %level) unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp = icmp slt i64 %size, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call fastcc void @simple_sort(i64* %result, i64* %array, i64 %size, i32 %reverse)
  br label %if.end37

if.else:                                          ; preds = %entry
  %inc = add nsw i32 %level, 1
  %tobool = icmp ne i32 %reverse, 0
  %div1675 = lshr i64 %size, 1
  %lnot18 = xor i1 %tobool, true
  %lnot.ext19 = zext i1 %lnot18 to i32
  br i1 %tobool, label %if.else15, label %if.then1
; CHECK: if.else:
; CHECK: %tobool = icmp ne i32 %reverse, 0
; CHECK: [[SETJMP:%[a-zA-Z0-9._]+]] = call i32 @llvm.eh.sjlj.setjmp
; CHECK: [[SETJMPBOOL:%[a-zA-Z0-9._]+]] = icmp eq i32 [[SETJMP]], 0
; CHECK-NOT: br i1 %tobool
; CHECK: br i1 [[SETJMPBOOL]]

if.then1:                                         ; preds = %if.else
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %if.then1
  tail call fastcc void @cilk_sort_routine(i64* %result, i64* %array, i64 %div1675, i32 %lnot.ext19, i32 %inc)
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %if.then1
  %add.ptr = getelementptr inbounds i64, i64* %result, i64 %div1675
  %add.ptr5 = getelementptr inbounds i64, i64* %array, i64 %div1675
  %sub = sub nsw i64 %size, %div1675
  tail call fastcc void @cilk_sort_routine(i64* %add.ptr, i64* %add.ptr5, i64 %sub, i32 1, i32 %inc)
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont
  tail call fastcc void @parallel_merge(i64* %result, i64* %array, i64* %add.ptr5, i64 %div1675, i64 %sub)
  br label %if.end37

if.else15:                                        ; preds = %if.else
  detach within %syncreg, label %det.achd20, label %det.cont21

det.achd20:                                       ; preds = %if.else15
  tail call fastcc void @cilk_sort_routine(i64* %result, i64* %array, i64 %div1675, i32 %lnot.ext19, i32 %inc)
  reattach within %syncreg, label %det.cont21

det.cont21:                                       ; preds = %det.achd20, %if.else15
  %add.ptr23 = getelementptr inbounds i64, i64* %result, i64 %div1675
  %add.ptr25 = getelementptr inbounds i64, i64* %array, i64 %div1675
  %sub27 = sub nsw i64 %size, %div1675
  tail call fastcc void @cilk_sort_routine(i64* %add.ptr23, i64* %add.ptr25, i64 %sub27, i32 0, i32 %inc)
  sync within %syncreg, label %sync.continue31

sync.continue31:                                  ; preds = %det.cont21
  tail call fastcc void @parallel_merge(i64* %array, i64* %result, i64* %add.ptr23, i64 %div1675, i64 %sub27)
  br label %if.end37

if.end37:                                         ; preds = %sync.continue, %sync.continue31, %if.then
  ret void
}
