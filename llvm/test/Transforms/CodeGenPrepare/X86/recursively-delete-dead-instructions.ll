; RUN: opt -codegenprepare -S -mtriple=x86_64-linux < %s | FileCheck %s

declare void @llvm.assume(i1 noundef) nounwind willreturn

; Recursively deleting dead operands of assume() may result in its next
; instruction deleted and the iterator pointing to the next instruction
; invalidated. This prevents the following simple loop in
; CodeGenPrepare::optimizeBlock() unless CurInstIterator is fixed:
;
;   CurInstIterator = BB.begin();
;   while (CurInstIterator != BB.end())
;     optimizeInst(&*CurInstIterator++, ModifiedDT);
;
define i32 @test_assume_in_loop(i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test_assume_in_loop(
; CHECK-NEXT:  entry:
entry:
  br label %loop

; CHECK: loop:
; CHECK-NEXT:  br label %loop
loop:
  %cond3 = phi i1 [%cond1, %entry], [%cond4, %loop]
  call void @llvm.assume(i1 %cond3)
  %cond4 = icmp ult i1 %cond1, %cond2
  br label %loop
}
