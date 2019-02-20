; RUN: opt -S -objc-arc < %s | FileCheck %s
; rdar://9503416

; Detect loop boundaries and don't move retains and releases
; across them.

declare void @use_pointer(i8*)
declare i8* @llvm.objc.retain(i8*)
declare void @llvm.objc.release(i8*)
declare void @callee()
declare void @block_callee(void ()*)

; CHECK-LABEL: define void @test0(
; CHECK:   call i8* @llvm.objc.retain
; CHECK: for.body:
; CHECK-NOT: @objc
; CHECK: for.end:
; CHECK:   call void @llvm.objc.release
; CHECK: }
define void @test0(i8* %digits) {
entry:
  %tmp1 = call i8* @llvm.objc.retain(i8* %digits) nounwind
  call void @use_pointer(i8* %digits)
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %upcDigitIndex.01 = phi i64 [ 2, %entry ], [ %inc, %for.body ]
  call void @use_pointer(i8* %digits)
  %inc = add i64 %upcDigitIndex.01, 1
  %cmp = icmp ult i64 %inc, 12
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  call void @llvm.objc.release(i8* %digits) nounwind, !clang.imprecise_release !0
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK:   call i8* @llvm.objc.retain
; CHECK: for.body:
; CHECK-NOT: @objc
; CHECK: for.end:
; CHECK:   void @llvm.objc.release
; CHECK: }
define void @test1(i8* %digits) {
entry:
  %tmp1 = call i8* @llvm.objc.retain(i8* %digits) nounwind
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %upcDigitIndex.01 = phi i64 [ 2, %entry ], [ %inc, %for.body ]
  call void @use_pointer(i8* %digits)
  call void @use_pointer(i8* %digits)
  %inc = add i64 %upcDigitIndex.01, 1
  %cmp = icmp ult i64 %inc, 12
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  call void @llvm.objc.release(i8* %digits) nounwind, !clang.imprecise_release !0
  ret void
}

; CHECK-LABEL: define void @test2(
; CHECK:   call i8* @llvm.objc.retain
; CHECK: for.body:
; CHECK-NOT: @objc
; CHECK: for.end:
; CHECK:   void @llvm.objc.release
; CHECK: }
define void @test2(i8* %digits) {
entry:
  %tmp1 = call i8* @llvm.objc.retain(i8* %digits) nounwind
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %upcDigitIndex.01 = phi i64 [ 2, %entry ], [ %inc, %for.body ]
  call void @use_pointer(i8* %digits)
  %inc = add i64 %upcDigitIndex.01, 1
  %cmp = icmp ult i64 %inc, 12
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  call void @use_pointer(i8* %digits)
  call void @llvm.objc.release(i8* %digits) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete nested retain+release pairs around loops.

;      CHECK: define void @test3(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW:#[0-9]+]]
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   call void @llvm.objc.release(i8* %a)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test3(i8* %a) nounwind {
entry:
  %outer = call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  call void @callee()
  store i8 0, i8* %a
  br i1 undef, label %loop, label %exit

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test4(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   call void @llvm.objc.release(i8* %a)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test4(i8* %a) nounwind {
entry:
  %outer = call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br label %more

more:
  call void @callee()
  call void @callee()
  store i8 0, i8* %a
  br i1 undef, label %loop, label %exit

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test5(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   call void @callee()
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   call void @use_pointer(i8* %a)
; CHECK-NEXT:   call void @llvm.objc.release(i8* %a)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test5(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  call void @callee()
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @use_pointer(i8* %a)
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test6(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   call void @use_pointer(i8* %a)
; CHECK-NEXT:   call void @llvm.objc.release(i8* %a)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test6(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  call void @callee()
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @use_pointer(i8* %a)
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test7(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   call void @callee()
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   call void @llvm.objc.release(i8* %a)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test7(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  call void @callee()
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  call void @use_pointer(i8* %a)
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test8(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   call void @llvm.objc.release(i8* %a)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test8(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  call void @callee()
  call void @use_pointer(i8* %a)
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test9(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test9(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  call void @use_pointer(i8* %a)
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test10(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test10(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  call void @callee()
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

;      CHECK: define void @test11(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test11(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  br label %more

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

; Don't delete anything if they're not balanced.

;      CHECK: define void @test12(i8* %a) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %outer = tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   %inner = tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK-NEXT:   br label %loop
;  CHECK-NOT:   @llvm.objc.
;      CHECK: exit:
; CHECK-NEXT: call void @llvm.objc.release(i8* %a) [[NUW]]
; CHECK-NEXT: call void @llvm.objc.release(i8* %a) [[NUW]], !clang.imprecise_release !0
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test12(i8* %a) nounwind {
entry:
  %outer = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  %inner = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  br i1 undef, label %true, label %more

true:
  ret void

more:
  br i1 undef, label %exit, label %loop

exit:
  call void @llvm.objc.release(i8* %a) nounwind
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

; Do not improperly pair retains in a for loop with releases outside of a for
; loop when the proper pairing is disguised by a separate provenance represented
; by an alloca.
; rdar://12969722

; CHECK: define void @test13(i8* %a) [[NUW]] {
; CHECK: entry:
; CHECK:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK: loop:
; CHECK:   tail call i8* @llvm.objc.retain(i8* %a) [[NUW]]
; CHECK:   call void @block_callee
; CHECK:   call void @llvm.objc.release(i8* %reloaded_a) [[NUW]]
; CHECK: exit:
; CHECK:   call void @llvm.objc.release(i8* %a) [[NUW]]
; CHECK: }
define void @test13(i8* %a) nounwind {
entry:
  %block = alloca i8*
  %a1 = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  br label %loop

loop:
  %a2 = tail call i8* @llvm.objc.retain(i8* %a) nounwind
  store i8* %a, i8** %block, align 8
  %casted_block = bitcast i8** %block to void ()*
  call void @block_callee(void ()* %casted_block)
  %reloaded_a = load i8*, i8** %block, align 8
  call void @llvm.objc.release(i8* %reloaded_a) nounwind, !clang.imprecise_release !0
  br i1 undef, label %loop, label %exit
  
exit:
  call void @llvm.objc.release(i8* %a) nounwind, !clang.imprecise_release !0
  ret void
}

; CHECK: attributes [[NUW]] = { nounwind }

!0 = !{}
