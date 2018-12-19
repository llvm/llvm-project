; RUN: opt -objc-arc -S < %s | FileCheck %s

declare i8* @llvm.objc.retain(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare void @llvm.objc.release(i8*)
declare i8* @llvm.objc.autorelease(i8*)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare void @llvm.objc.autoreleasePoolPop(i8*)
declare i8* @llvm.objc.autoreleasePoolPush()
declare i8* @llvm.objc.retainBlock(i8*)

declare i8* @objc_retainedObject(i8*)
declare i8* @objc_unretainedObject(i8*)
declare i8* @objc_unretainedPointer(i8*)

declare void @use_pointer(i8*)
declare void @callee()
declare void @callee_fnptr(void ()*)
declare void @invokee()
declare i8* @returner()
declare i8* @returner1()
declare i8* @returner2()
declare void @bar(i32 ()*)
declare void @use_alloca(i8**)

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare i8* @llvm.objc.msgSend(i8*, i8*, ...)


; In the presence of allocas, unconditionally remove retain/release pairs only
; if they are known safe in both directions. This prevents matching up an inner
; retain with the boundary guarding release in the following situation:
; 
; %A = alloca
; retain(%x)
; retain(%x) <--- Inner Retain
; store %x, %A
; %y = load %A
; ... DO STUFF ...
; release(%y)
; release(%x) <--- Guarding Release
;
; rdar://13750319

; CHECK: define void @test1a(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1a(i8* %x) {
entry:
  %A = alloca i8*
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  store i8* %x, i8** %A, align 8
  %y = load i8*, i8** %A
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1b(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1b(i8* %x) {
entry:
  %A = alloca i8*
  %gep = getelementptr i8*, i8** %A, i32 0
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8*, i8** %A
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}


; CHECK: define void @test1c(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1c(i8* %x) {
entry:
  %A = alloca i8*, i32 3
  %gep = getelementptr i8*, i8** %A, i32 2
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8*, i8** %gep
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}


; CHECK: define void @test1d(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1d(i8* %x) {
entry:
  br i1 undef, label %use_allocaA, label %use_allocaB

use_allocaA:
  %allocaA = alloca i8*
  br label %exit

use_allocaB:
  %allocaB = alloca i8*
  br label %exit

exit:
  %A = phi i8** [ %allocaA, %use_allocaA ], [ %allocaB, %use_allocaB ]
  %gep = getelementptr i8*, i8** %A, i32 0
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8*, i8** %gep
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1e(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1e(i8* %x) {
entry:
  br i1 undef, label %use_allocaA, label %use_allocaB

use_allocaA:
  %allocaA = alloca i8*, i32 4
  br label %exit

use_allocaB:
  %allocaB = alloca i8*, i32 4
  br label %exit

exit:
  %A = phi i8** [ %allocaA, %use_allocaA ], [ %allocaB, %use_allocaB ]
  %gep = getelementptr i8*, i8** %A, i32 2
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8*, i8** %gep
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1f(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1f(i8* %x) {
entry:
  %allocaOne = alloca i8*
  %allocaTwo = alloca i8*
  %A = select i1 undef, i8** %allocaOne, i8** %allocaTwo
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  store i8* %x, i8** %A, align 8
  %y = load i8*, i8** %A
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; Make sure that if a store is in a different basic block we handle known safe
; conservatively.


; CHECK: define void @test2a(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2a(i8* %x) {
entry:
  %A = alloca i8*
  store i8* %x, i8** %A, align 8
  %y = load i8*, i8** %A
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2b(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2b(i8* %x) {
entry:
  %A = alloca i8*
  %gep1 = getelementptr i8*, i8** %A, i32 0
  store i8* %x, i8** %gep1, align 8
  %gep2 = getelementptr i8*, i8** %A, i32 0
  %y = load i8*, i8** %gep2
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call i8* @llvm.objc.retain(i8* %x)
  tail call i8* @llvm.objc.retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2c(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2c(i8* %x) {
entry:
  %A = alloca i8*, i32 3
  %gep1 = getelementptr i8*, i8** %A, i32 2
  store i8* %x, i8** %gep1, align 8
  %gep2 = getelementptr i8*, i8** %A, i32 2
  %y = load i8*, i8** %gep2
  tail call i8* @llvm.objc.retain(i8* %x)
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call i8* @llvm.objc.retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2d(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.retain(i8* %x)
; CHECK: @llvm.objc.release(i8* %y)
; CHECK: @llvm.objc.release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2d(i8* %x) {
entry:
  tail call i8* @llvm.objc.retain(i8* %x)
  br label %bb1

bb1:
  %Abb1 = alloca i8*, i32 3
  %gepbb11 = getelementptr i8*, i8** %Abb1, i32 2
  store i8* %x, i8** %gepbb11, align 8
  %gepbb12 = getelementptr i8*, i8** %Abb1, i32 2
  %ybb1 = load i8*, i8** %gepbb12
  br label %bb3

bb2:
  %Abb2 = alloca i8*, i32 4
  %gepbb21 = getelementptr i8*, i8** %Abb2, i32 2
  store i8* %x, i8** %gepbb21, align 8
  %gepbb22 = getelementptr i8*, i8** %Abb2, i32 2
  %ybb2 = load i8*, i8** %gepbb22
  br label %bb3

bb3:
  %A = phi i8** [ %Abb1, %bb1 ], [ %Abb2, %bb2 ]
  %y = phi i8* [ %ybb1, %bb1 ], [ %ybb2, %bb2 ]
  tail call i8* @llvm.objc.retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @llvm.objc.release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @llvm.objc.release(i8* %x), !clang.imprecise_release !0
  ret void
}

; Make sure in the presence of allocas, if we find a cfghazard we do not perform
; code motion even if we are known safe. These two concepts are separate and
; should be treated as such.
;
; rdar://13949644

; CHECK: define void @test3a() {
; CHECK: entry:
; CHECK:   @llvm.objc.retainAutoreleasedReturnValue
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.retain
; CHECK: arraydestroy.body:
; CHECK:   @llvm.objc.release
; CHECK-NOT: @llvm.objc.release
; CHECK: arraydestroy.done:
; CHECK-NOT: @llvm.objc.release
; CHECK: arraydestroy.body1:
; CHECK:   @llvm.objc.release
; CHECK-NOT: @llvm.objc.release
; CHECK: arraydestroy.done1:
; CHECK: @llvm.objc.release
; CHECK: ret void
; CHECK: }
define void @test3a() {
entry:
  %keys = alloca [2 x i8*], align 16
  %objs = alloca [2 x i8*], align 16
  
  %call1 = call i8* @returner()
  %tmp0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call1)

  %objs.begin = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 0
  tail call i8* @llvm.objc.retain(i8* %call1)
  store i8* %call1, i8** %objs.begin, align 8
  %objs.elt = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 1
  tail call i8* @llvm.objc.retain(i8* %call1)
  store i8* %call1, i8** %objs.elt

  %call2 = call i8* @returner1()
  %call3 = call i8* @returner2()
  %keys.begin = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 0
  tail call i8* @llvm.objc.retain(i8* %call2)
  store i8* %call2, i8** %keys.begin, align 8
  %keys.elt = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 1
  tail call i8* @llvm.objc.retain(i8* %call3)
  store i8* %call3, i8** %keys.elt  
  
  %gep = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 2
  br label %arraydestroy.body

arraydestroy.body:
  %arraydestroy.elementPast = phi i8** [ %gep, %entry ], [ %arraydestroy.element, %arraydestroy.body ]
  %arraydestroy.element = getelementptr inbounds i8*, i8** %arraydestroy.elementPast, i64 -1
  %destroy_tmp = load i8*, i8** %arraydestroy.element, align 8
  call void @llvm.objc.release(i8* %destroy_tmp), !clang.imprecise_release !0
  %objs_ptr = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 0
  %arraydestroy.cmp = icmp eq i8** %arraydestroy.element, %objs_ptr
  br i1 %arraydestroy.cmp, label %arraydestroy.done, label %arraydestroy.body

arraydestroy.done:
  %gep1 = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 2
  br label %arraydestroy.body1

arraydestroy.body1:
  %arraydestroy.elementPast1 = phi i8** [ %gep1, %arraydestroy.done ], [ %arraydestroy.element1, %arraydestroy.body1 ]
  %arraydestroy.element1 = getelementptr inbounds i8*, i8** %arraydestroy.elementPast1, i64 -1
  %destroy_tmp1 = load i8*, i8** %arraydestroy.element1, align 8
  call void @llvm.objc.release(i8* %destroy_tmp1), !clang.imprecise_release !0
  %keys_ptr = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 0
  %arraydestroy.cmp1 = icmp eq i8** %arraydestroy.element1, %keys_ptr
  br i1 %arraydestroy.cmp1, label %arraydestroy.done1, label %arraydestroy.body1

arraydestroy.done1:
  call void @llvm.objc.release(i8* %call1), !clang.imprecise_release !0
  ret void
}

; Make sure that even though we stop said code motion we still allow for
; pointers to be removed if we are known safe in both directions.
;
; rdar://13949644

; CHECK: define void @test3b() {
; CHECK: entry:
; CHECK:   @llvm.objc.retainAutoreleasedReturnValue
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.retain
; CHECK:   @llvm.objc.retain
; CHECK: arraydestroy.body:
; CHECK:   @llvm.objc.release
; CHECK-NOT: @llvm.objc.release
; CHECK: arraydestroy.done:
; CHECK-NOT: @llvm.objc.release
; CHECK: arraydestroy.body1:
; CHECK:   @llvm.objc.release
; CHECK-NOT: @llvm.objc.release
; CHECK: arraydestroy.done1:
; CHECK: @llvm.objc.release
; CHECK: ret void
; CHECK: }
define void @test3b() {
entry:
  %keys = alloca [2 x i8*], align 16
  %objs = alloca [2 x i8*], align 16
  
  %call1 = call i8* @returner()
  %tmp0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call1)
  %tmp1 = tail call i8* @llvm.objc.retain(i8* %call1)

  %objs.begin = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 0
  tail call i8* @llvm.objc.retain(i8* %call1)
  store i8* %call1, i8** %objs.begin, align 8
  %objs.elt = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 1
  tail call i8* @llvm.objc.retain(i8* %call1)
  store i8* %call1, i8** %objs.elt

  %call2 = call i8* @returner1()
  %call3 = call i8* @returner2()
  %keys.begin = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 0
  tail call i8* @llvm.objc.retain(i8* %call2)
  store i8* %call2, i8** %keys.begin, align 8
  %keys.elt = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 1
  tail call i8* @llvm.objc.retain(i8* %call3)
  store i8* %call3, i8** %keys.elt  
  
  %gep = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 2
  br label %arraydestroy.body

arraydestroy.body:
  %arraydestroy.elementPast = phi i8** [ %gep, %entry ], [ %arraydestroy.element, %arraydestroy.body ]
  %arraydestroy.element = getelementptr inbounds i8*, i8** %arraydestroy.elementPast, i64 -1
  %destroy_tmp = load i8*, i8** %arraydestroy.element, align 8
  call void @llvm.objc.release(i8* %destroy_tmp), !clang.imprecise_release !0
  %objs_ptr = getelementptr inbounds [2 x i8*], [2 x i8*]* %objs, i64 0, i64 0
  %arraydestroy.cmp = icmp eq i8** %arraydestroy.element, %objs_ptr
  br i1 %arraydestroy.cmp, label %arraydestroy.done, label %arraydestroy.body

arraydestroy.done:
  %gep1 = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 2
  br label %arraydestroy.body1

arraydestroy.body1:
  %arraydestroy.elementPast1 = phi i8** [ %gep1, %arraydestroy.done ], [ %arraydestroy.element1, %arraydestroy.body1 ]
  %arraydestroy.element1 = getelementptr inbounds i8*, i8** %arraydestroy.elementPast1, i64 -1
  %destroy_tmp1 = load i8*, i8** %arraydestroy.element1, align 8
  call void @llvm.objc.release(i8* %destroy_tmp1), !clang.imprecise_release !0
  %keys_ptr = getelementptr inbounds [2 x i8*], [2 x i8*]* %keys, i64 0, i64 0
  %arraydestroy.cmp1 = icmp eq i8** %arraydestroy.element1, %keys_ptr
  br i1 %arraydestroy.cmp1, label %arraydestroy.done1, label %arraydestroy.body1

arraydestroy.done1:
  call void @llvm.objc.release(i8* %call1), !clang.imprecise_release !0
  call void @llvm.objc.release(i8* %call1), !clang.imprecise_release !0
  ret void
}

!0 = !{}

declare i32 @__gxx_personality_v0(...)
