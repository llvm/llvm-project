; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.autorelease(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare void @llvm.objc.autoreleasePoolPop(ptr)
declare ptr @llvm.objc.autoreleasePoolPush()
declare ptr @llvm.objc.retainBlock(ptr)

declare ptr @objc_retainedObject(ptr)
declare ptr @objc_unretainedObject(ptr)
declare ptr @objc_unretainedPointer(ptr)

declare void @use_pointer(ptr)
declare void @callee()
declare void @callee_fnptr(ptr)
declare void @invokee()
declare ptr @returner()
declare ptr @returner1()
declare ptr @returner2()
declare void @bar(ptr)
declare void @use_alloca(ptr)

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare ptr @objc_msgSend(ptr, ptr, ...)


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

; CHECK: define void @test1a(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test1a(ptr %x) {
entry:
  %A = alloca ptr
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  store ptr %x, ptr %A, align 8
  %y = load ptr, ptr %A
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1b(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test1b(ptr %x) {
entry:
  %A = alloca ptr
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  store ptr %x, ptr %A, align 8
  %y = load ptr, ptr %A
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}


; CHECK: define void @test1c(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test1c(ptr %x) {
entry:
  %A = alloca ptr, i32 3
  %gep = getelementptr ptr, ptr %A, i32 2
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  store ptr %x, ptr %gep, align 8
  %y = load ptr, ptr %gep
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}


; CHECK: define void @test1d(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test1d(ptr %x) {
entry:
  br i1 undef, label %use_allocaA, label %use_allocaB

use_allocaA:
  %allocaA = alloca ptr
  br label %exit

use_allocaB:
  %allocaB = alloca ptr
  br label %exit

exit:
  %A = phi ptr [ %allocaA, %use_allocaA ], [ %allocaB, %use_allocaB ]
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  store ptr %x, ptr %A, align 8
  %y = load ptr, ptr %A
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1e(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test1e(ptr %x) {
entry:
  br i1 undef, label %use_allocaA, label %use_allocaB

use_allocaA:
  %allocaA = alloca ptr, i32 4
  br label %exit

use_allocaB:
  %allocaB = alloca ptr, i32 4
  br label %exit

exit:
  %A = phi ptr [ %allocaA, %use_allocaA ], [ %allocaB, %use_allocaB ]
  %gep = getelementptr ptr, ptr %A, i32 2
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  store ptr %x, ptr %gep, align 8
  %y = load ptr, ptr %gep
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1f(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test1f(ptr %x) {
entry:
  %allocaOne = alloca ptr
  %allocaTwo = alloca ptr
  %A = select i1 undef, ptr %allocaOne, ptr %allocaTwo
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  store ptr %x, ptr %A, align 8
  %y = load ptr, ptr %A
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; Make sure that if a store is in a different basic block we handle known safe
; conservatively.


; CHECK: define void @test2a(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test2a(ptr %x) {
entry:
  %A = alloca ptr
  store ptr %x, ptr %A, align 8
  %y = load ptr, ptr %A
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2b(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test2b(ptr %x) {
entry:
  %A = alloca ptr
  store ptr %x, ptr %A, align 8
  %y = load ptr, ptr %A
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call ptr @llvm.objc.retain(ptr %x)
  tail call ptr @llvm.objc.retain(ptr %x)
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2c(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test2c(ptr %x) {
entry:
  %A = alloca ptr, i32 3
  %gep1 = getelementptr ptr, ptr %A, i32 2
  store ptr %x, ptr %gep1, align 8
  %gep2 = getelementptr ptr, ptr %A, i32 2
  %y = load ptr, ptr %gep2
  tail call ptr @llvm.objc.retain(ptr %x)
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call ptr @llvm.objc.retain(ptr %x)
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2d(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.retain(ptr %x)
; CHECK: @llvm.objc.release(ptr %y)
; CHECK: @llvm.objc.release(ptr %x)
; CHECK: ret void
; CHECK: }
define void @test2d(ptr %x) {
entry:
  tail call ptr @llvm.objc.retain(ptr %x)
  br label %bb1

bb1:
  %Abb1 = alloca ptr, i32 3
  %gepbb11 = getelementptr ptr, ptr %Abb1, i32 2
  store ptr %x, ptr %gepbb11, align 8
  %gepbb12 = getelementptr ptr, ptr %Abb1, i32 2
  %ybb1 = load ptr, ptr %gepbb12
  br label %bb3

bb2:
  %Abb2 = alloca ptr, i32 4
  %gepbb21 = getelementptr ptr, ptr %Abb2, i32 2
  store ptr %x, ptr %gepbb21, align 8
  %gepbb22 = getelementptr ptr, ptr %Abb2, i32 2
  %ybb2 = load ptr, ptr %gepbb22
  br label %bb3

bb3:
  %A = phi ptr [ %Abb1, %bb1 ], [ %Abb2, %bb2 ]
  %y = phi ptr [ %ybb1, %bb1 ], [ %ybb2, %bb2 ]
  tail call ptr @llvm.objc.retain(ptr %x)
  call void @use_alloca(ptr %A)
  call void @llvm.objc.release(ptr %y), !clang.imprecise_release !0
  call void @use_pointer(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
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
  %keys = alloca [2 x ptr], align 16
  %objs = alloca [2 x ptr], align 16
  
  %call1 = call ptr @returner()
  %tmp0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call1)

  tail call ptr @llvm.objc.retain(ptr %call1)
  store ptr %call1, ptr %objs, align 8
  %objs.elt = getelementptr inbounds [2 x ptr], ptr %objs, i64 0, i64 1
  tail call ptr @llvm.objc.retain(ptr %call1)
  store ptr %call1, ptr %objs.elt

  %call2 = call ptr @returner1()
  %call3 = call ptr @returner2()
  tail call ptr @llvm.objc.retain(ptr %call2)
  store ptr %call2, ptr %keys, align 8
  %keys.elt = getelementptr inbounds [2 x ptr], ptr %keys, i64 0, i64 1
  tail call ptr @llvm.objc.retain(ptr %call3)
  store ptr %call3, ptr %keys.elt  
  
  %gep = getelementptr inbounds [2 x ptr], ptr %objs, i64 0, i64 2
  br label %arraydestroy.body

arraydestroy.body:
  %arraydestroy.elementPast = phi ptr [ %gep, %entry ], [ %arraydestroy.element, %arraydestroy.body ]
  %arraydestroy.element = getelementptr inbounds ptr, ptr %arraydestroy.elementPast, i64 -1
  %destroy_tmp = load ptr, ptr %arraydestroy.element, align 8
  call void @llvm.objc.release(ptr %destroy_tmp), !clang.imprecise_release !0
  %arraydestroy.cmp = icmp eq ptr %arraydestroy.element, %objs
  br i1 %arraydestroy.cmp, label %arraydestroy.done, label %arraydestroy.body

arraydestroy.done:
  %gep1 = getelementptr inbounds [2 x ptr], ptr %keys, i64 0, i64 2
  br label %arraydestroy.body1

arraydestroy.body1:
  %arraydestroy.elementPast1 = phi ptr [ %gep1, %arraydestroy.done ], [ %arraydestroy.element1, %arraydestroy.body1 ]
  %arraydestroy.element1 = getelementptr inbounds ptr, ptr %arraydestroy.elementPast1, i64 -1
  %destroy_tmp1 = load ptr, ptr %arraydestroy.element1, align 8
  call void @llvm.objc.release(ptr %destroy_tmp1), !clang.imprecise_release !0
  %arraydestroy.cmp1 = icmp eq ptr %arraydestroy.element1, %keys
  br i1 %arraydestroy.cmp1, label %arraydestroy.done1, label %arraydestroy.body1

arraydestroy.done1:
  call void @llvm.objc.release(ptr %call1), !clang.imprecise_release !0
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
  %keys = alloca [2 x ptr], align 16
  %objs = alloca [2 x ptr], align 16
  
  %call1 = call ptr @returner()
  %tmp0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call1)
  %tmp1 = tail call ptr @llvm.objc.retain(ptr %call1)

  tail call ptr @llvm.objc.retain(ptr %call1)
  store ptr %call1, ptr %objs, align 8
  %objs.elt = getelementptr inbounds [2 x ptr], ptr %objs, i64 0, i64 1
  tail call ptr @llvm.objc.retain(ptr %call1)
  store ptr %call1, ptr %objs.elt

  %call2 = call ptr @returner1()
  %call3 = call ptr @returner2()
  tail call ptr @llvm.objc.retain(ptr %call2)
  store ptr %call2, ptr %keys, align 8
  %keys.elt = getelementptr inbounds [2 x ptr], ptr %keys, i64 0, i64 1
  tail call ptr @llvm.objc.retain(ptr %call3)
  store ptr %call3, ptr %keys.elt  
  
  %gep = getelementptr inbounds [2 x ptr], ptr %objs, i64 0, i64 2
  br label %arraydestroy.body

arraydestroy.body:
  %arraydestroy.elementPast = phi ptr [ %gep, %entry ], [ %arraydestroy.element, %arraydestroy.body ]
  %arraydestroy.element = getelementptr inbounds ptr, ptr %arraydestroy.elementPast, i64 -1
  %destroy_tmp = load ptr, ptr %arraydestroy.element, align 8
  call void @llvm.objc.release(ptr %destroy_tmp), !clang.imprecise_release !0
  %arraydestroy.cmp = icmp eq ptr %arraydestroy.element, %objs
  br i1 %arraydestroy.cmp, label %arraydestroy.done, label %arraydestroy.body

arraydestroy.done:
  %gep1 = getelementptr inbounds [2 x ptr], ptr %keys, i64 0, i64 2
  br label %arraydestroy.body1

arraydestroy.body1:
  %arraydestroy.elementPast1 = phi ptr [ %gep1, %arraydestroy.done ], [ %arraydestroy.element1, %arraydestroy.body1 ]
  %arraydestroy.element1 = getelementptr inbounds ptr, ptr %arraydestroy.elementPast1, i64 -1
  %destroy_tmp1 = load ptr, ptr %arraydestroy.element1, align 8
  call void @llvm.objc.release(ptr %destroy_tmp1), !clang.imprecise_release !0
  %arraydestroy.cmp1 = icmp eq ptr %arraydestroy.element1, %keys
  br i1 %arraydestroy.cmp1, label %arraydestroy.done1, label %arraydestroy.body1

arraydestroy.done1:
  call void @llvm.objc.release(ptr %call1), !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %call1), !clang.imprecise_release !0
  ret void
}

!0 = !{}

declare i32 @__gxx_personality_v0(...)
