; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)

%Ty = type <{ i32, i32 }>

; Check if the pass doesn't modify anything that doesn't need changing. We feed
; an unused argument to each function to lure it into changing _something_ about
; the function and then changing too much.

; This checks if the return value attributes are not removed
; CHECK: define internal zeroext i32 @test1.argelim() #1
define internal zeroext i32 @test1(i32 %DEADARG1) nounwind {
;
;
  ret i32 1
}

; This checks if the struct doesn't get non-packed
; CHECK-LABEL: define internal <{ i32, i32 }> @test2.argelim(
define internal <{ i32, i32 }> @test2(i32 %DEADARG1) {
;
;
  ret <{ i32, i32 }> <{ i32 1, i32 2 }>
}

; We use this external function to make sure the return values don't become dead
declare void @user(i32, <{ i32, i32 }>)

define void @caller() {
;
;
  %B = call i32 @test1(i32 1)
  %C = call <{ i32, i32 }> @test2(i32 2)
  call void @user(i32 %B, <{ i32, i32 }> %C)
  ret void
}

; We can't remove 'this' here, as that would put argmem in ecx instead of
; memory.
define internal x86_thiscallcc i32 @unused_this(ptr %this, ptr inalloca(i32) %argmem) {
;
;
  %v = load i32, ptr %argmem
  ret i32 %v
}
; CHECK-LABEL: define internal x86_thiscallcc i32 @unused_this(ptr %this, ptr inalloca(i32) %argmem)

define i32 @caller2() {
;
;
  %t = alloca i32
  %m = alloca inalloca i32
  store i32 42, ptr %m
  %v = call x86_thiscallcc i32 @unused_this(ptr %t, ptr inalloca(i32) %m)
  ret i32 %v
}

; We can't remove 'this' here, as that would put argmem in ecx instead of
; memory.
define internal x86_thiscallcc i32 @unused_this_preallocated(ptr %this, ptr preallocated(i32) %argmem) {
;
;
  %v = load i32, ptr %argmem
  ret i32 %v
}
; CHECK-LABEL: define internal x86_thiscallcc i32 @unused_this_preallocated(ptr %this, ptr preallocated(i32) %argmem)

define i32 @caller3() {
;
;
  %t = alloca i32
  %c = call token @llvm.call.preallocated.setup(i32 1)
  %M = call ptr @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  store i32 42, ptr %M
  %v = call x86_thiscallcc i32 @unused_this_preallocated(ptr %t, ptr preallocated(i32) %M) ["preallocated"(token %c)]
  ret i32 %v
}

; CHECK: attributes #0 = { nocallback nofree nosync nounwind willreturn }
; CHECK: attributes #1 = { nounwind }
