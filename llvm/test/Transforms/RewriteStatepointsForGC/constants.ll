; RUN: opt -S -rewrite-statepoints-for-gc %s | FileCheck %s

declare void @foo()
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

; constants don't get relocated.
define i8 @test() gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*)
entry:
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %res = load i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*)
  ret i8 %res
}


; Mostly just here to show reasonable code test can come from.  
define i8 @test2(i8 addrspace(1)* %p) gc "statepoint-example" {
; CHECK-LABEL: @test2
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: icmp
entry:
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %cmp = icmp eq i8 addrspace(1)* %p, null
  br i1 %cmp, label %taken, label %not_taken

taken:
  ret i8 0

not_taken:
  %cmp2 = icmp ne i8 addrspace(1)* %p, null
  br i1 %cmp2, label %taken, label %dead

dead:
  ; We see that dead can't be reached, but the optimizer might not.  It's 
  ; completely legal for it to exploit the fact that if dead executed, %p 
  ; would have to equal null.  This can produce intermediate states which 
  ; look like that of test above, even if arbitrary constant addresses aren't
  ; legal in the source language
  %addr = getelementptr i8, i8 addrspace(1)* %p, i32 15
  %res = load i8, i8addrspace(1)* %addr
  ret i8 %res
}

@G = addrspace(1) global i8 5

; Globals don't move and thus don't get relocated
define i8 @test3(i1 %always_true) gc "statepoint-example" {
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, i8 addrspace(1)* @G
entry:
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %res = load i8, i8 addrspace(1)* @G, align 1
  ret i8 %res
}


