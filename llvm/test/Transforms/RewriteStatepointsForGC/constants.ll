; RUN: opt -S -rewrite-statepoints-for-gc %s | FileCheck %s

declare void @foo()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

; constants don't get relocated.
define i8 @test() gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*)
entry:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
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
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
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
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %res = load i8, i8 addrspace(1)* @G, align 1
  ret i8 %res
}

; Even for source languages without constant references, we can
; see constants can show up along paths where the value is dead.
; This is particular relevant when computing bases of PHIs.  
define i8 addrspace(1)* @test4(i8 addrspace(1)* %p) gc "statepoint-example" {
; CHECK-LABEL: @test4
entry:
  %is_null = icmp eq i8 addrspace(1)* %p, null
  br i1 %is_null, label %split, label %join

split:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %arg_value_addr.i = getelementptr inbounds i8, i8 addrspace(1)* %p, i64 8
  %arg_value_addr_casted.i = bitcast i8 addrspace(1)* %arg_value_addr.i to i8 addrspace(1)* addrspace(1)*
  br label %join

join:
; CHECK-LABEL: join
; CHECK: %addr2.base =
  %addr2 = phi i8 addrspace(1)* addrspace(1)* [ %arg_value_addr_casted.i, %split ], [ inttoptr (i64 8 to i8 addrspace(1)* addrspace(1)*), %entry ]
  ;; NOTE: This particular example can be jump-threaded, but in general,
  ;; we can't, and have to deal with the resulting IR.
  br i1 %is_null, label %early-exit, label %use

early-exit:
  ret i8 addrspace(1)* null

use:
; CHECK-LABEL: use:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %res = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %addr2, align 1
  ret i8 addrspace(1)* %res
}

; Globals don't move and thus don't get relocated
define i8 addrspace(1)* @test5(i1 %always_true) gc "statepoint-example" {
; CHECK-LABEL: @test5
; CHECK: gc.statepoint
; CHECK-NEXT: %res = extractelement <2 x i8 addrspace(1)*> <i8 addrspace(1)* @G, i8 addrspace(1)* @G>, i32 0
entry:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %res = extractelement <2 x i8 addrspace(1)*> <i8 addrspace(1)* @G, i8 addrspace(1)* @G>, i32 0
  ret i8 addrspace(1)* %res
}
