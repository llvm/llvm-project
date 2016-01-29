; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s

; constants don't get relocated.
@G = addrspace(1) global i8 5

declare void @foo()

define i8 @test() gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*)
; Mostly just here to show reasonable code test can come from.  
entry:
  call void @foo() [ "deopt"() ]
  %res = load i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*)
  ret i8 %res
}

define i8 @test2(i8 addrspace(1)* %p) gc "statepoint-example" {
; CHECK-LABEL: @test2
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: icmp
; Globals don't move and thus don't get relocated
entry:
  call void @foo() [ "deopt"() ]
  %cmp = icmp eq i8 addrspace(1)* %p, null
  br i1 %cmp, label %taken, label %not_taken

taken:                                            ; preds = %not_taken, %entry
  ret i8 0

not_taken:                                        ; preds = %entry
  %cmp2 = icmp ne i8 addrspace(1)* %p, null
  br i1 %cmp2, label %taken, label %dead

dead:                                             ; preds = %not_taken
  %addr = getelementptr i8, i8 addrspace(1)* %p, i32 15
  %res = load i8, i8 addrspace(1)* %addr
  ret i8 %res
}

define i8 @test3(i1 %always_true) gc "statepoint-example" {
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, i8 addrspace(1)* @G
entry:
  call void @foo() [ "deopt"() ]
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
  call void @foo()
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
  call void @foo()
  %res = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %addr2, align 1
  ret i8 addrspace(1)* %res
}

; Globals don't move and thus don't get relocated
define i8 addrspace(1)* @test5(i1 %always_true) gc "statepoint-example" {
; CHECK-LABEL: @test5
; CHECK: gc.statepoint
; CHECK-NEXT: %res = extractelement <2 x i8 addrspace(1)*> <i8 addrspace(1)* @G, i8 addrspace(1)* @G>, i32 0
entry:
  call void @foo()
  %res = extractelement <2 x i8 addrspace(1)*> <i8 addrspace(1)* @G, i8 addrspace(1)* @G>, i32 0
  ret i8 addrspace(1)* %res
}
