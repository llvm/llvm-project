; RUN: opt -S -rewrite-statepoints-for-gc -rs4gc-use-deopt-bundles < %s | FileCheck %s

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
