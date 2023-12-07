; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

target datalayout = "e-ni:1:6"

; constants don't get relocated.
@G = addrspace(1) global i8 5

declare void @foo()

define i8 @test() gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, ptr addrspace(1) inttoptr (i64 15 to ptr addrspace(1))
; Mostly just here to show reasonable code test can come from.
entry:
  call void @foo() [ "deopt"() ]
  %res = load i8, ptr addrspace(1) inttoptr (i64 15 to ptr addrspace(1))
  ret i8 %res
}

define i8 @test2(ptr addrspace(1) %p) gc "statepoint-example" {
; CHECK-LABEL: @test2
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: icmp
; Globals don't move and thus don't get relocated
entry:
  call void @foo() [ "deopt"() ]
  %cmp = icmp eq ptr addrspace(1) %p, null
  br i1 %cmp, label %taken, label %not_taken

taken:                                            ; preds = %not_taken, %entry
  ret i8 0

not_taken:                                        ; preds = %entry
  %cmp2 = icmp ne ptr addrspace(1) %p, null
  br i1 %cmp2, label %taken, label %dead

dead:                                             ; preds = %not_taken
  %addr = getelementptr i8, ptr addrspace(1) %p, i32 15
  %res = load i8, ptr addrspace(1) %addr
  ret i8 %res
}

define i8 @test3(i1 %always_true) gc "statepoint-example" {
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK-NEXT: load i8, ptr addrspace(1) @G
entry:
  call void @foo() [ "deopt"() ]
  %res = load i8, ptr addrspace(1) @G, align 1
  ret i8 %res
}

; Even for source languages without constant references, we can
; see constants can show up along paths where the value is dead.
; This is particular relevant when computing bases of PHIs.
define ptr addrspace(1) @test4(ptr addrspace(1) %p) gc "statepoint-example" {
; CHECK-LABEL: @test4
entry:
  %is_null = icmp eq ptr addrspace(1) %p, null
  br i1 %is_null, label %split, label %join

split:
  call void @foo()
  %arg_value_addr.i = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  br label %join

join:
; CHECK-LABEL: join
; CHECK: %addr2.base =
  %addr2 = phi ptr addrspace(1) [ %arg_value_addr.i, %split ], [ inttoptr (i64 8 to ptr addrspace(1)), %entry ]
  ;; NOTE: This particular example can be jump-threaded, but in general,
  ;; we can't, and have to deal with the resulting IR.
  br i1 %is_null, label %early-exit, label %use

early-exit:
  ret ptr addrspace(1) null

use:
; CHECK-LABEL: use:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  call void @foo()
  %res = load ptr addrspace(1), ptr addrspace(1) %addr2, align 1
  ret ptr addrspace(1) %res
}

; Globals don't move and thus don't get relocated
define ptr addrspace(1) @test5(i1 %always_true) gc "statepoint-example" {
; CHECK-LABEL: @test5
; CHECK: gc.statepoint
; CHECK-NEXT: %res = extractelement <2 x ptr addrspace(1)> <ptr addrspace(1) @G, ptr addrspace(1) @G>, i32 0
entry:
  call void @foo()
  %res = extractelement <2 x ptr addrspace(1)> <ptr addrspace(1) @G, ptr addrspace(1) @G>, i32 0
  ret ptr addrspace(1) %res
}

define ptr addrspace(1) @test6(i64 %arg) gc "statepoint-example" {
entry:
  ; Don't fail any assertions and don't record null as a live value
  ; CHECK-LABEL: test6
  ; CHECK: gc.statepoint
  ; CHECK-NOT: call {{.*}}gc.relocate
  %load_addr = getelementptr i8, ptr addrspace(1) null, i64 %arg
  call void @foo() [ "deopt"() ]
  ret ptr addrspace(1) %load_addr
}

define ptr addrspace(1) @test7(i64 %arg) gc "statepoint-example" {
entry:
  ; Same as test7 but use regular constant instead of a null
  ; CHECK-LABEL: test7
  ; CHECK: gc.statepoint
  ; CHECK-NOT: call {{.*}}gc.relocate
  %load_addr = getelementptr i8, ptr addrspace(1) inttoptr (i64 15 to ptr addrspace(1)), i64 %arg
  call void @foo() [ "deopt"() ]
  ret ptr addrspace(1) %load_addr
}

define i8 @test8(ptr addrspace(1) %p) gc "statepoint-example" {
; Checks that base( phi(gep null, oop) ) = phi(null, base(oop)) and that we
; correctly relocate this value
; CHECK-LABEL: @test8
entry:
  %is_null = icmp eq ptr addrspace(1) %p, null
  br i1 %is_null, label %null.crit-edge, label %not-null

not-null:
  %load_addr = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  br label %join

null.crit-edge:
  %load_addr.const = getelementptr inbounds i8, ptr addrspace(1) null, i64 8
  br label %join

join:
  %addr = phi ptr addrspace(1) [ %load_addr, %not-null ], [%load_addr.const, %null.crit-edge]
  ; CHECK: %addr.base = phi ptr addrspace(1)
  ; CHECK-DAG: [ %p, %not-null ]
  ; CHECK-DAG: [ null, %null.crit-edge ]
  ; CHECK: gc.statepoint
  call void @foo() [ "deopt"() ]
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%addr.base, %addr.base)
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%addr.base, %addr)
  br i1 %is_null, label %early-exit, label %use

early-exit:
  ret i8 0

use:
  %res = load i8, ptr addrspace(1) %addr, align 1
  ret i8 %res
}

define i8 @test9(ptr addrspace(1) %p) gc "statepoint-example" {
; Checks that base( phi(inttoptr, oop) ) = phi(null, base(oop)) and that we
; correctly relocate this value
; CHECK-LABEL: @test9
entry:
  %is_null = icmp eq ptr addrspace(1) %p, null
  br i1 %is_null, label %null.crit-edge, label %not-null

not-null:
  %load_addr = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  br label %join

null.crit-edge:
  br label %join

join:
  %addr = phi ptr addrspace(1) [ %load_addr, %not-null ], [inttoptr (i64 8 to ptr addrspace(1)), %null.crit-edge]
  ; CHECK: %addr.base = phi ptr addrspace(1)
  ; CHECK-DAG: [ %p, %not-null ]
  ; CHECK-DAG: [ null, %null.crit-edge ]
  ; CHECK: gc.statepoint
  call void @foo() [ "deopt"() ]
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%addr.base, %addr.base)
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%addr.base, %addr)
  br i1 %is_null, label %early-exit, label %use

early-exit:
  ret i8 0

use:
  %res = load i8, ptr addrspace(1) %addr, align 1
  ret i8 %res
}

define i8 @test10(ptr addrspace(1) %p) gc "statepoint-example" {
; Checks that base( phi(const gep, oop) ) = phi(null, base(oop)) and that we
; correctly relocate this value
; CHECK-LABEL: @test10
entry:
  %is_null = icmp eq ptr addrspace(1) %p, null
  br i1 %is_null, label %null.crit-edge, label %not-null

not-null:
  %load_addr = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  br label %join

null.crit-edge:
  br label %join

join:
  %addr = phi ptr addrspace(1) [ %load_addr, %not-null ], [getelementptr (i8, ptr addrspace(1) null, i64 8), %null.crit-edge]
  ; CHECK: %addr.base = phi ptr addrspace(1)
  ; CHECK-DAG: [ %p, %not-null ]
  ; CHECK-DAG: [ null, %null.crit-edge ]
  ; CHECK: gc.statepoint
  call void @foo() [ "deopt"() ]
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%addr.base, %addr.base)
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%addr.base, %addr)
  br i1 %is_null, label %early-exit, label %use

early-exit:
  ret i8 0

use:
  %res = load i8, ptr addrspace(1) %addr, align 1
  ret i8 %res
}

define ptr addrspace(1) @test11(i1 %c) gc "statepoint-example" {
; CHECK-LABEL: @test11
; Checks that base( select(const1, const2) ) == null and that we don't record
; such value in the oop map
entry:
  %val = select i1 %c, ptr addrspace(1) inttoptr (i64 8 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 15 to ptr addrspace(1))
  ; CHECK: gc.statepoint
  ; CHECK-NOT: call {{.*}}gc.relocate
  call void @foo() [ "deopt"() ]
  ret ptr addrspace(1) %val
}


define <2 x ptr addrspace(1)> @test12(i1 %c) gc "statepoint-example" {
; CHECK-LABEL: @test12
; Same as test11 but with vectors
entry:
  %val = select i1 %c, <2 x ptr addrspace(1)> <ptr addrspace(1) inttoptr (i64 5 to ptr addrspace(1)),
                                                ptr addrspace(1) inttoptr (i64 15 to ptr addrspace(1))>,
                       <2 x ptr addrspace(1)> <ptr addrspace(1) inttoptr (i64 30 to ptr addrspace(1)),
                                                ptr addrspace(1) inttoptr (i64 60 to ptr addrspace(1))>
  ; CHECK: gc.statepoint
  ; CHECK-NOT: call {{.*}}gc.relocate
  call void @foo() [ "deopt"() ]
  ret <2 x ptr addrspace(1)> %val
}

define <2 x ptr addrspace(1)> @test13(i1 %c, <2 x ptr addrspace(1)> %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test13
; Similar to test8, test9 and test10 but with vectors
entry:
  %val = select i1 %c, <2 x ptr addrspace(1)> %ptr,
                       <2 x ptr addrspace(1)> <ptr addrspace(1) inttoptr (i64 30 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 60 to ptr addrspace(1))>
  ; CHECK: %val.base = select i1 %c, <2 x ptr addrspace(1)> %ptr, <2 x ptr addrspace(1)> zeroinitializer, !is_base_value !0
  ; CHECK: gc.statepoint
  call void @foo() [ "deopt"() ]
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%val.base, %val.base)
  ; CHECK-DAG: call {{.*}}gc.relocate{{.*}}(%val.base, %val)
  ret <2 x ptr addrspace(1)> %val
}
