; RUN: opt -opaque-pointers=0 < %s -passes=rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S | FileCheck %s


declare void @foo()

declare void @use(...) "gc-leaf-function"

define i64 addrspace(1)* @test1(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2, i1 %condition) gc "statepoint-example" {
; CHECK-LABEL: @test1
; CHECK-DAG: %obj.relocated
; CHECK-DAG: %obj2.relocated
entry:
  call void @foo() [ "deopt"() ]
  br label %joint

joint:                                            ; preds = %joint2, %entry
; CHECK-LABEL: joint:
; CHECK: %phi1 = phi i64 addrspace(1)* [ %obj.relocated.casted, %entry ], [ %obj3, %joint2 ]
  %phi1 = phi i64 addrspace(1)* [ %obj, %entry ], [ %obj3, %joint2 ]
  br i1 %condition, label %use, label %joint2

use:                                              ; preds = %joint
  br label %joint2

joint2:                                           ; preds = %use, %joint
; CHECK-LABEL: joint2:
; CHECK: %phi2 = phi i64 addrspace(1)* [ %obj.relocated.casted, %use ], [ %obj2.relocated.casted, %joint ]
; CHECK: %obj3 = getelementptr i64, i64 addrspace(1)* %obj2.relocated.casted, i32 1
  %phi2 = phi i64 addrspace(1)* [ %obj, %use ], [ %obj2, %joint ]
  %obj3 = getelementptr i64, i64 addrspace(1)* %obj2, i32 1
  br label %joint
}

declare i64 addrspace(1)* @generate_obj() "gc-leaf-function"

declare void @consume_obj(i64 addrspace(1)*) "gc-leaf-function"

declare i1 @rt() "gc-leaf-function"

define void @test2() gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
  %obj_init = call i64 addrspace(1)* @generate_obj()
  %obj = getelementptr i64, i64 addrspace(1)* %obj_init, i32 42
  br label %loop

loop:                                             ; preds = %loop.backedge, %entry
; CHECK: loop:
; CHECK-DAG: [ %obj_init.relocated.casted, %loop.backedge ]
; CHECK-DAG: [ %obj_init, %entry ]
; CHECK-DAG: [ %obj.relocated.casted, %loop.backedge ]
; CHECK-DAG: [ %obj, %entry ]
; CHECK-NOT: %location = getelementptr i64, i64 addrspace(1)* %obj, i32 %index
  %index = phi i32 [ 0, %entry ], [ %index.inc, %loop.backedge ]
  %location = getelementptr i64, i64 addrspace(1)* %obj, i32 %index
  call void @consume_obj(i64 addrspace(1)* %location)
  %index.inc = add i32 %index, 1
  %condition = call i1 @rt()
  br i1 %condition, label %loop_x, label %loop_y

loop_x:                                           ; preds = %loop
  br label %loop.backedge

loop.backedge:                                    ; preds = %loop_y, %loop_x
  call void @do_safepoint() [ "deopt"() ]
  br label %loop

loop_y:                                           ; preds = %loop
  br label %loop.backedge
}

declare void @some_call(i8 addrspace(1)*) "gc-leaf-function"

define void @relocate_merge(i1 %cnd, i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: @relocate_merge

bci_0:
  br i1 %cnd, label %if_branch, label %else_branch

if_branch:                                        ; preds = %bci_0
; CHECK-LABEL: if_branch:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  call void @foo() [ "deopt"() ]
  br label %join

else_branch:                                      ; preds = %bci_0
; CHECK-LABEL: else_branch:
; CHECK: gc.statepoint
; CHECK: gc.relocate
; We need to end up with a single relocation phi updated from both paths
  call void @foo() [ "deopt"() ]
  br label %join

join:                                             ; preds = %else_branch, %if_branch
; CHECK-LABEL: join:
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: [ %arg.relocated, %if_branch ]
; CHECK-DAG: [ %arg.relocated2, %else_branch ]
; CHECK-NOT: phi
  call void @some_call(i8 addrspace(1)* %arg)
  ret void
}

declare void @goo(i64)

declare i32 @moo(i64 addrspace(1)*)

; Make sure a use in a statepoint gets properly relocated at a previous one.
; This is basically just making sure that statepoints aren't accidentally
; treated specially.
define void @test3(i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.statepoint
entry:
  call void @goo(i64 undef) [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  %0 = call i32 @moo(i64 addrspace(1)* %obj) [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret void
}

declare i8 addrspace(1)* @boo()

; Check specifically for the case where the result of a statepoint needs to
; be relocated itself
define void @test4() gc "statepoint-example" {
; CHECK-LABEL: @test4
; CHECK: gc.statepoint
; CHECK: gc.result
; CHECK: gc.statepoint
; CHECK: [[RELOCATED:%[^ ]+]] = call {{.*}}gc.relocate
; CHECK: @use(i8 addrspace(1)* [[RELOCATED]])
  %1 = call i8 addrspace(1)* @boo() [ "deopt"() ]
  %2 = call i8 addrspace(1)* @boo() [ "deopt"() ]
  call void (...) @use(i8 addrspace(1)* %1)
  ret void
}

; Test updating a phi where not all inputs are live to begin with
define void @test5(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: test5
entry:
  %0 = call i8 addrspace(1)* @boo() [ "deopt"() ]
  switch i32 undef, label %kill [
    i32 10, label %merge
    i32 13, label %merge
  ]

kill:                                             ; preds = %entry
  br label %merge

merge:                                            ; preds = %kill, %entry, %entry
; CHECK: merge:
; CHECK: %test = phi i8 addrspace(1)
; CHECK-DAG: [ null, %kill ]
; CHECK-DAG: [ %arg.relocated, %entry ]
; CHECK-DAG: [ %arg.relocated, %entry ]
  %test = phi i8 addrspace(1)* [ null, %kill ], [ %arg, %entry ], [ %arg, %entry ]
  call void (...) @use(i8 addrspace(1)* %test)
  ret void
}

; Check to make sure we handle values live over an entry statepoint
define void @test6(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, i8 addrspace(1)* %arg3, i1 %c) gc "statepoint-example" {
; CHECK-LABEL: @test6
entry:
  br i1 %c, label %gc.safepoint_poll.exit2, label %do_safepoint

do_safepoint:                                     ; preds = %entry
; CHECK-LABEL: do_safepoint:
; CHECK: gc.statepoint
; CHECK: arg1.relocated =
; CHECK: arg2.relocated =
; CHECK: arg3.relocated =
  call void @foo() [ "deopt"(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, i8 addrspace(1)* %arg3) ]
  br label %gc.safepoint_poll.exit2

gc.safepoint_poll.exit2:                          ; preds = %do_safepoint, %entry
; CHECK-LABEL: gc.safepoint_poll.exit2:
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: [ %arg3, %entry ]
; CHECK-DAG: [ %arg3.relocated, %do_safepoint ]
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: [ %arg2, %entry ]
; CHECK-DAG: [ %arg2.relocated, %do_safepoint ]
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: [ %arg1, %entry ]
; CHECK-DAG:  [ %arg1.relocated, %do_safepoint ]
  call void (...) @use(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, i8 addrspace(1)* %arg3)
  ret void
}

; Check relocation in a loop nest where a relocation happens in the outer
; but not the inner loop
define void @test_outer_loop(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, i1 %cmp) gc "statepoint-example" {
; CHECK-LABEL: @test_outer_loop

bci_0:
  br label %outer-loop

outer-loop:                                       ; preds = %outer-inc, %bci_0
; CHECK-LABEL: outer-loop:
; CHECK: phi i8 addrspace(1)* [ %arg2, %bci_0 ], [ %arg2.relocated, %outer-inc ]
; CHECK: phi i8 addrspace(1)* [ %arg1, %bci_0 ], [ %arg1.relocated, %outer-inc ]
  br label %inner-loop

inner-loop:                                       ; preds = %inner-loop, %outer-loop
  br i1 %cmp, label %inner-loop, label %outer-inc

outer-inc:                                        ; preds = %inner-loop
; CHECK-LABEL: outer-inc:
; CHECK: %arg1.relocated
; CHECK: %arg2.relocated
  call void @foo() [ "deopt"(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2) ]
  br label %outer-loop
}

; Check that both inner and outer loops get phis when relocation is in
;  inner loop
define void @test_inner_loop(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, i1 %cmp) gc "statepoint-example" {
; CHECK-LABEL: @test_inner_loop

bci_0:
  br label %outer-loop

outer-loop:                                       ; preds = %outer-inc, %bci_0
; CHECK-LABEL: outer-loop:
; CHECK: phi i8 addrspace(1)* [ %arg2, %bci_0 ], [ %arg2.relocated, %outer-inc ]
; CHECK: phi i8 addrspace(1)* [ %arg1, %bci_0 ], [ %arg1.relocated, %outer-inc ]
  br label %inner-loop
; CHECK-LABEL: inner-loop
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: %outer-loop ]
; CHECK-DAG: [ %arg2.relocated, %inner-loop ]
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: %outer-loop ]
; CHECK-DAG: [ %arg1.relocated, %inner-loop ]
; CHECK: gc.statepoint
; CHECK: %arg1.relocated
; CHECK: %arg2.relocated

inner-loop:                                       ; preds = %inner-loop, %outer-loop
  call void @foo() [ "deopt"(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2) ]
  br i1 %cmp, label %inner-loop, label %outer-inc

outer-inc:                                        ; preds = %inner-loop
; CHECK-LABEL: outer-inc:
; This test shows why updating just those uses of the original value being
; relocated dominated by the inserted relocation is not always sufficient.
  br label %outer-loop
}

define i64 addrspace(1)* @test7(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2, i1 %condition) gc "statepoint-example" {
; CHECK-LABEL: @test7
entry:
  br i1 %condition, label %branch2, label %join

branch2:                                          ; preds = %entry
  br i1 %condition, label %callbb, label %join2

callbb:                                           ; preds = %branch2
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %join

join:                                             ; preds = %callbb, %entry
; CHECK-LABEL: join:
; CHECK: phi i64 addrspace(1)* [ %obj.relocated.casted, %callbb ], [ %obj, %entry ]
; CHECK: phi i64 addrspace(1)*
; CHECK-DAG: [ %obj, %entry ]
; CHECK-DAG: [ %obj2.relocated.casted, %callbb ]
  %phi1 = phi i64 addrspace(1)* [ %obj, %entry ], [ %obj2, %callbb ]
  br label %join2

join2:                                            ; preds = %join, %branch2
; CHECK-LABEL: join2:
; CHECK: phi2 = phi i64 addrspace(1)*
; CHECK-DAG: %join ]
; CHECK-DAG:  [ %obj2, %branch2 ]
  %phi2 = phi i64 addrspace(1)* [ %obj, %join ], [ %obj2, %branch2 ]
  ret i64 addrspace(1)* %phi2
}

declare void @do_safepoint()
