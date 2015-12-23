; RUN: opt %s -rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S 2>&1 | FileCheck %s


declare void @foo()
declare void @use(...)

define i64 addrspace(1)* @test1(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2, i1 %condition) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test1
; CHECK-DAG: %obj.relocated
; CHECK-DAG: %obj2.relocated
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %joint

joint:
; CHECK-LABEL: joint:
; CHECK: %phi1 = phi i64 addrspace(1)* [ %obj.relocated.casted, %entry ], [ %obj3, %joint2 ]
  %phi1 = phi i64 addrspace(1)* [ %obj, %entry ], [ %obj3, %joint2 ]
  br i1 %condition, label %use, label %joint2

use:
  br label %joint2

joint2:
; CHECK-LABEL: joint2:
; CHECK: %phi2 = phi i64 addrspace(1)* [ %obj.relocated.casted, %use ], [ %obj2.relocated.casted, %joint ]
; CHECK: %obj3 = getelementptr i64, i64 addrspace(1)* %obj2.relocated.casted, i32 1
  %phi2 = phi i64 addrspace(1)* [ %obj, %use ], [ %obj2, %joint ]
  %obj3 = getelementptr i64, i64 addrspace(1)* %obj2, i32 1
  br label %joint
}

declare i64 addrspace(1)* @generate_obj()

declare void @consume_obj(i64 addrspace(1)*)

declare i1 @rt()

define void @test2() gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
  %obj_init = call i64 addrspace(1)* @generate_obj()
  %obj = getelementptr i64, i64 addrspace(1)* %obj_init, i32 42
  br label %loop

loop:
; CHECK: loop:
; CHECK-DAG: [ %obj_init.relocated.casted, %loop.backedge ]
; CHECK-DAG: [ %obj_init, %entry ]
; CHECK-DAG: [ %obj.relocated.casted, %loop.backedge ]
; CHECK-DAG: [ %obj, %entry ]
  %index = phi i32 [ 0, %entry ], [ %index.inc, %loop.backedge ]
; CHECK-NOT: %location = getelementptr i64, i64 addrspace(1)* %obj, i32 %index
  %location = getelementptr i64, i64 addrspace(1)* %obj, i32 %index
  call void @consume_obj(i64 addrspace(1)* %location)
  %index.inc = add i32 %index, 1
  %condition = call i1 @rt()
  br i1 %condition, label %loop_x, label %loop_y

loop_x:
  br label %loop.backedge

loop.backedge:
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  br label %loop

loop_y:
  br label %loop.backedge
}

declare void @some_call(i8 addrspace(1)*)

define void @relocate_merge(i1 %cnd, i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: @relocate_merge
bci_0:
  br i1 %cnd, label %if_branch, label %else_branch

if_branch:
; CHECK-LABEL: if_branch:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %join

else_branch:
; CHECK-LABEL: else_branch:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  %safepoint_token1 = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %join

join:
; We need to end up with a single relocation phi updated from both paths 
; CHECK-LABEL: join:
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: [ %arg.relocated, %if_branch ]
; CHECK-DAG: [ %arg.relocated4, %else_branch ]
; CHECK-NOT: phi
  call void (i8 addrspace(1)*) @some_call(i8 addrspace(1)* %arg)
  ret void
}

; Make sure a use in a statepoint gets properly relocated at a previous one.  
; This is basically just making sure that statepoints aren't accidentally 
; treated specially.
define void @test3(i64 addrspace(1)* %obj) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.statepoint
  %safepoint_token = call i32 (i64, i32, void (i64)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi64f(i64 0, i32 0, void (i64)* undef, i32 1, i32 0, i64 undef, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  %safepoint_token1 = call i32 (i64, i32, i32 (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32p1i64f(i64 0, i32 0, i32 (i64 addrspace(1)*)* undef, i32 1, i32 0, i64 addrspace(1)* %obj, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret void
}

; Check specifically for the case where the result of a statepoint needs to 
; be relocated itself
define void @test4() gc "statepoint-example" {
; CHECK-LABEL: @test4
; CHECK: gc.statepoint
; CHECK: gc.result
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK: @use(i8 addrspace(1)* %res.relocated)
  %safepoint_token2 = tail call i32 (i64, i32, i8 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i8f(i64 0, i32 0, i8 addrspace(1)* ()* undef, i32 0, i32 0, i32 0, i32 0)
  %res = call i8 addrspace(1)* @llvm.experimental.gc.result.p1i8(i32 %safepoint_token2)
  call i32 (i64, i32, i8 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i8f(i64 0, i32 0, i8 addrspace(1)* ()* undef, i32 0, i32 0, i32 0, i32 0)
  call void (...) @use(i8 addrspace(1)* %res)
  unreachable
}


; Test updating a phi where not all inputs are live to begin with
define void @test5(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: test5
entry:
  call i32 (i64, i32, i8 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i8f(i64 0, i32 0, i8 addrspace(1)* ()* undef, i32 0, i32 0, i32 0, i32 0)
  switch i32 undef, label %kill [
    i32 10, label %merge
    i32 13, label %merge
  ]

kill:
  br label %merge

merge:
; CHECK: merge:
; CHECK: %test = phi i8 addrspace(1)
; CHECK-DAG: [ null, %kill ]
; CHECK-DAG: [ %arg.relocated, %entry ]
; CHECK-DAG: [ %arg.relocated, %entry ]
  %test = phi i8 addrspace(1)* [ null, %kill ], [ %arg, %entry ], [ %arg, %entry ]
  call void (...) @use(i8 addrspace(1)* %test)
  unreachable
}


; Check to make sure we handle values live over an entry statepoint
define void @test6(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, 
                  i8 addrspace(1)* %arg3) gc "statepoint-example" {
; CHECK-LABEL: @test6
entry:
  br i1 undef, label %gc.safepoint_poll.exit2, label %do_safepoint

do_safepoint:
; CHECK-LABEL: do_safepoint:
; CHECK: gc.statepoint
; CHECK: arg1.relocated = 
; CHECK: arg2.relocated = 
; CHECK: arg3.relocated = 
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 3, i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, i8 addrspace(1)* %arg3)
  br label %gc.safepoint_poll.exit2

gc.safepoint_poll.exit2:
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
define void @test_outer_loop(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, 
                  i1 %cmp) gc "statepoint-example" {
; CHECK-LABEL: @test_outer_loop
bci_0:
  br label %outer-loop

outer-loop:
; CHECK-LABEL: outer-loop:
; CHECK: phi i8 addrspace(1)* [ %arg2, %bci_0 ], [ %arg2.relocated, %outer-inc ]
; CHECK: phi i8 addrspace(1)* [ %arg1, %bci_0 ], [ %arg1.relocated, %outer-inc ]
  br label %inner-loop

inner-loop:
  br i1 %cmp, label %inner-loop, label %outer-inc

outer-inc:
; CHECK-LABEL: outer-inc:
; CHECK: %arg1.relocated
; CHECK: %arg2.relocated
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 2, i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2)
  br label %outer-loop
}

; Check that both inner and outer loops get phis when relocation is in
;  inner loop
define void @test_inner_loop(i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2, 
                  i1 %cmp) gc "statepoint-example" {
; CHECK-LABEL: @test_inner_loop
bci_0:
  br label %outer-loop

outer-loop:
; CHECK-LABEL: outer-loop:
; CHECK: phi i8 addrspace(1)* [ %arg2, %bci_0 ], [ %arg2.relocated, %outer-inc ]
; CHECK: phi i8 addrspace(1)* [ %arg1, %bci_0 ], [ %arg1.relocated, %outer-inc ]
  br label %inner-loop

inner-loop:
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
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 2, i8 addrspace(1)* %arg1, i8 addrspace(1)* %arg2)
  br i1 %cmp, label %inner-loop, label %outer-inc

outer-inc:
; CHECK-LABEL: outer-inc:
  br label %outer-loop
}


; This test shows why updating just those uses of the original value being
; relocated dominated by the inserted relocation is not always sufficient.
define i64 addrspace(1)* @test7(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2, i1 %condition) gc "statepoint-example" {
; CHECK-LABEL: @test7
entry:
  br i1 %condition, label %branch2, label %join

branch2:
  br i1 %condition, label %callbb, label %join2

callbb:
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %join

join:
; CHECK-LABEL: join:
; CHECK: phi i64 addrspace(1)* [ %obj.relocated.casted, %callbb ], [ %obj, %entry ]
; CHECK: phi i64 addrspace(1)* 
; CHECK-DAG: [ %obj, %entry ]
; CHECK-DAG: [ %obj2.relocated.casted, %callbb ]
  ; This is a phi outside the dominator region of the new defs inserted by
  ; the safepoint, BUT we can't stop the search here or we miss the second
  ; phi below.
  %phi1 = phi i64 addrspace(1)* [ %obj, %entry ], [ %obj2, %callbb ]
  br label %join2

join2:
; CHECK-LABEL: join2:
; CHECK: phi2 = phi i64 addrspace(1)* 
; CHECK-DAG: %join ] 
; CHECK-DAG:  [ %obj2, %branch2 ]
  %phi2 = phi i64 addrspace(1)* [ %obj, %join ], [ %obj2, %branch2 ]
  ret i64 addrspace(1)* %phi2
}


declare void @do_safepoint()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 @llvm.experimental.gc.statepoint.p0f_p1i8f(i64, i32, i8 addrspace(1)* ()*, i32, i32, ...)
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidi64f(i64, i32, void (i64)*, i32, i32, ...)
declare i32 @llvm.experimental.gc.statepoint.p0f_i32p1i64f(i64, i32, i32 (i64 addrspace(1)*)*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.result.p1i8(i32) #3



