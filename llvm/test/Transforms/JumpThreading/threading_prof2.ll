; RUN: opt -jump-threading -S < %s | FileCheck %s
; RUN: opt -passes=jump-threading -S < %s | FileCheck %s
define void @test() {
bb:
  %tmp = call i32 @a()
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb2
; CHECK: br i1 %tmp1,{{.*}} !prof ![[PROF1:[0-9]+]]

bb2:                                              
  %tmp3 = call i32 @b()
  %tmp4 = icmp ne i32 %tmp3, 1
  br label %bb5
; CHECK: br i1 %tmp4, {{.*}} !prof ![[PROF2:[0-9]+]]

bb5:                                             
  %tmp6 = phi i1 [ false, %bb ], [ %tmp4, %bb2 ]
  br i1 %tmp6, label %bb8, label %bb7, !prof !0

bb7:                                            
  call void @bar()
  br label %bb9

bb8: 
  call void @foo()
  br label %bb9

bb9:                                           
  ret void
}

declare void @bar()

declare void @foo()

declare i32 @a()

declare i32 @b()

!0 = !{!"branch_weights", i32 2146410443, i32 1073205}
;CHECK: ![[PROF1]] = !{!"branch_weights", i32 1073205, i32 2146410443}
;CHECK: ![[PROF2]] = !{!"branch_weights", i32 2146410443, i32 1073205}
