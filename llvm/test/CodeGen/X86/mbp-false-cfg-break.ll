; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

define void @test(i1 %cnd) !prof !{!"function_entry_count", i64 1024} {
; CHECK-LABEL: @test
; Using the assembly comments to indicate block order..
; CHECK: # %loop
; CHECK: # %backedge
; CHECK: # %exit
; CHECK: # %rare
; CHECK: # %rare.1

  br i1 undef, label %rare.1, label %preheader, !prof !{!"branch_weights", i32 0, i32 1000}
rare.1:
  call void @foo()
  br label %preheader

preheader:
  br label %loop

loop:
  %iv = phi i32 [0, %preheader], [%iv.next, %backedge]
  call void @foo()
  br i1 %cnd, label %backedge, label %rare, !prof !{!"branch_weights", i32 1000000, i32 1}
rare:
  call void @foo()
  br label %backedge
backedge:
  call void @foo()
  %iv.next = add i32 %iv, 1
  %cmp = icmp eq i32 %iv.next, 200
  br i1 %cmp, label %loop, label %exit, !prof !{!"branch_weights", i32 1000, i32 1}

exit:
  ret void

}


declare void @foo()
