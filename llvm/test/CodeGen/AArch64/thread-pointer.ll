; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

@x = thread_local local_unnamed_addr global i32 0, align 4
@y = thread_local local_unnamed_addr global i32 0, align 4

; Machine LICM should hoist the mrs into the loop preheader.
; CHECK-LABEL: @test1
; CHECK: BB#1:
; CHECK:   mrs x[[BASE:[0-9]+]], TPIDR_EL0
; CHECK:   add x[[REG1:[0-9]+]], x[[BASE]], :tprel_hi12:x
; CHECK:   add x[[REG2:[0-9]+]], x[[REG1]], :tprel_lo12_nc:x
;
; CHECK: .LBB0_2:
; CHECK:   ldr w0, [x[[REG2]]]
; CHECK:   bl bar
; CHECK:   subs w[[REG3:[0-9]+]], w{{[0-9]+}}, #1
; CHECK:   b.ne .LBB0_2

define void @test1(i32 %n) local_unnamed_addr {
entry:
  %cmp3 = icmp sgt i32 %n, 0
  br i1 %cmp3, label %bb1, label %bb2

bb1:
  br label %for.body

for.body:
  %i.04 = phi i32 [ %inc, %for.body ], [ 0, %bb1 ]
  %0 = load i32, i32* @x, align 4
  tail call void @bar(i32 %0) #2
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %bb2, label %for.body

bb2:
  ret void
}

; Machine CSE should combine the the mrs between the load of %x and %y.
; CHECK-LABEL: @test2
; CHECK: mrs x{{[0-9]+}}, TPIDR_EL0
; CHECK-NOT: mrs x{{[0-9]+}}, TPIDR_EL0
; CHECK: ret
define void @test2(i32 %c) local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* @x, align 4
  tail call void @bar(i32 %0) #2
  %cmp = icmp eq i32 %c, 0
  br i1 %cmp, label %if.end, label %if.then

if.then:
  %1 = load i32, i32* @y, align 4
  tail call void @bar(i32 %1) #2
  br label %if.end

if.end:
  ret void
}

declare void @bar(i32) local_unnamed_addr
