; RUN: opt -S -jump-threading < %s | FileCheck %s

declare void @foo()
declare void @bar()
declare void @baz()
declare void @quux()


; Jump threading of branch with select as condition.
; Mostly theoretical since instruction combining simplifies all selects of
; booleans where at least one operand is true/false/undef.

; CHECK: @test_br
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %cond, label %L1,
define void @test_br(i1 %cond, i1 %value) nounwind {
entry:
  br i1 %cond, label %L0, label %L3
L0:
  %expr = select i1 %cond, i1 true, i1 %value
  br i1 %expr, label %L1, label %L2

L1:
  call void @foo()
  ret void
L2:
  call void @bar()
  ret void
L3:
  call void @baz()
  br label %L0
}


; Jump threading of switch with select as condition.

; CHECK: @test_switch
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %cond, label %L1,
define void @test_switch(i1 %cond, i8 %value) nounwind {
entry:
  br i1 %cond, label %L0, label %L4
L0:
  %expr = select i1 %cond, i8 1, i8 %value
  switch i8 %expr, label %L3 [i8 1, label %L1 i8 2, label %L2]

L1:
  call void @foo()
  ret void
L2:
  call void @bar()
  ret void
L3:
  call void @baz()
  ret void
L4:
  call void @quux()
  br label %L0
}

; Make sure the blocks in the indirectbr test aren't trivially removable as
; successors by taking their addresses.
@anchor = constant [3 x i8*] [
  i8* blockaddress(@test_indirectbr, %L1),
  i8* blockaddress(@test_indirectbr, %L2),
  i8* blockaddress(@test_indirectbr, %L3)
]


; Jump threading of indirectbr with select as address.

; CHECK: @test_indirectbr
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %cond, label %L1, label %L3
define void @test_indirectbr(i1 %cond, i8* %address) nounwind {
entry:
  br i1 %cond, label %L0, label %L3
L0:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@test_indirectbr, %L1), i8* %address
  indirectbr i8* %indirect.goto.dest, [label %L1, label %L2, label %L3]

L1:
  call void @foo()
  ret void
L2:
  call void @bar()
  ret void
L3:
  call void @baz()
  ret void
}


; A more complicated case: the condition is a select based on a comparison.

; CHECK: @test_switch_cmp
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %cond, label %L0, label %[[THREADED:[A-Za-z.0-9]+]]
; CHECK: [[THREADED]]:
; CHECK-NEXT: call void @quux
; CHECK-NEXT: br label %L1
define void @test_switch_cmp(i1 %cond, i32 %val, i8 %value) nounwind {
entry:
  br i1 %cond, label %L0, label %L4
L0:
  %val.phi = phi i32 [%val, %entry], [-1, %L4]
  %cmp = icmp slt i32 %val.phi, 0
  %expr = select i1 %cmp, i8 1, i8 %value
  switch i8 %expr, label %L3 [i8 1, label %L1 i8 2, label %L2]

L1:
  call void @foo()
  ret void
L2:
  call void @bar()
  ret void
L3:
  call void @baz()
  ret void
L4:
  call void @quux()
  br label %L0
}

; Make sure the edge value of %0 from entry to L2 includes 0 and L3 is
; reachable.
; CHECK: test_switch_default
; CHECK: entry:
; CHECK: load
; CHECK: switch
; CHECK: [[THREADED:[A-Za-z.0-9]+]]:
; CHECK: store
; CHECK: br
; CHECK: L2:
; CHECK: icmp
define void @test_switch_default(i32* nocapture %status) nounwind {
entry:
  %0 = load i32* %status, align 4
  switch i32 %0, label %L2 [
    i32 5061, label %L1
    i32 0, label %L2
  ]

L1:
  store i32 10025, i32* %status, align 4
  br label %L2

L2:
  %1 = load i32* %status, align 4
  %cmp57.i = icmp eq i32 %1, 0
  br i1 %cmp57.i, label %L3, label %L4

L3:
  store i32 10000, i32* %status, align 4
  br label %L4

L4:
  ret void
}
