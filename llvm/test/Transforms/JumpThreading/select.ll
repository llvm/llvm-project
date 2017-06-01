; RUN: opt -S -jump-threading < %s | FileCheck %s

declare void @foo()
declare void @bar()
declare void @baz()
declare void @quux()


; Jump threading of branch with select as condition.
; Mostly theoretical since instruction combining simplifies all selects of
; booleans where at least one operand is true/false/undef.

; CHECK-LABEL: @test_br(
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

; CHECK-LABEL: @test_switch(
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

; CHECK-LABEL: @test_indirectbr(
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


; Jump threading of indirectbr with select as address.  Test increased
; duplication threshold for cases where indirectbr is being threaded
; through.

; CHECK-LABEL: @test_indirectbr_thresh(
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %cond, label %L1, label %L3
; CHECK-NOT: indirectbr
define void @test_indirectbr_thresh(i1 %cond, i8* %address) nounwind {
entry:
  br i1 %cond, label %L0, label %L3
L0:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@test_indirectbr_thresh, %L1), i8* %address
  call void @quux()
  call void @quux()
  call void @quux()
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

; CHECK-LABEL: @test_switch_cmp(
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
; CHECK: icmp
; CHECK: [[THREADED:[A-Za-z.0-9]+]]:
; CHECK: store
; CHECK: br
; CHECK: L2:
; CHECK: icmp
define void @test_switch_default(i32* nocapture %status) nounwind {
entry:
  %0 = load i32, i32* %status, align 4
  switch i32 %0, label %L2 [
    i32 5061, label %L1
    i32 0, label %L2
  ]

L1:
  store i32 10025, i32* %status, align 4
  br label %L2

L2:
  %1 = load i32, i32* %status, align 4
  %cmp57.i = icmp eq i32 %1, 0
  br i1 %cmp57.i, label %L3, label %L4

L3:
  store i32 10000, i32* %status, align 4
  br label %L4

L4:
  ret void
}

define void @unfold1(double %x, double %y) nounwind {
entry:
  %sub = fsub double %x, %y
  %cmp = fcmp ogt double %sub, 1.000000e+01
  br i1 %cmp, label %cond.end4, label %cond.false

cond.false:                                       ; preds = %entry
  %add = fadd double %x, %y
  %cmp1 = fcmp ogt double %add, 1.000000e+01
  %add. = select i1 %cmp1, double %add, double 0.000000e+00
  br label %cond.end4

cond.end4:                                        ; preds = %entry, %cond.false
  %cond5 = phi double [ %add., %cond.false ], [ %sub, %entry ]
  %cmp6 = fcmp oeq double %cond5, 0.000000e+00
  br i1 %cmp6, label %if.then, label %if.end

if.then:                                          ; preds = %cond.end4
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %cond.end4
  ret void

; CHECK-LABEL: @unfold1
; CHECK: br i1 %cmp, label %cond.end4, label %cond.false
; CHECK: br i1 %cmp1, label %cond.end4, label %if.then
; CHECK: br i1 %cmp6, label %if.then, label %if.end
; CHECK: br label %if.end
}


define void @unfold2(i32 %x, i32 %y) nounwind {
entry:
  %sub = sub nsw i32 %x, %y
  %cmp = icmp sgt i32 %sub, 10
  br i1 %cmp, label %cond.end4, label %cond.false

cond.false:                                       ; preds = %entry
  %add = add nsw i32 %x, %y
  %cmp1 = icmp sgt i32 %add, 10
  %add. = select i1 %cmp1, i32 0, i32 %add
  br label %cond.end4

cond.end4:                                        ; preds = %entry, %cond.false
  %cond5 = phi i32 [ %add., %cond.false ], [ %sub, %entry ]
  %cmp6 = icmp eq i32 %cond5, 0
  br i1 %cmp6, label %if.then, label %if.end

if.then:                                          ; preds = %cond.end4
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %cond.end4
  ret void

; CHECK-LABEL: @unfold2
; CHECK: br i1 %cmp, label %if.end, label %cond.false
; CHECK: br i1 %cmp1, label %if.then, label %cond.end4
; CHECK: br i1 %cmp6, label %if.then, label %if.end
; CHECK: br label %if.end
}


define i32 @unfold3(i32 %u, i32 %v, i32 %w, i32 %x, i32 %y, i32 %z, i32 %j) nounwind {
entry:
  %add3 = add nsw i32 %j, 2
  %cmp.i = icmp slt i32 %u, %v
  br i1 %cmp.i, label %.exit, label %cond.false.i

cond.false.i:                                     ; preds = %entry
  %cmp4.i = icmp sgt i32 %u, %v
  br i1 %cmp4.i, label %.exit, label %cond.false.6.i

cond.false.6.i:                                   ; preds = %cond.false.i
  %cmp8.i = icmp slt i32 %w, %x
  br i1 %cmp8.i, label %.exit, label %cond.false.10.i

cond.false.10.i:                                  ; preds = %cond.false.6.i
  %cmp13.i = icmp sgt i32 %w, %x
  br i1 %cmp13.i, label %.exit, label %cond.false.15.i

cond.false.15.i:                                  ; preds = %cond.false.10.i
  %phitmp = icmp sge i32 %y, %z
  br label %.exit

.exit:                                  ; preds = %entry, %cond.false.i, %cond.false.6.i, %cond.false.10.i, %cond.false.15.i
  %cond23.i = phi i1 [ false, %entry ], [ true, %cond.false.i ], [ false, %cond.false.6.i ], [ %phitmp, %cond.false.15.i ], [ true, %cond.false.10.i ]
  %j.add3 = select i1 %cond23.i, i32 %j, i32 %add3
  ret i32 %j.add3

; CHECK-LABEL: @unfold3
; CHECK: br i1 %cmp.i, label %.exit.thread2, label %cond.false.i 
; CHECK: br i1 %cmp4.i, label %.exit.thread, label %cond.false.6.i
; CHECK: br i1 %cmp8.i, label %.exit.thread2, label %cond.false.10.i
; CHECK: br i1 %cmp13.i, label %.exit.thread, label %.exit
; CHECK: br i1 %phitmp, label %.exit.thread, label %.exit.thread2
; CHECK: br label %.exit.thread2
}
