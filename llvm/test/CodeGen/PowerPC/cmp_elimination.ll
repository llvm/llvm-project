; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s

; Test cases for compare elimination in PPCMIPeephole pass

define void @func1(i32 signext %a) {
; We should have only one compare instruction
; CHECK-LABEL: @func1
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i32 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp slt i32 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func2(i32 signext %a) {
; CHECK-LABEL: @func2
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i32 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, 100
  br i1 %cmp1, label %if.end3, label %if.then2

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func3(i32 signext %a) {
; CHECK-LABEL: @func3
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp sgt i32 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func4(i32 zeroext %a) {
; CHECK-LABEL: @func4
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i32 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp ult i32 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func5(i32 zeroext %a) {
; CHECK-LABEL: @func5
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ult i32 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, 100
  br i1 %cmp1, label %if.end3, label %if.then2

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func6(i32 zeroext %a) {
; CHECK-LABEL: @func6
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ugt i32 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func7(i64 %a) {
; CHECK-LABEL: @func7
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i64 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp slt i64 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func8(i64 %a) {
; CHECK-LABEL: @func8
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i64 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i64 %a, 100
  br i1 %cmp1, label %if.end3, label %if.then2

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func9(i64 %a) {
; CHECK-LABEL: @func9
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp sgt i64 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i64 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func10(i64 %a) {
; CHECK-LABEL: @func10
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i64 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp ult i64 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func11(i64 %a) {
; CHECK-LABEL: @func11
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ult i64 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i64 %a, 100
  br i1 %cmp1, label %if.end3, label %if.then2

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func12(i64 %a) {
; CHECK-LABEL: @func12
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ugt i64 %a, 100
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i64 %a, 100
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func13(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: @func13
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp slt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func14(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: @func14
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func15(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: @func15
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i32 %b, %a
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func16(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: @func16
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp ult i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func17(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: @func17
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func18(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: @func18
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ult i32 %b, %a
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func19(i64 %a, i64 %b) {
; CHECK-LABEL: @func19
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i64 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp slt i64 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func20(i64 %a, i64 %b) {
; CHECK-LABEL: @func20
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i64 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp sgt i64 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func21(i64 %a, i64 %b) {
; CHECK-LABEL: @func21
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i64 %b, %a
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i64 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func22(i64 %a, i64 %b) {
; CHECK-LABEL: @func22
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i64 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp ult i64 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func23(i64 %a, i64 %b) {
; CHECK-LABEL: @func23
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ult i64 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp ugt i64 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func24(i64 %a, i64 %b) {
; CHECK-LABEL: @func24
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp ult i64 %b, %a
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i64 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}


define void @func25(i64 %a, i64 %b) {
; CHECK-LABEL: @func25
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp slt i64 %b, %a
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  tail call void @dummy1()
  br label %if.end6

if.else:
  %cmp2 = icmp eq i64 %a, %b
  br i1 %cmp2, label %if.then4, label %if.else5

if.then4:
  tail call void @dummy2()
  br label %if.end6

if.else5:
  tail call void @dummy3()
  br label %if.end6

if.end6:
  ret void
}


define void @func26(i32 signext %a) {
; CHECK-LABEL: @func26
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: blr
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else, !prof !2

if.then:
  tail call void @dummy1()
  br label %if.end9

if.else:
  %cmp2 = icmp eq i32 %a, 0
  br i1 %cmp2, label %if.then7, label %if.else8, !prof !2

if.then7:
  tail call void @dummy2()
  br label %if.end9

if.else8:
  tail call void @dummy3()
  br label %if.end9

if.end9:
  ret void
}

@g1 = external local_unnamed_addr global i32, align 4
@g2 = external local_unnamed_addr global i32, align 4

define void @func27(i32 signext %a) {
; CHECK-LABEL: @func27
; CHECK: cmp
; CHECK: beq
; CHECK-NOT: cmp
; CHECK: bgelr
; CHECK: blr
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.end3.sink.split, label %if.else

if.else:
  %cmp1 = icmp slt i32 %a, 0
  br i1 %cmp1, label %if.end3.sink.split, label %if.end

if.end3.sink.split:
  %g2.sink = phi ptr [ @g2, %if.else ], [ @g1, %entry ]
  store i32 0, ptr %g2.sink, align 4
  br label %if.end

if.end:
  ret void
}

; partially redundant case
define void @func28(i32 signext %a) {
; CHECK-LABEL: @func28
; CHECK: cmplwi	 [[REG1:[0-9]+]], [[REG2:[0-9]+]]
; CHECK: .[[LABEL2:[A-Z0-9_]+]]:
; CHECK: cmpwi   [[REG1]], [[REG2]]
; CHECK: ble     0, .[[LABEL1:[A-Z0-9_]+]]
; CHECK-NOT: cmp
; CHECK: bne     0, .[[LABEL2]]
; CHECK: bl dummy1
; CHECK: b .[[LABEL2]]
; CHECK: .[[LABEL1]]:
; CHECK: blr
entry:
  br label %do.body

do.body:
  %a.addr.0 = phi i32 [ %a, %entry ], [ %call, %if.end ]
  %cmp = icmp eq i32 %a.addr.0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @dummy1() #2
  br label %if.end

if.end:
  %call = tail call signext i32 @func(i32 signext %a.addr.0) #2
  %cmp1 = icmp sgt i32 %call, 0
  br i1 %cmp1, label %do.body, label %do.end

do.end:
  ret void
}

define void @func29(i32 signext %a) {
; We cannot merge two compares due to difference in sign extension behaviors.
; equivalent C code example:
;   int a = .. ;
;   if (a == -1) dummy1();
;   if (a == (uint16_t)-1) dummy2();

; CHECK-LABEL: @func29
; CHECK: cmp
; CHECK: cmp
; CHECK: blr
entry:
  %cmp = icmp eq i32 %a, -1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  tail call void @dummy1()
  br label %if.end3

if.else:
  %cmp1 = icmp eq i32 %a, 65535
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:
  tail call void @dummy2()
  br label %if.end3

if.end3:
  ret void
}

;; The result of %cmp may change in a tail call. Don't lift %cmp to the entry block.
; CHECK-LABEL: func_tailrecurse:
; CHECK-NOT:     cmp
; CHECK:       .LBB{{.*}}:
; CHECK:         cmplw
; CHECK:         blt
define fastcc zeroext i32 @func_tailrecurse(i32 zeroext %a, i32 zeroext %b) {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %tailrecurse, %entry
  %a.tr = phi i32 [ %a, %entry ], [ %b.tr, %tailrecurse ]
  %b.tr = phi i32 [ %b, %entry ], [ %a.tr, %tailrecurse ]
  %cmp = icmp ult i32 %a.tr, %b.tr
  %conv = zext i1 %cmp to i32
  %ignore = call signext i32 (i32) @func(i32 %conv)
  br i1 %cmp, label %tailrecurse, label %if.end

if.end:                                           ; preds = %tailrecurse
  %sub = sub nsw i32 %a.tr, %b.tr
  ret i32 %sub
}

declare void @dummy1()
declare void @dummy2()
declare void @dummy3()
declare signext i32 @func(i32 signext)

!1 = !{!"branch_weights", i32 2000, i32 1}
!2 = !{!"branch_weights", i32 1, i32 2000}
