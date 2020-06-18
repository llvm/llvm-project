; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

@X = common local_unnamed_addr global i32 0, align 4

define i32 @test1() {
; CHECK-LABEL: test1:
; CHECK:         .word b
; CHECK-NEXT:    .word .Ltmp0
; CHECK-LABEL: .LBB0_1: // %cleanup
; CHECK-LABEL: .Ltmp0:
; CHECK-LABEL: .LBB0_2: // %indirect
entry:
  callbr void asm sideeffect "1:\0A\09.word b, ${0:l}\0A\09", "X"(i8* blockaddress(@test1, %indirect))
          to label %cleanup [label %indirect]

indirect:
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ 1, %indirect ], [ 0, %entry ]
  ret i32 %retval.0
}

define void @test2() {
; CHECK-LABEL: test2:
entry:
  %0 = load i32, i32* @X, align 4
  %and = and i32 %0, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end10, label %if.then

if.then:
; CHECK:       .word b
; CHECK-NEXT:  .word .Ltmp2
; CHECK-LABEL: .Ltmp2:
; CHECK-NEXT:  .LBB1_3: // %if.end6
  callbr void asm sideeffect "1:\0A\09.word b, ${0:l}\0A\09", "X"(i8* blockaddress(@test2, %if.end6))
          to label %if.then4 [label %if.end6]

if.then4:
  %call5 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  br label %if.end6

if.end6:
  %.pre = load i32, i32* @X, align 4
  %.pre13 = and i32 %.pre, 1
  %phitmp = icmp eq i32 %.pre13, 0
  br i1 %phitmp, label %if.end10, label %if.then9

if.then9:
; CHECK-LABEL: .Ltmp4:
; CHECK-NEXT:  .LBB1_5: // %l_yes
  callbr void asm sideeffect "", "X"(i8* blockaddress(@test2, %l_yes))
          to label %if.end10 [label %l_yes]

if.end10:
  br label %l_yes

l_yes:
  ret void
}

declare i32 @g(...)
