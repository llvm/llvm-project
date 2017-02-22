; RUN: llc -mtriple=i686-unknown -verify-machineinstrs < %s | FileCheck %s
; RUN: opt < %s -codegenprepare -S -mtriple=x86_64-unknown-unknown | FileCheck --check-prefix=CHECK-CGP %s

@A = global i32 zeroinitializer
@B = global i32 zeroinitializer
@C = global i32 zeroinitializer

; Test that 'and' is sunk into bb0.
define i32 @and_sink1(i32 %a, i1 %c) {
; CHECK-LABEL: and_sink1:
; CHECK: testb $1,
; CHECK: je
; CHECK-NOT: andl $4,
; CHECK: movl $0, A
; CHECK: testb $4,
; CHECK: jne

; CHECK-CGP-LABEL: @and_sink1(
; CHECK-CGP-NOT: and i32
  %and = and i32 %a, 4
  br i1 %c, label %bb0, label %bb2
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP: and i32
; CHECK-CGP-NEXT: icmp eq i32
; CHECK-CGP-NEXT: store
; CHECK-CGP-NEXT: br
  %cmp = icmp eq i32 %and, 0
  store i32 0, i32* @A
  br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 0
}

; Test that both 'and' and cmp get sunk to bb1.
define i32 @and_sink2(i32 %a, i1 %c, i1 %c2) {
; CHECK-LABEL: and_sink2:
; CHECK: movl $0, A
; CHECK: testb $1,
; CHECK: je
; CHECK-NOT: andl $4,
; CHECK: movl $0, B
; CHECK: testb $1,
; CHECK: je
; CHECK: movl $0, C
; CHECK: testb $4,
; CHECK: jne

; CHECK-CGP-LABEL: @and_sink2(
; CHECK-CGP-NOT: and i32
  %and = and i32 %a, 4
  store i32 0, i32* @A
  br i1 %c, label %bb0, label %bb3
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP-NOT: and i32
; CHECK-CGP-NOT: icmp
  %cmp = icmp eq i32 %and, 0
  store i32 0, i32* @B
  br i1 %c2, label %bb1, label %bb3
bb1:
; CHECK-CGP-LABEL: bb1:
; CHECK-CGP: and i32
; CHECK-CGP-NEXT: icmp eq i32
; CHECK-CGP-NEXT: store
; CHECK-CGP-NEXT: br
  store i32 0, i32* @C
  br i1 %cmp, label %bb2, label %bb0
bb2:
  ret i32 1
bb3:
  ret i32 0
}

; Test that CodeGenPrepare doesn't get stuck in a loop sinking and hoisting a masked load.
define i32 @and_sink3(i1 %c, i32* %p) {
; CHECK-LABEL: and_sink3:
; CHECK: testb $1,
; CHECK: je
; CHECK: movzbl
; CHECK: movl $0, A
; CHECK: testl %
; CHECK: je

; CHECK-CGP-LABEL: @and_sink3(
; CHECK-CGP: load i32
; CHECK-CGP-NEXT: and i32
  %load = load i32, i32* %p
  %and = and i32 %load, 255
  br i1 %c, label %bb0, label %bb2
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP-NOT: and i32
; CHECK-CGP: icmp eq i32
  %cmp = icmp eq i32 %and, 0
  store i32 0, i32* @A
  br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 0
}

; Test that CodeGenPrepare sinks/duplicates non-immediate 'and'.
define i32 @and_sink4(i32 %a, i32 %b, i1 %c) {
; CHECK-LABEL: and_sink4:
; CHECK: testb $1,
; CHECK: je
; CHECK-NOT: andl
; CHECK: movl $0, A
; CHECK: testl [[REG1:%[a-z0-9]+]], [[REG2:%[a-z0-9]+]]
; CHECK: jne
; CHECK: movl {{%[a-z0-9]+}}, B
; CHECK: testl [[REG1]], [[REG2]]
; CHECK: je

; CHECK-CGP-LABEL: @and_sink4(
; CHECK-CGP-NOT: and i32
; CHECK-CGP-NOT: icmp
  %and = and i32 %a, %b
  %cmp = icmp eq i32 %and, 0
  br i1 %c, label %bb0, label %bb3
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP: and i32
; CHECK-CGP-NEXT: icmp eq i32
  store i32 0, i32* @A
  br i1 %cmp, label %bb1, label %bb3
bb1:
; CHECK-CGP-LABEL: bb1:
; CHECK-CGP: and i32
; CHECK-CGP-NEXT: icmp eq i32
  %add = add i32 %a, %b
  store i32 %add, i32* @B
  br i1 %cmp, label %bb2, label %bb3
bb2:
  ret i32 1
bb3:
  ret i32 0
}


; Test that CodeGenPrepare doesn't sink/duplicate non-immediate 'and'
; when it would increase register pressure.
define i32 @and_sink5(i32 %a, i32 %b, i32 %a2, i32 %b2, i1 %c) {
; CHECK-LABEL: and_sink5:
; CHECK: testb $1,
; CHECK: je
; CHECK: andl {{[0-9]+\(%[a-z0-9]+\)}}, [[REG:%[a-z0-9]+]]
; CHECK: movl $0, A
; CHECK: jne
; CHECK: movl {{%[a-z0-9]+}}, B
; CHECK: testl [[REG]], [[REG]]
; CHECK: je

; CHECK-CGP-LABEL: @and_sink5(
; CHECK-CGP: and i32
; CHECK-CGP-NOT: icmp
  %and = and i32 %a, %b
  %cmp = icmp eq i32 %and, 0
  br i1 %c, label %bb0, label %bb3
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP-NOT: and i32
; CHECK-CGP: icmp eq i32
  store i32 0, i32* @A
  br i1 %cmp, label %bb1, label %bb3
bb1:
; CHECK-CGP-LABEL: bb1:
; CHECK-CGP-NOT: and i32
; CHECK-CGP: icmp eq i32
  %add = add i32 %a2, %b2
  store i32 %add, i32* @B
  br i1 %cmp, label %bb2, label %bb3
bb2:
  ret i32 1
bb3:
  ret i32 0
}
