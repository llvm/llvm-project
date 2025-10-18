; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/probable-or.prof > %t/probable-or.ll
; RUN: cat %t/main.ll %t/probable-and.prof > %t/probable-and.ll
; RUN: opt -passes='loop-mssa(simple-loop-unswitch)' -S %t/probable-or.ll -o - | FileCheck %t/probable-or.prof
; RUN: opt -passes='loop-mssa(simple-loop-unswitch)' -S %t/probable-and.ll -o - | FileCheck %t/probable-and.prof
;
; RUN: opt -passes='module(print<block-freq>),function(loop-mssa(simple-loop-unswitch)),module(print<block-freq>)' \
; RUN:   %t/probable-or.ll -disable-output -simple-loop-unswitch-estimate-profile=0 2>&1 | FileCheck %t/probable-or.prof --check-prefixes=PROFILE-COM,PROFILE-REF

; RUN: opt -passes='module(print<block-freq>),function(loop-mssa(simple-loop-unswitch)),module(print<block-freq>)' \
; RUN:   %t/probable-or.ll -disable-output -simple-loop-unswitch-estimate-profile=1 2>&1 | FileCheck %t/probable-or.prof --check-prefixes=PROFILE-COM,PROFILE-CHK

; RUN: opt -passes='module(print<block-freq>),function(loop-mssa(simple-loop-unswitch)),module(print<block-freq>)' \
; RUN:   %t/probable-and.ll -disable-output -simple-loop-unswitch-estimate-profile=0 2>&1 | FileCheck %t/probable-and.prof --check-prefixes=PROFILE-COM,PROFILE-REF

; RUN: opt -passes='module(print<block-freq>),function(loop-mssa(simple-loop-unswitch)),module(print<block-freq>)' \
; RUN:   %t/probable-and.ll -disable-output -simple-loop-unswitch-estimate-profile=1 2>&1 | FileCheck %t/probable-and.prof --check-prefixes=PROFILE-COM,PROFILE-CHK

;--- main.ll
declare void @some_func() noreturn

define i32 @or(i1 %cond1, i32 %var1) !prof !0 {
entry:
  br label %loop_begin

loop_begin:
  %var3 = phi i32 [%var1, %entry], [%var2, %do_something]
  %cond2 = icmp eq i32 %var3, 10
  %cond.or = or i1 %cond1, %cond2
  br i1 %cond.or, label %loop_exit, label %do_something, !prof !1

do_something:
  %var2 = add i32 %var3, 1
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

define i32 @and(i1 %cond1, i32 %var1) !prof !0 {
entry:
  br label %loop_begin

loop_begin:
  %var3 = phi i32 [%var1, %entry], [%var2, %do_something]
  %cond2 = icmp eq i32 %var3, 10
  %cond.and = and i1 %cond1, %cond2
  br i1 %cond.and, label %do_something, label %loop_exit, !prof !1

do_something:
  %var2 = add i32 %var3, 1
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

;--- probable-or.prof
!0 = !{!"function_entry_count", i32 10}
!1 = !{!"branch_weights", i32 1, i32 1000}
; CHECK-LABEL: @or
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond1.fr = freeze i1 %cond1
; CHECK-NEXT:   br i1 %cond1.fr, label %loop_exit.split, label %entry.split, !prof !1
; CHECK-LABEL: @and
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond1.fr = freeze i1 %cond1
; CHECK-NEXT:   br i1 %cond1.fr, label %entry.split, label %loop_exit.split, !prof !2
; CHECK: !1 = !{!"branch_weights", i32 1, i32 1000}
; CHECK: !2 = !{!"unknown", !"simple-loop-unswitch"}

; PROFILE-COM: Printing analysis results of BFI for function 'or':
; PROFILE-COM: block-frequency-info: or
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-COM: - loop_begin: {{.*}} count = 10010
 ; PROFILE-COM: - do_something: {{.*}} count = 10000
 ; PROFILE-COM: - loop_exit: {{.*}} count = 10

; PROFILE-COM: Printing analysis results of BFI for function 'and':
; PROFILE-COM: block-frequency-info: and
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-COM: - loop_begin: {{.*}} count = 10
 ; PROFILE-COM: - do_something: {{.*}} count = 0
 ; PROFILE-COM: - loop_exit: {{.*}} count = 10

; PROFILE-COM: Printing analysis results of BFI for function 'or':
; PROFILE-COM: block-frequency-info: or
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-REF: - entry.split: {{.*}} count = 5
 ; PROFILE-CHK: - entry.split: {{.*}} count = 10
 ; PROFILE-REF: - loop_begin: {{.*}} count = 5005
 ; PROFILE-CHK: - loop_begin: {{.*}} count = 10000
 ; PROFILE-REF: - do_something: {{.*}} count = 5000
 ; PROFILE-CHK: - do_something: {{.*}} count = 9990
 ; PROFILE-REF: - loop_exit: {{.*}} count = 5
 ; PROFILE-CHK: - loop_exit: {{.*}} count = 10
 ; PROFILE-COM: - loop_exit.split: {{.*}} count = 10

; PROFILE-COM: Printing analysis results of BFI for function 'and':
; PROFILE-COM: block-frequency-info: and
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-COM: - entry.split: {{.*}} count = 5
 ; PROFILE-COM: - loop_begin: {{.*}} count = 5
 ; PROFILE-COM: - do_something: {{.*}} count = 0
 ; PROFILE-COM: - loop_exit: {{.*}} count = 5
 ; PROFILE-COM: - loop_exit.split: {{.*}} count = 10

;--- probable-and.prof
!0 = !{!"function_entry_count", i32 10}
!1 = !{!"branch_weights", i32 1000, i32 1}
; CHECK-LABEL: @or
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond1.fr = freeze i1 %cond1
; CHECK-NEXT:   br i1 %cond1.fr, label %loop_exit.split, label %entry.split, !prof !1
; CHECK-LABEL: @and
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond1.fr = freeze i1 %cond1
; CHECK-NEXT:   br i1 %cond1.fr, label %entry.split, label %loop_exit.split, !prof !2
; CHECK: !1 = !{!"unknown", !"simple-loop-unswitch"}
; CHECK: !2 = !{!"branch_weights", i32 1000, i32 1}
; PROFILE-COM: Printing analysis results of BFI for function 'or':
; PROFILE-COM: block-frequency-info: or
 ; PROFILE-COM: - entry: {{.*}}, count = 10
 ; PROFILE-COM: - loop_begin: {{.*}}, count = 10
 ; PROFILE-COM: - do_something: {{.*}}, count = 0
 ; PROFILE-COM: - loop_exit: {{.*}}, count = 10

; PROFILE-COM: Printing analysis results of BFI for function 'and':
; PROFILE-COM: block-frequency-info: and
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-COM: - loop_begin: {{.*}} count = 10010
 ; PROFILE-COM: - do_something: {{.*}} count = 10000
 ; PROFILE-COM: - loop_exit: {{.*}} count = 10

; PROFILE-COM: Printing analysis results of BFI for function 'or':
; PROFILE-COM: block-frequency-info: or
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-COM: - entry.split: {{.*}} count = 5
 ; PROFILE-COM: - loop_begin: {{.*}} count = 5
 ; PROFILE-COM: - do_something: {{.*}} count = 0
 ; PROFILE-COM: - loop_exit: {{.*}} count = 5
 ; PROFILE-COM: - loop_exit.split: {{.*}} count = 10

; PROFILE-COM: Printing analysis results of BFI for function 'and':
; PROFILE-COM: block-frequency-info: and
 ; PROFILE-COM: - entry: {{.*}} count = 10
 ; PROFILE-REF: - entry.split: {{.*}} count = 5
 ; PROFILE-CHK: - entry.split: {{.*}} count = 10
 ; PROFILE-REF: - loop_begin: {{.*}} count = 5005
 ; PROFILE-CHK: - loop_begin: {{.*}} count = 10000
 ; PROFILE-REF: - do_something: {{.*}} count = 5000
 ; PROFILE-CHK: - do_something: {{.*}} count = 9990
 ; PROFILE-REF: - loop_exit: {{.*}} count = 5
 ; PROFILE-CHK: - loop_exit: {{.*}} count = 10
 ; PROFILE-COM: - loop_exit.split: {{.*}} count = 10
