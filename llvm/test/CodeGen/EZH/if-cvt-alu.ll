; RUN: llc -mtriple=ezh-none-elf -mattr=-bitslice-interrupts -O3 < %s | FileCheck %s

define i32 @test_select_add(i32 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: test_select_add:
; CHECK: e_sub_imms r0, r0, 0
; CHECK: e_add_ze r1, r1, r2
; CHECK: e_mov pc, ra
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.true, label %if.end

if.true:
  %add = add i32 %a, %b
  br label %if.end

if.end:
  %res = phi i32 [ %add, %if.true ], [ %a, %entry ]
  ret i32 %res
}

define i32 @test_select_sub(i32 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: test_select_sub:
; CHECK: e_sub_imms r0, r0, 0
; CHECK: e_sub_ze r1, r1, r2
; CHECK: e_mov pc, ra
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.true, label %if.end

if.true:
  %sub = sub i32 %a, %b
  br label %if.end

if.end:
  %res = phi i32 [ %sub, %if.true ], [ %a, %entry ]
  ret i32 %res
}

define i32 @test_select_and(i32 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: test_select_and:
; CHECK: e_sub_imms r0, r0, 0
; CHECK: e_and_ze r1, r1, r2
; CHECK: e_mov pc, ra
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.true, label %if.end

if.true:
  %and = and i32 %a, %b
  br label %if.end

if.end:
  %res = phi i32 [ %and, %if.true ], [ %a, %entry ]
  ret i32 %res
}

define i32 @test_select_or(i32 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: test_select_or:
; CHECK: e_sub_imms r0, r0, 0
; CHECK: e_or_ze r1, r1, r2
; CHECK: e_mov pc, ra
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.true, label %if.end

if.true:
  %or = or i32 %a, %b
  br label %if.end

if.end:
  %res = phi i32 [ %or, %if.true ], [ %a, %entry ]
  ret i32 %res
}

define i32 @test_select_xor(i32 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: test_select_xor:
; CHECK: e_sub_imms r0, r0, 0
; CHECK: e_xor_ze r1, r1, r2
; CHECK: e_mov pc, ra
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.true, label %if.end

if.true:
  %xor = xor i32 %a, %b
  br label %if.end

if.end:
  %res = phi i32 [ %xor, %if.true ], [ %a, %entry ]
  ret i32 %res
}

define i32 @test_select_shl(i32 %cond, i32 %a) {
; CHECK-LABEL: test_select_shl:
; CHECK: e_sub_imms r0, r0, 0
; CHECK: e_lsl_ze r1, r1, 2
; CHECK: e_mov pc, ra
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.true, label %if.end

if.true:
  %shl = shl i32 %a, 2
  br label %if.end

if.end:
  %res = phi i32 [ %shl, %if.true ], [ %a, %entry ]
  ret i32 %res
}
