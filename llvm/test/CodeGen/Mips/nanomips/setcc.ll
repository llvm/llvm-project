; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i1 @test_seteq(i32 %a, i32 %b) {
; CHECK: xor $a0, $a0, $a1
; CHECK: XOR_NM
; CHECK: sltiu $a0, $a0, 1
; CHECK: SLTIU_NM
  %cmp = icmp eq i32 %a, %b
  ret i1 %cmp
}

define i1 @test_seteq0(i32 %a, i32 %b) {
; CHECK: seqi $a0, $a0, 0
; CHECK: SEQI_NM
  %cmp = icmp eq i32 %a, 0
  ret i1 %cmp
}

define i1 @test_setne(i32 %a, i32 %b) {
; CHECK: xor $a0, $a0, $a1
; CHECK: XOR_NM
; CHECK: sltu $a0, $zero, $a0
; CHECK: SLTU_NM
  %cmp = icmp ne i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setne0(i32 %a, i32 %b) {
; CHECK: sltu $a0, $zero, $a0
; CHECK: SLTU_NM
  %cmp = icmp ne i32 %a, 0
  ret i1 %cmp
}

define i1 @test_setlt(i32 %a, i32 %b) {
; CHECK: slt $a0, $a0, $a1
; CHECK: SLT_NM
  %cmp = icmp slt i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setult(i32 %a, i32 %b) {
; CHECK: sltu $a0, $a0, $a1
; CHECK: SLTU_NM
  %cmp = icmp ult i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setle(i32 %a, i32 %b) {
; CHECK: slt $a0, $a1, $a0
; CHECK: SLT_NM
; CHECK: xori $a0, $a0, 1
; CHECK: XORI_NM
  %cmp = icmp sle i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setule(i32 %a, i32 %b) {
; CHECK: sltu $a0, $a1, $a0
; CHECK: SLTU_NM
; CHECK: xori $a0, $a0, 1
; CHECK: XORI_NM
  %cmp = icmp ule i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setgt(i32 %a, i32 %b) {
; CHECK: slt $a0, $a1, $a0
; CHECK: SLT_NM
  %cmp = icmp sgt i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setugt(i32 %a, i32 %b) {
; CHECK: sltu $a0, $a1, $a0
; CHECK: SLTU_NM
  %cmp = icmp ugt i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setge(i32 %a, i32 %b) {
; CHECK: slt $a0, $a0, $a1
; CHECK: SLT_NM
; CHECK: xori $a0, $a0, 1
; CHECK: XORI_NM
  %cmp = icmp sge i32 %a, %b
  ret i1 %cmp
}

define i1 @test_setuge(i32 %a, i32 %b) {
; CHECK: sltu $a0, $a0, $a1
; CHECK: SLTU_NM
; CHECK: xori $a0, $a0
; CHECK: XORI_NM
  %cmp = icmp uge i32 %a, %b
  ret i1 %cmp
}

; Making sure slti immediate limits are respected.
define i1 @test_slti(i32 %a) {
; CHECK: slti $a0, $a0, 4095
; CHECK: SLTI_NM
  %cmp = icmp slt i32 %a, 4095
  ret i1 %cmp
}

; Making sure slti immediate limits are respected.
define i1 @test_not_slti(i32 %a) {
; CHECK-NOT: slti $a0, $a0, 4096
; CHECK-NOT: SLTI_NM
; CHECK: li $a1, 4096
; CHECK: Li_NM
; CHECK: slt $a0, $a0, $a1
; CHECK: SLT_NM
  %cmp = icmp slt i32 %a, 4096
  ret i1 %cmp
}

; Making sure sltiu immediate limits are respected.
define i1 @test_sltiu(i32 %a) {
; CHECK: sltiu $a0, $a0, 4095
; CHECK: SLTIU_NM
  %cmp = icmp ult i32 %a, 4095
  ret i1 %cmp
}

; Making sure sltiu immediate limits are respected.
define i1 @test_not_sltiu(i32 %a) {
; CHECK-NOT: slti $a0, $a0, 4096
; CHECK-NOT: SLTI_NM
; CHECK: li $a1, 4096
; CHECK: Li_NM
; CHECK: sltu $a0, $a0, $a1
; CHECK: SLTU_NM
  %cmp = icmp ult i32 %a, 4096
  ret i1 %cmp
}

; Making sure seqi immediate limits are respected.
define i1 @test_seqi(i32 %a) {
; CHECK: seqi $a0, $a0, 4095
; CHECK: SEQI_NM
  %cmp = icmp eq i32 %a, 4095
  ret i1 %cmp
}

; Making sure seqi immediate limits are respected.
define i1 @test_not_seqi(i32 %a) {
; CHECK-NOT: seqi $a0, $a0, 4096
; CHECK-NOT: SLTI_NM
; CHECK: li $a1, 4096
; CHECK: Li_NM
; CHECK: xor $a0, $a0, $a1
; CHECK: XOR_NM
; CHECK: sltiu $a0, $a0, 1
; CHECK: SLTIU_NM
  %cmp = icmp eq i32 %a, 4096
  ret i1 %cmp
}
