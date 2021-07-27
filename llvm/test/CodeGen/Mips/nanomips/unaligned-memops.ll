; RUN: llc -march=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @ualw_1(i32* %a) {
; CHECK: ualw $a0, 0($a0)
; CHECK: UALW_NM
  %1 = load i32, i32* %a, align 1
  ret i32 %1
}

define i32 @ualw_2(i32* %a) {
; CHECK: addiu $a0, $a0, 256
; CHECK: ADDiu_NM
; CHECK: ualw $a0, 0($a0)
; CHECK: UALW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 64
  %1 = load i32, i32* %a1, align 1
  ret i32 %1
}

define i32 @ualw_3(i32* %a) {
; CHECK: ualw $a0, 252($a0)
; CHECK: UALW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 63
  %1 = load i32, i32* %a1, align 1
  ret i32 %1
}

define i32 @ualw_4(i32* %a) {
; CHECK: addiu $a0, $a0, -260
; CHECK: ADDiu_NM
; CHECK: ualw $a0, 0($a0)
; CHECK: UALW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 -65
  %1 = load i32, i32* %a1, align 1
  ret i32 %1
}

define i32 @ualw_5(i32* %a) {
; CHECK: ualw $a0, -256($a0)
; CHECK: UALW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 -64
  %1 = load i32, i32* %a1, align 1
  ret i32 %1
}

define void @uasw_1(i32* %a) {
; CHECK: uasw $a1, 0($a0)
; CHECK: UASW_NM
  store i32 1, i32* %a, align 1
  ret void
}

define void @uasw_2(i32* %a) {
; CHECK: addiu $a0, $a0, 256
; CHECK: ADDiu_NM
; CHECK: uasw $a1, 0($a0)
; CHECK: UASW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 64
  store i32 1, i32* %a1, align 1
  ret void
}

define void @uasw_3(i32* %a) {
; CHECK: uasw $a1, 252($a0)
; CHECK: UASW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 63
  store i32 1, i32* %a1, align 1
  ret void
}

define void @uasw_4(i32* %a) {
; CHECK: addiu $a0, $a0, -260
; CHECK: ADDiu_NM
; CHECK: uasw $a1, 0($a0)
; CHECK: UASW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 -65
  store i32 1, i32* %a1, align 1
  ret void
}

define void @uasw_5(i32* %a) {
; CHECK: uasw $a1, -256($a0)
; CHECK: UASW_NM
  %a1 = getelementptr inbounds i32, i32* %a, i64 -64
  store i32 1, i32* %a1, align 1
  ret void
}
