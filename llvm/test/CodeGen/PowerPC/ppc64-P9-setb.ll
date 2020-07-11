; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s -check-prefix=CHECK-PWR8    \
; RUN:   -implicit-check-not "\<setb\>"

; Test different patterns with type i64

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setne)), setlt
define i64 @setb1(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp ne i64 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb1:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb1
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setne)), setgt
define i64 @setb2(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %b, %a
  %t2 = icmp ne i64 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb2:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb2
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setne)), setlt
define i64 @setb3(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp ne i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb3:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb3
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setne)), setgt
define i64 @setb4(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %b, %a
  %t2 = icmp ne i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb4:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb4
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setgt)), setlt
define i64 @setb5(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp sgt i64 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb5:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb5
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setgt)), setgt
define i64 @setb6(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %b, %a
  %t2 = icmp sgt i64 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb6:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb6
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setlt)), setlt
define i64 @setb7(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp slt i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb7:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb7
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setlt)), setgt
define i64 @setb8(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %b, %a
  %t2 = icmp slt i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb8:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb8
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setne)), setgt
define i64 @setb9(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %a, %b
  %t2 = icmp ne i64 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb9:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subfic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb9
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subfic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc rhs, lhs, setne)), setlt
define i64 @setb10(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %b, %a
  %t2 = icmp ne i64 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb10:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subfic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb10
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subfic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc rhs, lhs, setne)), setgt
define i64 @setb11(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %a, %b
  %t2 = icmp ne i64 %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb11:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subfic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb11
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subfic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setne)), setlt
define i64 @setb12(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %b, %a
  %t2 = icmp ne i64 %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb12:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subfic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb12
; CHECK-PWR8-DAG: xor
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subfic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setlt)), setgt
define i64 @setb13(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %a, %b
  %t2 = icmp slt i64 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb13:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb13
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc rhs, lhs, setlt)), setlt
define i64 @setb14(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %b, %a
  %t2 = icmp slt i64 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb14:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb14
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc rhs, lhs, setgt)), setgt
define i64 @setb15(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %a, %b
  %t2 = icmp sgt i64 %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb15:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb15
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setgt)), setlt
define i64 @setb16(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %b, %a
  %t2 = icmp sgt i64 %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb16:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb16
; CHECK-PWR8-DAG: sradi
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8-DAG: xori
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc lhs, rhs, 1, -1, setgt), seteq
define i64 @setb17(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %a, %b
  %t2 = icmp sgt i64 %a, %b
  %t3 = select i1 %t2, i64 1, i64 -1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb17:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb17
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc rhs, lhs, 1, -1, setgt), seteq
define i64 @setb18(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %b, %a
  %t2 = icmp sgt i64 %a, %b
  %t3 = select i1 %t2, i64 1, i64 -1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb18:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb18
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc rhs, lhs, 1, -1, setlt), seteq
define i64 @setb19(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %a, %b
  %t2 = icmp slt i64 %b, %a
  %t3 = select i1 %t2, i64 1, i64 -1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb19:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb19
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc lhs, rhs, 1, -1, setlt), seteq
define i64 @setb20(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %b, %a
  %t2 = icmp slt i64 %b, %a
  %t3 = select i1 %t2, i64 1, i64 -1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb20:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb20
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc lhs, rhs, -1, 1, setlt), seteq
define i64 @setb21(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %a, %b
  %t2 = icmp slt i64 %a, %b
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb21:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb21
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc rhs, lhs, -1, 1, setlt), seteq
define i64 @setb22(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %b, %a
  %t2 = icmp slt i64 %a, %b
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb22:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb22
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc rhs, lhs, -1, 1, setgt), seteq
define i64 @setb23(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %a, %b
  %t2 = icmp sgt i64 %b, %a
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb23:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb23
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc lhs, rhs, -1, 1, setgt), seteq
define i64 @setb24(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %b, %a
  %t2 = icmp sgt i64 %b, %a
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb24:
; CHECK-NOT: li
; CHECK-NOT: cmpld
; CHECK-NOT: isel
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb24
; CHECK-PWR8: cmpd
; CHECK-PWR8: isel
; CHECK-PWR8: cmpld
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}
; end all patterns testing for i64

; Test with swapping the input parameters

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setne)), setlt
define i64 @setb25(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %b, %a
  %t2 = icmp ne i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb25:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK-NOT: cmpd
; CHECK: cmpd {{c?r?(0, )?}}r4, r3
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb25
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setne)), setgt
define i64 @setb26(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %a, %b
  %t2 = icmp ne i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setb26:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r4, r3
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb26
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; Test with different scalar integer type for selected value
; i32/i16/i8 rather than i64 above

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setne)), setlt
define i64 @setb27(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp ne i64 %b, %a
  %t3 = zext i1 %t2 to i32
  %t4 = select i1 %t1, i32 -1, i32 %t3
  %t5 = sext i32 %t4 to i64
  ret i64 %t5
; CHECK-LABEL: setb27:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: extsw
; CHECK: blr
; CHECK-PWR8-LABEL: setb27
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: extsw
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setne)), setgt
define i64 @setb28(i64 %a, i64 %b) {
  %t1 = icmp sgt i64 %b, %a
  %t2 = icmp ne i64 %b, %a
  %t3 = zext i1 %t2 to i16
  %t4 = select i1 %t1, i16 -1, i16 %t3
  %t5 = sext i16 %t4 to i64
  ret i64 %t5
; CHECK-LABEL: setb28:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: addic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: extsw
; CHECK: blr
; CHECK-PWR8-LABEL: setb28
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: addic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: extsw
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setgt)), setlt
define i64 @setb29(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp sgt i64 %a, %b
  %t3 = zext i1 %t2 to i8
  %t4 = select i1 %t1, i8 -1, i8 %t3
  %t5 = zext i8 %t4 to i64
  ret i64 %t5
; CHECK-LABEL: setb29:
; CHECK-NOT: sradi
; CHECK-NOT: rldicl
; CHECK-NOT: li
; CHECK: cmpd {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: adde
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setb29
; CHECK-PWR8-DAG: cmpd
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: adde
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; Testings to cover different comparison opcodes
; Test with integer type i32/i16/i8 for input parameter

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setne)), setlt
define i64 @setbsw1(i32 %a, i32 %b) {
  %t1 = icmp slt i32 %a, %b
  %t2 = icmp ne i32 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsw1:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsw1
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: srwi
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setne)), setgt
define i64 @setbsw2(i32 %a, i32 %b) {
  %t1 = icmp sgt i32 %b, %a
  %t2 = icmp ne i32 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsw2:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsw2
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: srwi
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc rhs, lhs, -1, 1, setgt), seteq
define i64 @setbsw3(i32 %a, i32 %b) {
  %t1 = icmp eq i32 %a, %b
  %t2 = icmp sgt i32 %b, %a
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsw3:
; CHECK-NOT: li
; CHECK: cmpw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK-NOT: cmplw
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsw3
; CHECK-PWR8: cmpw
; CHECK-PWR8: isel
; CHECK-PWR8: cmplw
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setne)), setlt
define i64 @setbsh1(i16 signext %a, i16 signext %b) {
  %t1 = icmp slt i16 %a, %b
  %t2 = icmp ne i16 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsh1:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsh1
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: srwi
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setne)), setgt
define i64 @setbsh2(i16 signext %a, i16 signext %b) {
  %t1 = icmp sgt i16 %b, %a
  %t2 = icmp ne i16 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsh2:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsh2
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8-DAG: srwi
; CHECK-PWR8-DAG: xori
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setgt)), setlt
define i64 @setbsc1(i8 %a, i8 %b) {
  %t1 = icmp slt i8 %a, %b
  %t2 = icmp sgt i8 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsc1:
; CHECK-DAG: extsb [[RA:r[0-9]+]], r3
; CHECK-DAG: extsb [[RB:r[0-9]+]], r4
; CHECK-NOT: li
; CHECK-NOT: sub
; CHECK: cmpw {{c?r?(0, )?}}[[RA]], [[RB]]
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: rldicl
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsc1
; CHECK-PWR8-DAG: extsb
; CHECK-PWR8-DAG: extsb
; CHECK-PWR8-DAG: extsw
; CHECK-PWR8-DAG: extsw
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setgt)), setgt
define i64 @setbsc2(i8 %a, i8 %b) {
  %t1 = icmp sgt i8 %b, %a
  %t2 = icmp sgt i8 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsc2:
; CHECK-DAG: extsb [[RA:r[0-9]+]], r3
; CHECK-DAG: extsb [[RB:r[0-9]+]], r4
; CHECK-NOT: li
; CHECK-NOT: sub
; CHECK: cmpw {{c?r?(0, )?}}[[RA]], [[RB]]
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: rldicl
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsc2
; CHECK-PWR8-DAG: extsb
; CHECK-PWR8-DAG: extsb
; CHECK-PWR8-DAG: extsw
; CHECK-PWR8-DAG: extsw
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setlt)), setlt
define i64 @setbsc3(i4 %a, i4 %b) {
  %t1 = icmp slt i4 %a, %b
  %t2 = icmp slt i4 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbsc3:
; CHECK-DAG: slwi [[RA:r[0-9]+]], r3, 28
; CHECK-DAG: slwi [[RB:r[0-9]+]], r4, 28
; CHECK-NOT: li
; CHECK-DAG: srawi [[RA1:r[0-9]+]], [[RA]], 28
; CHECK-DAG: srawi [[RB1:r[0-9]+]], [[RB]], 28
; CHECK-NOT: sub
; CHECK: cmpw {{c?r?(0, )?}}[[RA1]], [[RB1]]
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: rldicl
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbsc3
; CHECK-PWR8-DAG: slwi
; CHECK-PWR8-DAG: slwi
; CHECK-PWR8-DAG: srawi
; CHECK-PWR8-DAG: srawi
; CHECK-PWR8-DAG: extsw
; CHECK-PWR8-DAG: extsw
; CHECK-PWR8-DAG: cmpw
; CHECK-PWR8-DAG: rldicl
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; Test with unsigned integer type i64/i32/i16/i8 for input parameter

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setult)), setugt
define i64 @setbud1(i64 %a, i64 %b) {
  %t1 = icmp ugt i64 %b, %a
  %t2 = icmp ult i64 %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbud1:
; CHECK-NOT: li
; CHECK: cmpld {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subc
; CHECK-NOT: subfe
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbud1
; CHECK-PWR8-DAG: subc
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8-DAG: cmpld
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setne)), setugt
define i64 @setbud2(i64 %a, i64 %b) {
  %t1 = icmp ugt i64 %a, %b
  %t2 = icmp ne i64 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbud2:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmpld {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: subfic
; CHECK-NOT: subfe
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbud2
; CHECK-PWR8-DAG: cmpld
; CHECK-PWR8-DAG: subfic
; CHECK-PWR8-DAG: subfe
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc lhs, rhs, -1, 1, setugt), seteq
define i64 @setbud3(i64 %a, i64 %b) {
  %t1 = icmp eq i64 %b, %a
  %t2 = icmp ugt i64 %b, %a
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbud3:
; CHECK-NOT: li
; CHECK: cmpld {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: li
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbud3
; CHECK-PWR8-DAG: cmpld
; CHECK-PWR8-DAG: li
; CHECK-PWR8-DAG: li
; CHECK-PWR8: isel
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc rhs, lhs, setne)), setult
define i64 @setbuw1(i32 %a, i32 %b) {
  %t1 = icmp ult i32 %b, %a
  %t2 = icmp ne i32 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbuw1:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmplw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbuw1
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8-DAG: cmplw
; CHECK-PWR8-DAG: srwi
; CHECK-PWR8-DAG: xori
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc rhs, lhs, setne)), setugt
define i64 @setbuw2(i32 %a, i32 %b) {
  %t1 = icmp ugt i32 %a, %b
  %t2 = icmp ne i32 %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbuw2:
; CHECK-NOT: xor
; CHECK-NOT: li
; CHECK: cmplw {{c?r?(0, )?}}r3, r4
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbuw2
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8-DAG: cmplw
; CHECK-PWR8-DAG: srwi
; CHECK-PWR8-DAG: xori
; CHECK-PWR8-DAG: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setne)), setult
define i64 @setbuh(i16 %a, i16 %b) {
  %t1 = icmp ult i16 %b, %a
  %t2 = icmp ne i16 %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbuh:
; CHECK-DAG: clrlwi [[RA:r[0-9]+]], r3, 16
; CHECK-DAG: clrlwi [[RB:r[0-9]+]], r4, 16
; CHECK-NOT: li
; CHECK-NOT: xor
; CHECK: cmplw {{c?r?(0, )?}}[[RA]], [[RB]]
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: cntlzw
; CHECK-NOT: srwi
; CHECK-NOT: xori
; CHECK-NOT: neg
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbuh
; CHECK-PWR8: clrlwi
; CHECK-PWR8: clrlwi
; CHECK-PWR8-DAG: cmplw
; CHECK-PWR8-DAG: cntlzw
; CHECK-PWR8: srwi
; CHECK-PWR8: xori
; CHECK-PWR8: neg
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setult)), setugt
define i64 @setbuc(i8 %a, i8 %b) {
  %t1 = icmp ugt i8 %a, %b
  %t2 = icmp ult i8 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbuc:
; CHECK-DAG: clrlwi [[RA:r[0-9]+]], r3, 24
; CHECK-DAG: clrlwi [[RB:r[0-9]+]], r4, 24
; CHECK-NOT: li
; CHECK-NOT: clrldi
; CHECK: cmplw {{c?r?(0, )?}}[[RA]], [[RB]]
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: sub
; CHECK-NOT: sradi
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbuc
; CHECK-PWR8: clrlwi
; CHECK-PWR8: clrlwi
; CHECK-PWR8-DAG: clrldi
; CHECK-PWR8-DAG: clrldi
; CHECK-PWR8-DAG: cmplw
; CHECK-PWR8: sradi
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; Test with float/double/float128 for input parameter

; select_cc lhs, rhs, -1, (zext (setcc rhs, lhs, setlt)), setlt
define i64 @setbf1(float %a, float %b) {
  %t1 = fcmp nnan olt float %a, %b
  %t2 = fcmp nnan olt float %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbf1:
; CHECK-NOT: li
; CHECK: fcmpu cr0, f1, f2
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK-NOT: li
; CHECK: blr
; CHECK-PWR8-LABEL: setbf1
; CHECK-PWR8: isel
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, -1, (zext (setcc lhs, rhs, setlt)), setgt
define i64 @setbf2(float %a, float %b) {
  %t1 = fcmp nnan ogt float %b, %a
  %t2 = fcmp nnan olt float %b, %a
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbf2:
; CHECK-NOT: li
; CHECK: fcmpu cr0, f1, f2
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK-NOT: li
; CHECK: blr
; CHECK-PWR8-LABEL: setbf2
; CHECK-PWR8: isel
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 0, (select_cc lhs, rhs, -1, 1, setgt), seteq
define i64 @setbdf1(double %a, double %b) {
  %t1 = fcmp nnan oeq double %b, %a
  %t2 = fcmp nnan ogt double %b, %a
  %t3 = select i1 %t2, i64 -1, i64 1
  %t4 = select i1 %t1, i64 0, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbdf1:
; CHECK: xscmpudp cr0, f1, f2
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: li
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbdf1
; CHECK-PWR8: isel
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; select_cc lhs, rhs, 1, (sext (setcc lhs, rhs, setgt)), setlt
define i64 @setbdf2(double %a, double %b) {
  %t1 = fcmp nnan olt double %b, %a
  %t2 = fcmp nnan ogt double %b, %a
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbdf2:
; CHECK-NOT: fcmpu
; CHECK-NOT: li
; CHECK: xscmpudp cr0, f1, f2
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: li
; CHECK-NOT: isel
; CHECK: blr
; CHECK-PWR8-LABEL: setbdf2
; CHECK-PWR8: isel
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

define i64 @setbf128(fp128 %a, fp128 %b) {
  %t1 = fcmp nnan ogt fp128 %a, %b
  %t2 = fcmp nnan olt fp128 %a, %b
  %t3 = sext i1 %t2 to i64
  %t4 = select i1 %t1, i64 1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbf128:
; CHECK-NOT: li
; CHECK: xscmpuqp cr0, v2, v3
; CHECK-NEXT: setb r3, cr0
; CHECK-NOT: isel
; CHECK-NOT: li
; CHECK: blr
; CHECK-PWR8-LABEL: setbf128
; CHECK-PWR8: isel
; CHECK-PWR8: blr
}

; Some cases we can't leverage setb

define i64 @setbn1(i64 %a, i64 %b) {
  %t1 = icmp slt i64 %a, %b
  %t2 = icmp eq i64 %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbn1:
; CHECK-NOT: {{\<setb\>}}
; CHECK: isel
; CHECK: blr
}

define i64 @setbn2(double %a, double %b) {
  %t1 = fcmp olt double %a, %b
  %t2 = fcmp one double %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbn2:
; CHECK-NOT: {{\<setb\>}}
; CHECK: isel
; CHECK: blr
}

define i64 @setbn3(float %a, float %b) {
  %t1 = fcmp ult float %a, %b
  %t2 = fcmp une float %a, %b
  %t3 = zext i1 %t2 to i64
  %t4 = select i1 %t1, i64 -1, i64 %t3
  ret i64 %t4
; CHECK-LABEL: setbn3:
; CHECK-NOT: {{\<setb\>}}
; CHECK: isel
; CHECK: blr
}
