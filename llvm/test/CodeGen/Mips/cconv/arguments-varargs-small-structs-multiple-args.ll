; RUN: llc --mtriple=mips64 -mcpu=mips64r2 < %s | FileCheck %s

; Generated from the C program:
;  
; #include <stdio.h>
; #include <string.h>
; 
; struct SmallStruct_1b {
;  char x1;
; };
; 
; struct SmallStruct_2b {
;  char x1;
;  char x2;
; };
; 
; struct SmallStruct_3b {
;  char x1;
;  char x2;
;  char x3;
; };
; 
; struct SmallStruct_4b {
;  char x1;
;  char x2;
;  char x3;
;  char x4;
; };
; 
; struct SmallStruct_5b {
;  char x1;
;  char x2;
;  char x3;
;  char x4;
;  char x5;
; };
; 
; struct SmallStruct_6b {
;  char x1;
;  char x2;
;  char x3;
;  char x4;
;  char x5;
;  char x6;
; };
; 
; struct SmallStruct_7b {
;  char x1;
;  char x2;
;  char x3;
;  char x4;
;  char x5;
;  char x6;
;  char x7;
; };
; 
; struct SmallStruct_8b {
;  char x1;
;  char x2;
;  char x3;
;  char x4;
;  char x5;
;  char x6;
;  char x7;
;  char x8;
; };
; 
; struct SmallStruct_9b {
;  char x1;
;  char x2;
;  char x3;
;  char x4;
;  char x5;
;  char x6;
;  char x7;
;  char x8;
;  char x9;
; };
; 
; void varArgF_SmallStruct(char* c, ...);
; 
; void smallStruct_1b_x9(struct SmallStruct_1b* ss1,  struct SmallStruct_1b* ss2, struct SmallStruct_1b* ss3, struct SmallStruct_1b* ss4, struct SmallStruct_1b* ss5, struct SmallStruct_1b* ss6, struct SmallStruct_1b* ss7, struct SmallStruct_1b* ss8, struct SmallStruct_1b* ss9)
; {
;  varArgF_SmallStruct("", *ss1, *ss2, *ss3, *ss4, *ss5, *ss6, *ss7, *ss8, *ss9);
; }

%struct.SmallStruct_1b = type { i8 }

@.str = private unnamed_addr constant [3 x i8] c"01\00", align 1

declare void @varArgF_SmallStruct(ptr %c, ...) 

define void @smallStruct_1b_x9(ptr %ss1, ptr %ss2, ptr %ss3, ptr %ss4, ptr %ss5, ptr %ss6, ptr %ss7, ptr %ss8, ptr %ss9) #0 {
entry:
  %ss1.addr = alloca ptr, align 8
  %ss2.addr = alloca ptr, align 8
  %ss3.addr = alloca ptr, align 8
  %ss4.addr = alloca ptr, align 8
  %ss5.addr = alloca ptr, align 8
  %ss6.addr = alloca ptr, align 8
  %ss7.addr = alloca ptr, align 8
  %ss8.addr = alloca ptr, align 8
  %ss9.addr = alloca ptr, align 8
  store ptr %ss1, ptr %ss1.addr, align 8
  store ptr %ss2, ptr %ss2.addr, align 8
  store ptr %ss3, ptr %ss3.addr, align 8
  store ptr %ss4, ptr %ss4.addr, align 8
  store ptr %ss5, ptr %ss5.addr, align 8
  store ptr %ss6, ptr %ss6.addr, align 8
  store ptr %ss7, ptr %ss7.addr, align 8
  store ptr %ss8, ptr %ss8.addr, align 8
  store ptr %ss9, ptr %ss9.addr, align 8
  %0 = load ptr, ptr %ss1.addr, align 8
  %1 = load ptr, ptr %ss2.addr, align 8
  %2 = load ptr, ptr %ss3.addr, align 8
  %3 = load ptr, ptr %ss4.addr, align 8
  %4 = load ptr, ptr %ss5.addr, align 8
  %5 = load ptr, ptr %ss6.addr, align 8
  %6 = load ptr, ptr %ss7.addr, align 8
  %7 = load ptr, ptr %ss8.addr, align 8
  %8 = load ptr, ptr %ss9.addr, align 8
  %9 = load i8, ptr %0, align 1
  %10 = load i8, ptr %1, align 1
  %11 = load i8, ptr %2, align 1
  %12 = load i8, ptr %3, align 1
  %13 = load i8, ptr %4, align 1
  %14 = load i8, ptr %5, align 1
  %15 = load i8, ptr %6, align 1
  %16 = load i8, ptr %7, align 1
  %17 = load i8, ptr %8, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i8 inreg %9, i8 inreg %10, i8 inreg %11, i8 inreg %12, i8 inreg %13, i8 inreg %14, i8 inreg %15, i8 inreg %16, i8 inreg %17)
  ret void
 ; CHECK-LABEL: smallStruct_1b_x9:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
}
