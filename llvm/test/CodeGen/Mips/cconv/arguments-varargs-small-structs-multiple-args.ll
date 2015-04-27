; RUN: llc --march=mips64 -mcpu=mips64r2 < %s | FileCheck %s

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

declare void @varArgF_SmallStruct(i8* %c, ...) 

define void @smallStruct_1b_x9(%struct.SmallStruct_1b* %ss1, %struct.SmallStruct_1b* %ss2, %struct.SmallStruct_1b* %ss3, %struct.SmallStruct_1b* %ss4, %struct.SmallStruct_1b* %ss5, %struct.SmallStruct_1b* %ss6, %struct.SmallStruct_1b* %ss7, %struct.SmallStruct_1b* %ss8, %struct.SmallStruct_1b* %ss9) #0 {
entry:
  %ss1.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss2.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss3.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss4.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss5.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss6.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss7.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss8.addr = alloca %struct.SmallStruct_1b*, align 8
  %ss9.addr = alloca %struct.SmallStruct_1b*, align 8
  store %struct.SmallStruct_1b* %ss1, %struct.SmallStruct_1b** %ss1.addr, align 8
  store %struct.SmallStruct_1b* %ss2, %struct.SmallStruct_1b** %ss2.addr, align 8
  store %struct.SmallStruct_1b* %ss3, %struct.SmallStruct_1b** %ss3.addr, align 8
  store %struct.SmallStruct_1b* %ss4, %struct.SmallStruct_1b** %ss4.addr, align 8
  store %struct.SmallStruct_1b* %ss5, %struct.SmallStruct_1b** %ss5.addr, align 8
  store %struct.SmallStruct_1b* %ss6, %struct.SmallStruct_1b** %ss6.addr, align 8
  store %struct.SmallStruct_1b* %ss7, %struct.SmallStruct_1b** %ss7.addr, align 8
  store %struct.SmallStruct_1b* %ss8, %struct.SmallStruct_1b** %ss8.addr, align 8
  store %struct.SmallStruct_1b* %ss9, %struct.SmallStruct_1b** %ss9.addr, align 8
  %0 = load %struct.SmallStruct_1b** %ss1.addr, align 8
  %1 = load %struct.SmallStruct_1b** %ss2.addr, align 8
  %2 = load %struct.SmallStruct_1b** %ss3.addr, align 8
  %3 = load %struct.SmallStruct_1b** %ss4.addr, align 8
  %4 = load %struct.SmallStruct_1b** %ss5.addr, align 8
  %5 = load %struct.SmallStruct_1b** %ss6.addr, align 8
  %6 = load %struct.SmallStruct_1b** %ss7.addr, align 8
  %7 = load %struct.SmallStruct_1b** %ss8.addr, align 8
  %8 = load %struct.SmallStruct_1b** %ss9.addr, align 8
  %9 = bitcast %struct.SmallStruct_1b* %0 to { i8 }*
  %10 = getelementptr { i8 }* %9, i32 0, i32 0
  %11 = load i8* %10, align 1
  %12 = bitcast %struct.SmallStruct_1b* %1 to { i8 }*
  %13 = getelementptr { i8 }* %12, i32 0, i32 0
  %14 = load i8* %13, align 1
  %15 = bitcast %struct.SmallStruct_1b* %2 to { i8 }*
  %16 = getelementptr { i8 }* %15, i32 0, i32 0
  %17 = load i8* %16, align 1
  %18 = bitcast %struct.SmallStruct_1b* %3 to { i8 }*
  %19 = getelementptr { i8 }* %18, i32 0, i32 0
  %20 = load i8* %19, align 1
  %21 = bitcast %struct.SmallStruct_1b* %4 to { i8 }*
  %22 = getelementptr { i8 }* %21, i32 0, i32 0
  %23 = load i8* %22, align 1
  %24 = bitcast %struct.SmallStruct_1b* %5 to { i8 }*
  %25 = getelementptr { i8 }* %24, i32 0, i32 0
  %26 = load i8* %25, align 1
  %27 = bitcast %struct.SmallStruct_1b* %6 to { i8 }*
  %28 = getelementptr { i8 }* %27, i32 0, i32 0
  %29 = load i8* %28, align 1
  %30 = bitcast %struct.SmallStruct_1b* %7 to { i8 }*
  %31 = getelementptr { i8 }* %30, i32 0, i32 0
  %32 = load i8* %31, align 1
  %33 = bitcast %struct.SmallStruct_1b* %8 to { i8 }*
  %34 = getelementptr { i8 }* %33, i32 0, i32 0
  %35 = load i8* %34, align 1
  call void (i8*, ...)* @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i8 inreg %11, i8 inreg %14, i8 inreg %17, i8 inreg %20, i8 inreg %23, i8 inreg %26, i8 inreg %29, i8 inreg %32, i8 inreg %35)
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
