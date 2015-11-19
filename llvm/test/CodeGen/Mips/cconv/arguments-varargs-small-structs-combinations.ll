; RUN: llc --march=mips64 -mcpu=mips64r2 < %s | FileCheck %s

; Generated from the C program:
;
; #include <stdio.h>
; #include <string.h>
; 
; struct SmallStruct_1b1s {
;  char x1;
;  short x2;
; };
; 
; struct SmallStruct_1b1i {
;  char x1;
;  int x2;
; };
; 
; struct SmallStruct_1b1s1b {
;  char x1;
;  short x2;
;  char x3;
; };
; 
; struct SmallStruct_1s1i {
;  short x1;
;  int x2;
; };
; 
; struct SmallStruct_3b1s {
;  char x1;
;  char x2;
;  char x3;
;  short x4;
; };
; 
; void varArgF_SmallStruct(char* c, ...);
; 
; void smallStruct_1b1s(struct SmallStruct_1b1s* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_1b1i(struct SmallStruct_1b1i* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_1b1s1b(struct SmallStruct_1b1s1b* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_1s1i(struct SmallStruct_1s1i* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_3b1s(struct SmallStruct_3b1s* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }

%struct.SmallStruct_1b1s = type { i8, i16 }
%struct.SmallStruct_1b1i = type { i8, i32 }
%struct.SmallStruct_1b1s1b = type { i8, i16, i8 }
%struct.SmallStruct_1s1i = type { i16, i32 }
%struct.SmallStruct_3b1s = type { i8, i8, i8, i16 }

@.str = private unnamed_addr constant [3 x i8] c"01\00", align 1

declare void @varArgF_SmallStruct(i8* %c, ...) 

define void @smallStruct_1b1s(%struct.SmallStruct_1b1s* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_1b1s*, align 8
  store %struct.SmallStruct_1b1s* %ss, %struct.SmallStruct_1b1s** %ss.addr, align 8
  %0 = load %struct.SmallStruct_1b1s*, %struct.SmallStruct_1b1s** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_1b1s* %0 to { i32 }*
  %2 = getelementptr { i32 }, { i32 }* %1, i32 0, i32 0
  %3 = load i32, i32* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_1b1s:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 32
}

define void @smallStruct_1b1i(%struct.SmallStruct_1b1i* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_1b1i*, align 8
  store %struct.SmallStruct_1b1i* %ss, %struct.SmallStruct_1b1i** %ss.addr, align 8
  %0 = load %struct.SmallStruct_1b1i*, %struct.SmallStruct_1b1i** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_1b1i* %0 to { i64 }*
  %2 = getelementptr { i64 }, { i64 }* %1, i32 0, i32 0
  %3 = load i64, i64* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i64 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_1b1i:
 ; CHECK-NOT: dsll
}

define void @smallStruct_1b1s1b(%struct.SmallStruct_1b1s1b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_1b1s1b*, align 8
  %.coerce = alloca { i48 }
  store %struct.SmallStruct_1b1s1b* %ss, %struct.SmallStruct_1b1s1b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_1b1s1b*, %struct.SmallStruct_1b1s1b** %ss.addr, align 8
  %1 = bitcast { i48 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_1b1s1b* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 6, i32 0, i1 false)
  %3 = getelementptr { i48 }, { i48 }* %.coerce, i32 0, i32 0
  %4 = load i48, i48* %3, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i48 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_1b1s1b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

define void @smallStruct_1s1i(%struct.SmallStruct_1s1i* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_1s1i*, align 8
  store %struct.SmallStruct_1s1i* %ss, %struct.SmallStruct_1s1i** %ss.addr, align 8
  %0 = load %struct.SmallStruct_1s1i*, %struct.SmallStruct_1s1i** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_1s1i* %0 to { i64 }*
  %2 = getelementptr { i64 }, { i64 }* %1, i32 0, i32 0
  %3 = load i64, i64* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i64 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_1s1i:
 ; CHECK-NOT: dsll
}

define void @smallStruct_3b1s(%struct.SmallStruct_3b1s* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_3b1s*, align 8
  %.coerce = alloca { i48 }
  store %struct.SmallStruct_3b1s* %ss, %struct.SmallStruct_3b1s** %ss.addr, align 8
  %0 = load %struct.SmallStruct_3b1s*, %struct.SmallStruct_3b1s** %ss.addr, align 8
  %1 = bitcast { i48 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_3b1s* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 6, i32 0, i1 false)
  %3 = getelementptr { i48 }, { i48 }* %.coerce, i32 0, i32 0
  %4 = load i48, i48* %3, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i48 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_3b1s:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}
