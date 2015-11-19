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
; void smallStruct_1b(struct SmallStruct_1b* ss) {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_2b(struct SmallStruct_2b* ss) {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_3b(struct SmallStruct_3b* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_4b(struct SmallStruct_4b* ss)
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_5b(struct SmallStruct_5b* ss) 
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_6b(struct SmallStruct_6b* ss) 
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_7b(struct SmallStruct_7b* ss) 
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_8b(struct SmallStruct_8b* ss) 
; {
;  varArgF_SmallStruct("", *ss);
; }
; 
; void smallStruct_9b(struct SmallStruct_9b* ss) 
; {
;  varArgF_SmallStruct("", *ss);
; }

%struct.SmallStruct_1b = type { i8 }
%struct.SmallStruct_2b = type { i8, i8 }
%struct.SmallStruct_3b = type { i8, i8, i8 }
%struct.SmallStruct_4b = type { i8, i8, i8, i8 }
%struct.SmallStruct_5b = type { i8, i8, i8, i8, i8 }
%struct.SmallStruct_6b = type { i8, i8, i8, i8, i8, i8 }
%struct.SmallStruct_7b = type { i8, i8, i8, i8, i8, i8, i8 }
%struct.SmallStruct_8b = type { i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.SmallStruct_9b = type { i8, i8, i8, i8, i8, i8, i8, i8, i8 }

@.str = private unnamed_addr constant [3 x i8] c"01\00", align 1

declare void @varArgF_SmallStruct(i8* %c, ...) 

define void @smallStruct_1b(%struct.SmallStruct_1b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_1b*, align 8
  store %struct.SmallStruct_1b* %ss, %struct.SmallStruct_1b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_1b*, %struct.SmallStruct_1b** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_1b* %0 to { i8 }*
  %2 = getelementptr { i8 }, { i8 }* %1, i32 0, i32 0
  %3 = load i8, i8* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_1b: 
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
}

define void @smallStruct_2b(%struct.SmallStruct_2b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_2b*, align 8
  store %struct.SmallStruct_2b* %ss, %struct.SmallStruct_2b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_2b*, %struct.SmallStruct_2b** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_2b* %0 to { i16 }*
  %2 = getelementptr { i16 }, { i16 }* %1, i32 0, i32 0
  %3 = load i16, i16* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i16 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_2b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 48
}

define void @smallStruct_3b(%struct.SmallStruct_3b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_3b*, align 8
  %.coerce = alloca { i24 }
  store %struct.SmallStruct_3b* %ss, %struct.SmallStruct_3b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_3b*, %struct.SmallStruct_3b** %ss.addr, align 8
  %1 = bitcast { i24 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_3b* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 3, i32 0, i1 false)
  %3 = getelementptr { i24 }, { i24 }* %.coerce, i32 0, i32 0
  %4 = load i24, i24* %3, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i24 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_3b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 40
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

define void @smallStruct_4b(%struct.SmallStruct_4b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_4b*, align 8
  store %struct.SmallStruct_4b* %ss, %struct.SmallStruct_4b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_4b*, %struct.SmallStruct_4b** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_4b* %0 to { i32 }*
  %2 = getelementptr { i32 }, { i32 }* %1, i32 0, i32 0
  %3 = load i32, i32* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_4b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 32
}

define void @smallStruct_5b(%struct.SmallStruct_5b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_5b*, align 8
  %.coerce = alloca { i40 }
  store %struct.SmallStruct_5b* %ss, %struct.SmallStruct_5b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_5b*, %struct.SmallStruct_5b** %ss.addr, align 8
  %1 = bitcast { i40 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_5b* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 5, i32 0, i1 false)
  %3 = getelementptr { i40 }, { i40 }* %.coerce, i32 0, i32 0
  %4 = load i40, i40* %3, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i40 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_5b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 24
}

define void @smallStruct_6b(%struct.SmallStruct_6b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_6b*, align 8
  %.coerce = alloca { i48 }
  store %struct.SmallStruct_6b* %ss, %struct.SmallStruct_6b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_6b*, %struct.SmallStruct_6b** %ss.addr, align 8
  %1 = bitcast { i48 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_6b* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 6, i32 0, i1 false)
  %3 = getelementptr { i48 }, { i48 }* %.coerce, i32 0, i32 0
  %4 = load i48, i48* %3, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i48 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_6b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}

define void @smallStruct_7b(%struct.SmallStruct_7b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_7b*, align 8
  %.coerce = alloca { i56 }
  store %struct.SmallStruct_7b* %ss, %struct.SmallStruct_7b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_7b*, %struct.SmallStruct_7b** %ss.addr, align 8
  %1 = bitcast { i56 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_7b* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 7, i32 0, i1 false)
  %3 = getelementptr { i56 }, { i56 }* %.coerce, i32 0, i32 0
  %4 = load i56, i56* %3, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i56 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_7b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 8
}

define void @smallStruct_8b(%struct.SmallStruct_8b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_8b*, align 8
  store %struct.SmallStruct_8b* %ss, %struct.SmallStruct_8b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_8b*, %struct.SmallStruct_8b** %ss.addr, align 8
  %1 = bitcast %struct.SmallStruct_8b* %0 to { i64 }*
  %2 = getelementptr { i64 }, { i64 }* %1, i32 0, i32 0
  %3 = load i64, i64* %2, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i64 inreg %3)
  ret void
 ; CHECK-LABEL: smallStruct_8b:
 ; CHECK-NOT: dsll
}

define void @smallStruct_9b(%struct.SmallStruct_9b* %ss) #0 {
entry:
  %ss.addr = alloca %struct.SmallStruct_9b*, align 8
  %.coerce = alloca { i64, i8 }
  store %struct.SmallStruct_9b* %ss, %struct.SmallStruct_9b** %ss.addr, align 8
  %0 = load %struct.SmallStruct_9b*, %struct.SmallStruct_9b** %ss.addr, align 8
  %1 = bitcast { i64, i8 }* %.coerce to i8*
  %2 = bitcast %struct.SmallStruct_9b* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 9, i32 0, i1 false)
  %3 = getelementptr { i64, i8 }, { i64, i8 }* %.coerce, i32 0, i32 0
  %4 = load i64, i64* %3, align 1
  %5 = getelementptr { i64, i8 }, { i64, i8 }* %.coerce, i32 0, i32 1
  %6 = load i8, i8* %5, align 1
  call void (i8*, ...) @varArgF_SmallStruct(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i64 inreg %4, i8 inreg %6)
  ret void
 ; CHECK-LABEL: smallStruct_9b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
}
