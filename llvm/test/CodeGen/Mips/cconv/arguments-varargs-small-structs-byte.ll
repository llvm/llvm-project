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

declare void @varArgF_SmallStruct(ptr %c, ...) 

define void @smallStruct_1b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i8, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i8 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_1b: 
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
}

define void @smallStruct_2b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i16, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i16 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_2b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 48
}

define void @smallStruct_3b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i24 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 3, i1 false)
  %1 = load i24, ptr %.coerce, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i24 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_3b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 40
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1

define void @smallStruct_4b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i32, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i32 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_4b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 32
}

define void @smallStruct_5b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i40 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 5, i1 false)
  %1 = load i40, ptr %.coerce, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i40 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_5b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 24
}

define void @smallStruct_6b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i48 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 6, i1 false)
  %1 = load i48, ptr %.coerce, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i48 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_6b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}

define void @smallStruct_7b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i56 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 7, i1 false)
  %1 = load i56, ptr %.coerce, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i56 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_7b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 8
}

define void @smallStruct_8b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i64, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i64 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_8b:
 ; Check that the structure is not shifted before the pointer to str is loaded.
 ; CHECK-NOT: dsll
 ; CHECK: lui
}

define void @smallStruct_9b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i64, i8 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 9, i1 false)
  %1 = getelementptr { i64, i8 }, ptr %.coerce, i32 0, i32 0
  %2 = load i64, ptr %1, align 1
  %3 = getelementptr { i64, i8 }, ptr %.coerce, i32 0, i32 1
  %4 = load i8, ptr %3, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i64 inreg %2, i8 inreg %4)
  ret void
 ; CHECK-LABEL: smallStruct_9b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 56
}
