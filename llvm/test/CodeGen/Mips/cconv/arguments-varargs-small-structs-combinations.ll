; RUN: llc --mtriple=mips64 -mcpu=mips64r2 < %s | FileCheck %s

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

declare void @varArgF_SmallStruct(ptr %c, ...) 

define void @smallStruct_1b1s(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i32, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i32 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_1b1s:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 32
}

define void @smallStruct_1b1i(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i64, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i64 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_1b1i:
 ; CHECK-NOT: dsll
 ; CHECK: lui
}

define void @smallStruct_1b1s1b(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i48 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 6, i1 false)
  %1 = load i48, ptr %.coerce, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i48 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_1b1s1b:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1

define void @smallStruct_1s1i(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  %1 = load i64, ptr %0, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i64 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_1s1i:
 ; CHECK-NOT: dsll
 ; CHECK: lui
}

define void @smallStruct_3b1s(ptr %ss) #0 {
entry:
  %ss.addr = alloca ptr, align 8
  %.coerce = alloca { i48 }
  store ptr %ss, ptr %ss.addr, align 8
  %0 = load ptr, ptr %ss.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %.coerce, ptr %0, i64 6, i1 false)
  %1 = load i48, ptr %.coerce, align 1
  call void (ptr, ...) @varArgF_SmallStruct(ptr @.str, i48 inreg %1)
  ret void
 ; CHECK-LABEL: smallStruct_3b1s:
 ; CHECK: dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}
