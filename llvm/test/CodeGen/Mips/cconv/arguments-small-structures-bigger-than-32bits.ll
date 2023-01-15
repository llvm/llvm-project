; RUN: llc < %s -march=mips64   -target-abi n64 -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,MIPSEB
; RUN: llc < %s -march=mips64el -target-abi n64 -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,MIPSEL
; RUN: llc < %s -march=mips64   -target-abi n32 -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,MIPSEB
; RUN: llc < %s -march=mips64el -target-abi n32 -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,MIPSEL

; #include <stdio.h>
; 
; struct S1 {
;   char x1;
;   short x2;
;   char x3;
; };
; 
; struct S2 {
;   char x1;
;   char x2;
;   char x3;
;   char x4;
;   char x5;
; };
; 
; void fS1(struct S1 s);
; void fS2(struct S2 s);
; 
; void f1() {
;   struct S1 s1_1;
;   fS1(s1_1);
; }
; 
; void f2() {
;   struct S2 s2_1;
;   fS2(s2_1);
; }
; 
; int main() {
;   f1();
;   f2();
; }

%struct.S1 = type { i8, i16, i8 }
%struct.S2 = type { i8, i8, i8, i8, i8 }

declare void @fS1(i48 inreg) #1
declare void @fS2(i40 inreg) #1

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #2

define void @f1() #0 {
entry:
  %s1_1 = alloca %struct.S1, align 2
  %s1_1.coerce = alloca { i48 }
  call void @llvm.memcpy.p0.p0.i64(ptr %s1_1.coerce, ptr %s1_1, i64 6, i1 false)
  %0 = load i48, ptr %s1_1.coerce, align 1
  call void @fS1(i48 inreg %0)
  ret void
 ; ALL-LABEL: f1:

 ; MIPSEB:       dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
 ; MIPSEL-NOT:   dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 16
}

define void @f2() #0 {
entry:
  %s2_1 = alloca %struct.S2, align 1
  %s2_1.coerce = alloca { i40 }
  call void @llvm.memcpy.p0.p0.i64(ptr %s2_1.coerce, ptr %s2_1, i64 5, i1 false)
  %0 = load i40, ptr %s2_1.coerce, align 1
  call void @fS2(i40 inreg %0)
  ret void
 ; ALL-LABEL: f2:

 ; MIPSEB:       dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 24
 ; MIPSEL-NOT:   dsll $[[R1:[0-9]+]], $[[R2:[0-9]+]], 24
}
