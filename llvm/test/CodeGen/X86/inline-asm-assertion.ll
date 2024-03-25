; RUN: not llc -verify-machineinstrs -O0 < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -verify-machineinstrs -O2 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-O2
; CHECK: error: inline assembly requires more registers than available
; CHECK: .size   main, .Lfunc_end0-main
; CHECK-O2: error: inline assembly requires more registers than available

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %r0 = alloca i32, align 4
  %r1 = alloca i32, align 4
  %r2 = alloca i32, align 4
  %r3 = alloca i32, align 4
  %r4 = alloca i32, align 4
  %r5 = alloca i32, align 4
  %r6 = alloca i32, align 4
  %r7 = alloca i32, align 4
  %r8 = alloca i32, align 4
  %r9 = alloca i32, align 4
  %r10 = alloca i32, align 4
  %r11 = alloca i32, align 4
  %r12 = alloca i32, align 4
  %r13 = alloca i32, align 4
  %r14 = alloca i32, align 4
  %0 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm "movl $$0, $0;movl $$1, $1;movl $$2, $2;movl $$3, $3;movl $$4, $4;movl $$5, $5;movl $$6, $6;movl $$7, $7;movl $$8, $8;movl $$9, $9;movl $$10, $10;movl $$11, $11;movl $$12, $12;movl $$13, $13;movl $$14, $14;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,~{dirflag},~{fpsr},~{flags}"() #1
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 0
  %asmresult1 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 1
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 2
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 3
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 4
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 5
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 6
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 7
  %asmresult8 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 8
  %asmresult9 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 9
  %asmresult10 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 10
  %asmresult11 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 11
  %asmresult12 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 12
  %asmresult13 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 13
  %asmresult14 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %0, 14
  store i32 %asmresult, ptr %r0, align 4
  store i32 %asmresult1, ptr %r1, align 4
  store i32 %asmresult2, ptr %r2, align 4
  store i32 %asmresult3, ptr %r3, align 4
  store i32 %asmresult4, ptr %r4, align 4
  store i32 %asmresult5, ptr %r5, align 4
  store i32 %asmresult6, ptr %r6, align 4
  store i32 %asmresult7, ptr %r7, align 4
  store i32 %asmresult8, ptr %r8, align 4
  store i32 %asmresult9, ptr %r9, align 4
  store i32 %asmresult10, ptr %r10, align 4
  store i32 %asmresult11, ptr %r11, align 4
  store i32 %asmresult12, ptr %r12, align 4
  store i32 %asmresult13, ptr %r13, align 4
  store i32 %asmresult14, ptr %r14, align 4
  ret i32 0
}

attributes #0 = { "frame-pointer"="all" }
