; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-typeid-resolutions-import.ll -o %t-import.bc

; RUN: llvm-as -o %t1.bc %S/Inputs/cache-typeid-resolutions1.ll
; RUN: llvm-as -o %t2.bc %S/Inputs/cache-typeid-resolutions2.ll
; RUN: llvm-as -o %t3.bc %S/Inputs/cache-typeid-resolutions3.ll

; Two resolutions for typeid1: Unsat, Single
; where both t and t-import are sensitive to typeid1's resolution
; so 4 distinct objects in total.
; RUN: rm -rf %t.cache
; RUN: llvm-lto2 run -o %t.o %t.bc %t-import.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t.bc,f1_actual,plx -r=%t.bc,f2,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx -r=%t-import.bc,importf2,plx -r=%t-import.bc,f2,lx -whole-program-visibility
; RUN: llvm-lto2 run -o %t.o %t.bc %t-import.bc %t1.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t.bc,f1_actual,plx -r=%t.bc,f2,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx -r=%t-import.bc,importf2,plx -r=%t-import.bc,f2,lx -r=%t1.bc,vt1,plx -whole-program-visibility
; RUN: ls %t.cache | count 4

; Three resolutions for typeid2: Indir, SingleImpl, UniqueRetVal
; where both t and t-import are sensitive to typeid2's resolution
; so 6 distinct objects in total.
; RUN: rm -rf %t.cache
; RUN: llvm-lto2 run -o %t.o %t.bc %t-import.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t.bc,f2,plx -r=%t.bc,f1_actual,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx -r=%t-import.bc,importf2,plx -r=%t-import.bc,f2,lx -whole-program-visibility
; RUN: llvm-lto2 run -o %t.o %t.bc %t-import.bc %t2.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t.bc,f2,plx -r=%t.bc,f1_actual,plx -r=%t2.bc,vt2,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx -r=%t-import.bc,importf2,plx -r=%t-import.bc,f2,lx -whole-program-visibility
; RUN: llvm-lto2 run -o %t.o %t.bc %t-import.bc %t3.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t.bc,f2,plx -r=%t.bc,f1_actual,plx -r=%t3.bc,vt2a,plx -r=%t3.bc,vt2b,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx -r=%t-import.bc,importf2,plx -r=%t-import.bc,f2,lx -whole-program-visibility
; RUN: ls %t.cache | count 6

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@f1 = alias i1(ptr), ptr @f1_actual

define i1 @f1_actual(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

define i1 @f2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
