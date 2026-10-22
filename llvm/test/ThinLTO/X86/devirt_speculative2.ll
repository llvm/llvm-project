; REQUIRES: x86-registered-target

; Test multi-target speculative devirtualization requiring promotion of local targets.

; Index based WPD
; RUN: opt -thinlto-bc -o %t1.o %s
; RUN: opt -thinlto-bc -o %t2.o %p/Inputs/devirt_speculative2.ll
; RUN: llvm-lto2 run %t1.o %t2.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t1.o,test,px \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   -r=%t2.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Hybrid WPD
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t4.o %s
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t5.o %p/Inputs/devirt_speculative2.ll
; RUN: llvm-lto2 run %t4.o %t5.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -save-temps -pass-remarks=. \
; RUN:   -o %t6 \
; RUN:   -r=%t4.o,test,px \
; RUN:   -r=%t5.o,_ZN1A1fEv \
; RUN:   -r=%t5.o,_ZN1B1fEv \
; RUN:   -r=%t5.o,_ZN1A1fEv,p \
; RUN:   -r=%t5.o,_ZN1B1fEv,p \
; RUN:   -r=%t5.o,_ZTV1A \
; RUN:   -r=%t5.o,_ZTV1B \
; RUN:   -r=%t5.o,_ZTV1A,px \
; RUN:   -r=%t5.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t6.1.4.opt.bc -o - | FileCheck %s

; Regular LTO WPD
; RUN: opt -o %t7.o %s
; RUN: opt -o %t8.o %p/Inputs/devirt_speculative2.ll
; RUN: llvm-lto2 run %t7.o %t8.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -save-temps -pass-remarks=. \
; RUN:   -o %t9 \
; RUN:   -r=%t7.o,test,px \
; RUN:   -r=%t8.o,_ZTV1A,px \
; RUN:   -r=%t8.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t9.0.4.opt.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; REMARK: speculative: devirtualized a call to _ZN1A1fEv{{.*}}, _ZN1B1fEv

define i32 @test(ptr %p) {
entry:
  ; CHECK: icmp eq ptr %{{.*}}, @_ZN1A1fEv
  ; CHECK: icmp eq ptr %{{.*}}, @_ZN1B1fEv
  %0 = load ptr, ptr %p
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  %3 = call i32 %2(ptr %p)
  ret i32 %3
}

declare i1 @llvm.public.type.test(ptr, metadata)
declare void @llvm.assume(i1)
