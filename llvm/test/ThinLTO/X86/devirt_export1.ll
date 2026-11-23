; REQUIRES: x86-registered-target

; Test that devirtualized local targets are promoted to external when
; using Thin/Hybrid LTO.

; Index based WPD.
; RUN: opt -thinlto-bc -o %t1.o %s
; RUN: opt -thinlto-bc -o %t2.o %p/Inputs/devirt_export1.ll
; RUN: llvm-lto2 run %t1.o %t2.o -devirtualize-speculatively -save-temps -thinlto-emit-indexes -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t1.o,test,px \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1.o.thinlto.bc -o - | FileCheck %s --check-prefix=SUMMARY
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-THIN-LTO1
; RUN: llvm-dis %t3.2.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-THIN-LTO2

; Hybrid WPD.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t5.o %s
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t6.o %p/Inputs/devirt_export1.ll
; RUN: llvm-lto2 run %t5.o %t6.o -devirtualize-speculatively -save-temps -thinlto-emit-indexes -pass-remarks=. \
; RUN:   -o %t7 \
; RUN:   -r=%t5.o,test,px \
; RUN:   -r=%t6.o,_ZTV1A \
; RUN:   -r=%t6.o,_ZTV1A,px \
; RUN:   -r=%t6.o,_ZN1A1fEv \
; RUN:   -r=%t6.o,_ZN1A1fEv,p \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t5.o.thinlto.bc -o - | FileCheck %s --check-prefix=SUMMARY
; RUN: llvm-dis %t7.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-HYBRID-LTO1
; RUN: llvm-dis %t7.2.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-HYBRID-LTO2

; REMARK: devirtualized a call to _ZN1A1fEv

; SUMMARY: wpdRes: (kind: speculative, singleImplName: "_ZN1A1fEv.llvm.{{[0-9a-f]+}}")

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test(ptr %obj) {
entry:
  ; CHECK-THIN-LTO1-LABEL: define{{.*}} @test(
  ; CHECK-THIN-LTO1: @_ZN1A1fEv.llvm

  ; CHECK-HYBRID-LTO1-LABEL: define{{.*}} @test(
  ; CHECK-HYBRID-LTO1: @_ZN1A1fEv.llvm
  %0 = load ptr, ptr %obj
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  %3 = call i32 %2(ptr %obj)
  ret i32 %3
}

; CHECK-THIN-LTO2: define hidden{{.*}} @_ZN1A1fEv.llvm.
; CHECK-HYBRID-LTO2: define hidden{{.*}} @_ZN1A1fEv.llvm.
