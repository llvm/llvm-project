; REQUIRES: x86-registered-target

; Test that multi-target speculative devirtualization works with LTO
; when the call has 2 known targets.

; Index based WPD
; RUN: opt -passes=assign-guid -thinlto-bc -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZN1B1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   -r=%t2.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Hybrid WPD
; RUN: opt -passes=assign-guid -thinlto-bc -thinlto-split-lto-unit -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv \
; RUN:   -r=%t2.o,_ZN1B1fEv \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZN1B1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A \
; RUN:   -r=%t2.o,_ZTV1B \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   -r=%t2.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Regular LTO WPD
; RUN: opt -passes=assign-guid -o %t4.o %s
; RUN: llvm-lto2 run %t4.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=2 -save-temps -pass-remarks=. \
; RUN:   -o %t5 \
; RUN:   -r=%t4.o,test,px \
; RUN:   -r=%t4.o,_ZN1A1fEv,p \
; RUN:   -r=%t4.o,_ZN1B1fEv,p \
; RUN:   -r=%t4.o,_ZTV1A,px \
; RUN:   -r=%t4.o,_ZTV1B,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t5.0.4.opt.bc -o - | FileCheck %s

; Check that the call is not devirtualized if the parameter for the maximum
; number of targets is lower than 2.

; Index based WPD
; RUN: opt -passes=assign-guid -thinlto-bc -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=1 -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZN1B1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   -r=%t2.o,_ZTV1B,px 2>&1
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s --check-prefix=NO-DEVIRT

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; REMARK: speculative: devirtualized a call to _ZN1A1fEv{{.*}}, _ZN1B1fEv

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !0
@_ZTV1B = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B1fEv] }, !type !0

define i32 @test(ptr %p) {
entry:
  ; CHECK-LABEL: define i32 @test(
  ; CHECK-NEXT:  [[ENTRY:.*]]:
  ; CHECK-NEXT:   [[VPTR:%.*]] = load ptr, ptr %p
  ; CHECK-NEXT:   [[FPTR:%.*]] = load ptr, ptr [[VPTR:%.*]]
  ; CHECK-NEXT:   [[ICMP1:%.*]] = icmp eq ptr [[FPTR]], @_ZN1A1fEv
  ; CHECK-NEXT:   br i1 [[ICMP1]], label %[[IF_END:.*]], label %[[IF_FALSE1:.*]],
  ;
  ; CHECK:       [[IF_FALSE1]]:
  ; CHECK-NEXT:   [[ICMP2:%.*]] = icmp eq ptr [[FPTR]], @_ZN1B1fEv
  ; CHECK-NEXT:   br i1 [[ICMP2]], label %[[IF_END]], label %[[IF_FALSE2:.*]],
  ;
  ; CHECK:       [[IF_FALSE2]]:
  ; CHECK-NEXT:   [[TMP1:%.*]] = tail call i32 [[FPTR]]
  ; CHECK-NEXT:   br label %[[IF_END]]
  ;
  ; CHECK:       [[IF_END]]:
  ; CHECK-NEXT:   [[TMP2:%.*]] = phi i32
  ; CHECK-NEXT:   ret i32 [[TMP2]]
  ;
  ; NO-DEVIRT-LABEL: define i32 @test(
  ; NO-DEVIRT-NEXT:  [[ENTRY:.*]]:
  ; NO-DEVIRT-NEXT:   [[VPTR:%.*]] = load ptr, ptr %p
  ; NO-DEVIRT-NEXT:   [[FPTR:%.*]] = load ptr, ptr [[VPTR:%.*]]
  ; NO-DEVIRT-NEXT:   [[RET:%.*]] = tail call i32 [[FPTR]](
  ; NO-DEVIRT-NEXT:   ret i32 [[RET]]
  %0 = load ptr, ptr %p
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  %3 = call i32 %2(ptr %p)
  ret i32 %3
}

define i32 @_ZN1A1fEv(ptr %this) { ret i32 12 }
define i32 @_ZN1B1fEv(ptr %this) { ret i32 23 }

declare i1 @llvm.public.type.test(ptr, metadata)

!0 = !{i64 16, !"_ZTS1A"}
