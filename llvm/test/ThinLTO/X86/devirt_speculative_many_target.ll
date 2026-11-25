; REQUIRES: x86-registered-target

; Test that speculative devirtualization works with a higher number of
; speculated targets using LTO (4 in this case).

; Index based WPD
; RUN: opt -passes=assign-guid -thinlto-bc -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=4 -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZN1B1fEv,p \
; RUN:   -r=%t2.o,_ZN1C1fEv,p \
; RUN:   -r=%t2.o,_ZN1D1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   -r=%t2.o,_ZTV1B,px \
; RUN:   -r=%t2.o,_ZTV1C,px \
; RUN:   -r=%t2.o,_ZTV1D,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Hybrid WPD
; RUN: opt -passes=assign-guid -thinlto-bc -thinlto-split-lto-unit -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=4 -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv \
; RUN:   -r=%t2.o,_ZN1B1fEv \
; RUN:   -r=%t2.o,_ZN1C1fEv \
; RUN:   -r=%t2.o,_ZN1D1fEv \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZN1B1fEv,p \
; RUN:   -r=%t2.o,_ZN1C1fEv,p \
; RUN:   -r=%t2.o,_ZN1D1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A \
; RUN:   -r=%t2.o,_ZTV1B \
; RUN:   -r=%t2.o,_ZTV1C \
; RUN:   -r=%t2.o,_ZTV1D \
; RUN:   -r=%t2.o,_ZTV1A,px \
; RUN:   -r=%t2.o,_ZTV1B,px \
; RUN:   -r=%t2.o,_ZTV1C,px \
; RUN:   -r=%t2.o,_ZTV1D,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Regular LTO WPD
; RUN: opt -passes=assign-guid -o %t4.o %s
; RUN: llvm-lto2 run %t4.o -devirtualize-speculatively -wholeprogramdevirt-max-speculative-targets=4 -save-temps -pass-remarks=. \
; RUN:   -o %t5 \
; RUN:   -r=%t4.o,test,px \
; RUN:   -r=%t4.o,_ZN1A1fEv,p \
; RUN:   -r=%t4.o,_ZN1B1fEv,p \
; RUN:   -r=%t4.o,_ZN1C1fEv,p \
; RUN:   -r=%t4.o,_ZN1D1fEv,p \
; RUN:   -r=%t4.o,_ZTV1A,px \
; RUN:   -r=%t4.o,_ZTV1B,px \
; RUN:   -r=%t4.o,_ZTV1C,px \
; RUN:   -r=%t4.o,_ZTV1D,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t5.0.4.opt.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; REMARK: speculative: devirtualized a call to _ZN1A1fEv{{.*}}, _ZN1B1fEv{{.*}}, _ZN1C1fEv{{.*}}, _ZN1D1fEv

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !0
@_ZTV1B = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B1fEv] }, !type !0
@_ZTV1C = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1C1fEv] }, !type !0
@_ZTV1D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1D1fEv] }, !type !0

define i32 @test(ptr %p) {
entry:
  ; CHECK-LABEL: define i32 @test(
  ; CHECK-NEXT:  [[ENTRY:.*]]:
  ; CHECK-NEXT:   [[VPTR:%.*]] = load ptr, ptr %p
  ; CHECK-NEXT:   [[FPTR:%.*]] = load ptr, ptr [[VPTR:%.*]]
  ; CHECK-NEXT:   [[ICMP1:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1A1fEv
  ; CHECK-NEXT:   br i1 [[ICMP1:%.*]], label %[[IF_END:.*]], label %[[IF_FALSE1:.*]],
  ;
  ; CHECK:       [[IF_FALSE1]]:
  ; CHECK-NEXT:   [[ICMP2:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1B1fEv
  ; CHECK-NEXT:   br i1 [[ICMP2:%.*]], label %[[IF_END:.*]], label %[[IF_FALSE2:.*]],
  ;
  ; CHECK:       [[IF_FALSE2]]:
  ; CHECK-NEXT:   [[ICMP3:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1C1fEv
  ; CHECK-NEXT:   br i1 [[ICMP3:%.*]], label %[[IF_END:.*]], label %[[IF_FALSE3:.*]],
  ;
  ; CHECK:       [[IF_FALSE3]]:
  ; CHECK-NEXT:   [[ICMP4:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1D1fEv
  ; CHECK-NEXT:   br i1 [[ICMP4:%.*]], label %[[IF_END:.*]], label %[[IF_FALSE4:.*]],
  ;
  ; CHECK:       [[IF_FALSE4]]:
  ; CHECK-NEXT:   [[TMP1:%.*]] = tail call i32 [[FPTR:%.*]](
  ; CHECK-NEXT:   br label %[[IF_END:.*]]
  ;
  ; CHECK:       [[IF_END:.*]]:
  ; CHECK-NEXT:   [[TMP2:%.*]] = phi i32
  ; CHECK-NEXT:   ret i32 [[TMP2]]
  %0 = load ptr, ptr %p
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  %3 = call i32 %2(ptr %p)
  ret i32 %3
}

define i32 @_ZN1A1fEv(ptr %this) { ret i32 12 }
define i32 @_ZN1B1fEv(ptr %this) { ret i32 23 }
define i32 @_ZN1C1fEv(ptr %this) { ret i32 34 }
define i32 @_ZN1D1fEv(ptr %this) { ret i32 45 }

declare i1 @llvm.public.type.test(ptr, metadata)

!0 = !{i64 16, !"_ZTS1A"}
