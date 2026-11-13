; REQUIRES: x86-registered-target

; Test that declarations are not considered as empty functions and won't be
; ignored as a virtual call target during hybrid LTO.

; RUN: opt -passes=assign-guid -thinlto-bc -thinlto-split-lto-unit -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A \
; RUN:   -r=%t2.o,_ZTV1A,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; REMARK: <unknown>:0:0: single-impl: devirtualized a call to _ZN1A1fEv

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !0

define void @test(ptr %p) {
entry:
  ; CHECK-LABEL: define void @test(
  ; CHECK-NEXT:  [[ENTRY:.*]]:
  ; CHECK:         [[ICMP:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1A1fEv
  ; CHECK-NEXT:    br i1 [[ICMP]], label %[[IF_TRUE_DIRECT:.*]], label %[[IF_FALSE_ORIG_INDIRECT:.*]],
  ; CHECK:       [[IF_TRUE_DIRECT]]:
  ; CHECK-NEXT:    store i32 0
  ; CHECK-NEXT:    br label %[[IF_END:.*]]
  ; CHECK:       [[IF_FALSE_ORIG_INDIRECT]]:
  ; CHECK-NEXT:    tail call void [[FPTR]]
  ; CHECK-NEXT:    br label %[[IF_END]]
  ; CHECK:       [[IF_END]]:
  ; CHECK-NEXT:    ret void
  %0 = load ptr, ptr %p
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  call void %2(ptr %p)
  ret void
}

define void @_ZN1A1fEv(ptr %this) {
entry:
  store i32 0, ptr %this
  ret void
}

declare i1 @llvm.public.type.test(ptr, metadata)

!0 = !{i64 16, !"_ZTS1A"}
