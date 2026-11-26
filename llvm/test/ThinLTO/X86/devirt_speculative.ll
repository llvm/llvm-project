; REQUIRES: x86-registered-target

; Test that speculative devirtualization works with LTO

; Index based WPD
; RUN: opt -passes=assign-guid -thinlto-bc -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Hybrid WPD
; RUN: opt -passes=assign-guid -thinlto-bc -thinlto-split-lto-unit -o %t2.o %s
; RUN: llvm-lto2 run %t2.o -devirtualize-speculatively -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t2.o,test,px \
; RUN:   -r=%t2.o,_ZN1A1fEv \
; RUN:   -r=%t2.o,_ZN1A1fEv,p \
; RUN:   -r=%t2.o,_ZTV1A \
; RUN:   -r=%t2.o,_ZTV1A,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

; Regular LTO WPD
; RUN: opt -passes=assign-guid -o %t4.o %s
; RUN: llvm-lto2 run %t4.o -devirtualize-speculatively -save-temps -pass-remarks=. \
; RUN:   -o %t5 \
; RUN:   -r=%t4.o,test,px \
; RUN:   -r=%t4.o,_ZN1A1fEv,p \
; RUN:   -r=%t4.o,_ZTV1A,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t5.0.4.opt.bc -o - | FileCheck %s

; Index based WPD, distributed backends. The thin link records in the combined
; index that speculative devirtualization is enabled, so the backend applies
; the speculative resolution without being passed the option itself.
; RUN: opt -passes=assign-guid -thinlto-bc -o %t6.o %s
; RUN: llvm-lto2 run %t6.o -devirtualize-speculatively -save-temps \
; RUN:   -thinlto-distributed-indexes \
; RUN:   -o %t7 \
; RUN:   -r=%t6.o,test,px \
; RUN:   -r=%t6.o,_ZN1A1fEv,p \
; RUN:   -r=%t6.o,_ZTV1A,px
; RUN: llvm-dis %t6.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX
; RUN: opt -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import \
; RUN:   -wholeprogramdevirt-read-summary=%t6.o.thinlto.bc -S %t6.o -o - \
; RUN:   | FileCheck %s --check-prefix=DIST

; Index based WPD, legacy ThinLTOCodeGenerator
; RUN: llvm-lto -thinlto-action=run -devirtualize-speculatively %t6.o --thinlto-save-temps=%t8. --pass-remarks=. --exported-symbol=test --exported-symbol=_ZN1A1fEv --exported-symbol=_ZTV1A 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t8.0.4.opt.bc -o - | FileCheck %s

; INDEX: wpdRes: (kind: speculative, singleImplName: "_ZN1A1fEv")

; DIST-LABEL: define i32 @test(
; DIST:         [[ICMP:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1A1fEv
; DIST-NEXT:    br i1 [[ICMP]], label %[[IF_TRUE_DIRECT:.*]], label %[[IF_FALSE_ORIG_INDIRECT:.*]],
; DIST:       [[IF_TRUE_DIRECT]]:
; DIST-NEXT:    call i32 @_ZN1A1fEv(
; DIST:       [[IF_FALSE_ORIG_INDIRECT]]:
; DIST-NEXT:    call i32 [[FPTR]](

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; REMARK: <unknown>:0:0: single-impl: devirtualized a call to _ZN1A1fEv

%struct.A = type { ptr }

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !0, !vcall_visibility !1

define i32 @test(ptr %p) {
entry:
  ; CHECK-LABEL: define i32 @test(
  ; CHECK-NEXT:  [[ENTRY:.*]]:
  ; CHECK:         [[ICMP:%.*]] = icmp eq ptr [[FPTR:%.*]], @_ZN1A1fEv
  ; CHECK-NEXT:    br i1 [[ICMP]], label %[[IF_END:.*]], label %[[IF_FALSE_ORIG_INDIRECT:.*]],
  ; CHECK:       [[IF_FALSE_ORIG_INDIRECT]]:
  ; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 [[FPTR]]
  ; CHECK-NEXT:    br label %[[IF_END]]
  ; CHECK:       [[IF_END]]:
  ; CHECK-NEXT:    [[TMP1:%.*]] = phi i32 [ [[TMP0]], %[[IF_FALSE_ORIG_INDIRECT]] ], [ 12, %[[ENTRY]] ]
  ; CHECK-NEXT:    ret i32 [[TMP1]]
  %0 = load ptr, ptr %p
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  %3 = call i32 %2(ptr %p)
  ret i32 %3
}

define i32 @_ZN1A1fEv(ptr %this) {
entry:
  ret i32 12
}

declare i1 @llvm.public.type.test(ptr, metadata)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 0}
