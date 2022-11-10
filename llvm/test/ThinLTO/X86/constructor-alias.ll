;; The constructor alias example is reduced from
;;
;; template <typename T>
;; struct A { A() {} virtual ~A() {} };
;; template struct A<void>;
;; void *foo() { return new A<void>; }
;;
;; clang -c -fpic -O1 -flto=thin a.cc && cp a.o b.o && ld.lld -shared a.o b.so

; RUN: opt -module-summary %s -o %t1.bc
; RUN: cp %t1.bc %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -r=%t1.bc,_ZTV1A,pl -r=%t1.bc,_ZN1AD0Ev,pl -r=%t1.bc,_ZN1AD1Ev,pl -r=%t1.bc,_ZN1AD2Ev,pl -r=%t1.bc,D1_a,pl -r=%t1.bc,D1_a_a,pl \
; RUN:    -r=%t2.bc,_ZTV1A,l -r=%t2.bc,_ZN1AD0Ev,l -r=%t2.bc,_ZN1AD1Ev,l -r=%t2.bc,_ZN1AD2Ev,l -r=%t2.bc,D1_a,l -r=%t2.bc,D1_a_a,l -o %t3 --save-temps
; RUN: llvm-dis < %t3.2.1.promote.bc | FileCheck %s

; CHECK: @_ZTV1A = available_externally dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1AD1Ev, ptr @_ZN1AD0Ev] }
; CHECK: @D1_a = available_externally dso_local unnamed_addr alias void (ptr), ptr @_ZN1AD1Ev
; CHECK: @_ZN1AD1Ev = available_externally dso_local unnamed_addr alias void (ptr), ptr @_ZN1AD2Ev
; CHECK: @D1_a_a = available_externally dso_local unnamed_addr alias void (ptr), ptr @D1_a
; CHECK: define available_externally dso_local void @_ZN1AD2Ev(ptr noundef nonnull %0) unnamed_addr {
; CHECK: define available_externally dso_local void @_ZN1AD0Ev(ptr noundef nonnull %0) unnamed_addr {

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AD5Ev = comdat any
$_ZTV1A = comdat any

@_ZTV1A = weak_odr unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1AD1Ev, ptr @_ZN1AD0Ev] }, comdat

@D1_a = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AD1Ev
@_ZN1AD1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AD2Ev
@D1_a_a = weak_odr unnamed_addr alias void (ptr), ptr @D1_a

define weak_odr void @_ZN1AD2Ev(ptr noundef nonnull %0) unnamed_addr comdat($_ZN1AD5Ev) {
  ret void
}

define weak_odr void @_ZN1AD0Ev(ptr noundef nonnull %0) unnamed_addr comdat($_ZN1AD5Ev) {
  call void @D1_a(ptr noundef nonnull %0)
  call void @D1_a_a(ptr noundef nonnull %0)
  call void @_ZN1AD1Ev(ptr noundef nonnull %0)
  ret void
}
