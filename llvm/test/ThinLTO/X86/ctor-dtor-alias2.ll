;; Test mixed D0/D2 and D5 COMDATs. Reduced from:
;;
;; // a.cc
;; template <typename T>
;; struct A final { virtual ~A() {} };
;; extern "C" void aa() { A<int> a; }
;; // b.cc
;; template <typename T>
;; struct A final { virtual ~A() {} };
;; template struct A<int>;
;; extern "C" void bb(A<int> *a) { delete a; }
;;
;; clang -c -fpic -O0 -flto=thin a.cc && ld.lld -shared a.o b.o
;;
;; The file matches lld/test/ELF/lto/ctor-dtor-alias2.ll

; RUN: rm -rf %t && split-file %s %t && cd %t

;; a.bc defines D0 in comdat D0 and D2 in comdat D2. b.bc defines D0/D1/D2 in comdat D5.
; RUN: opt -module-summary a.ll -o a.bc
; RUN: opt -module-summary b.ll -o b.bc
; RUN: llvm-lto2 run a.bc b.bc -r=a.bc,aa,px -r=a.bc,_ZN1AIiED0Ev,px -r=a.bc,_ZN1AIiED2Ev,px -r=a.bc,_ZdlPv, \
; RUN:   -r=b.bc,bb,px -r=b.bc,_ZN1AIiED0Ev, -r=b.bc,_ZN1AIiED1Ev,px -r=b.bc,_ZN1AIiED2Ev, -r=b.bc,_ZdlPv, -o out --save-temps
; RUN: llvm-dis < out.2.1.promote.bc | FileCheck %s

;; Although D0/D2 in b.bc is non-prevailing, keep D1/D2 as definitions, otherwise
;; the output may have an undefined and unsatisfied D1.
; CHECK: @_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev
; CHECK: define weak_odr void @_ZN1AIiED2Ev(ptr noundef nonnull %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
; CHECK: define available_externally void @_ZN1AIiED0Ev(ptr noundef nonnull %this) unnamed_addr {

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED2Ev = comdat any

$_ZN1AIiED0Ev = comdat any

define void @aa() {
entry:
  %a = alloca ptr, align 8
  call void @_ZN1AIiED2Ev(ptr noundef nonnull %a)
  ret void
}

define linkonce_odr void @_ZN1AIiED2Ev(ptr noundef nonnull %this) unnamed_addr comdat {
  ret void
}

define linkonce_odr void @_ZN1AIiED0Ev(ptr noundef nonnull %this) unnamed_addr comdat {
entry:
  call void @_ZN1AIiED2Ev(ptr noundef nonnull %this)
  call void @_ZdlPv(ptr noundef %this)
  ret void
}

declare void @_ZdlPv(ptr noundef)

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED5Ev = comdat any

$_ZTV1AIiE = comdat any

@_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev

define weak_odr void @_ZN1AIiED2Ev(ptr noundef nonnull %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}

define weak_odr void @_ZN1AIiED0Ev(ptr noundef nonnull %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
entry:
  call void @_ZN1AIiED1Ev(ptr noundef nonnull %this)
  call void @_ZdlPv(ptr noundef %this)
  ret void
}

declare void @_ZdlPv(ptr noundef)

define void @bb(ptr noundef %a) {
entry:
  call void @_ZN1AIiED1Ev(ptr noundef nonnull %a)
  call void @_ZdlPv(ptr noundef %a)
  ret void
}
