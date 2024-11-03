; Test that linkonce_odr and weak_odr variables which are visible to regular
; object (and so are not readonly) are not internalized by thin LTO.
; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto2 run -save-temps %t.bc -o %t.out \
; RUN:               -r=%t.bc,_ZL5initSv,plx \
; RUN:               -r=%t.bc,_ZN9SingletonI1SE11getInstanceEv,lx \
; RUN:               -r=%t.bc,_ZZN9SingletonI1SE11getInstanceEvE8instance,lx \
; RUN:               -r=%t.bc,_ZZN9SingletonI1SE11getInstanceEvE13instance_weak,lx
; RUN: llvm-dis %t.out.1.1.promote.bc -o - | FileCheck %s
; RUN: llvm-dis %t.out.1.2.internalize.bc -o - | FileCheck %s --check-prefix=INTERNALIZE

; CHECK: @_ZZN9SingletonI1SE11getInstanceEvE8instance = available_externally dso_local global %struct.S zeroinitializer
; CHECK: @_ZZN9SingletonI1SE11getInstanceEvE13instance_weak = available_externally dso_local global ptr null, align 8

;; We should not internalize a linkonce_odr function when the IR definition(s)
;; are not prevailing (prevailing def in native object). This can break function
;; pointer equality (unless it has an unnamed_addr attribute indicating that the
;; address is not significant), and also can increase code size.
; CHECK: define available_externally dso_local dereferenceable(16) ptr @_ZN9SingletonI1SE11getInstanceEv()
; INTERNALIZE: define available_externally dso_local dereferenceable(16) ptr @_ZN9SingletonI1SE11getInstanceEv()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i64, i64 }

$_ZN9SingletonI1SE11getInstanceEv = comdat any

$_ZZN9SingletonI1SE11getInstanceEvE8instance = comdat any

$_ZZN9SingletonI1SE11getInstanceEvE13instance_weak = comdat any

@_ZZN9SingletonI1SE11getInstanceEvE8instance = linkonce_odr dso_local global %struct.S zeroinitializer, comdat, align 8

@_ZZN9SingletonI1SE11getInstanceEvE13instance_weak = weak_odr dso_local global ptr null, comdat, align 8

define dso_local void @_ZL5initSv() {
  %1 = call dereferenceable(16) ptr @_ZN9SingletonI1SE11getInstanceEv()
  store  ptr %1, ptr @_ZZN9SingletonI1SE11getInstanceEvE13instance_weak
  store i64 1, ptr %1, align 8
  ret void
}

define linkonce_odr dso_local dereferenceable(16) ptr @_ZN9SingletonI1SE11getInstanceEv() #0 comdat align 2 {
  ret ptr @_ZZN9SingletonI1SE11getInstanceEvE8instance
}

