;; Test to ensure that the summary is correctly printed when there is both a
;; global variable summary and a vtable typeid summary entry for the same
;; symbol.

; RUN: opt %s -S -module-summary | FileCheck %s
;; Make sure it round trips correctly.
; RUN: opt %s -S -module-summary -o - | llvm-as -o - | llvm-dis -o - | FileCheck %s

;; These summary entries should get numbered differently.
; CHECK: ^2 = gv: (name: "_ZTS1A"
; CHECK: ^6 = typeidCompatibleVTable: (name: "_ZTS1A"

; ModuleID = 'thinlto-vtable-summary2.cc'
source_filename = "thinlto-vtable-summary2.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS1A = dso_local constant [3 x i8] c"1A\00", align 1
@_ZTI1A = dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, align 8

define dso_local noundef i32 @_ZN1A3fooEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret i32 1
}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
