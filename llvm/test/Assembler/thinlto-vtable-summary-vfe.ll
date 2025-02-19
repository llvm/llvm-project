;; Test that a virtual function has a typeidCompatibleVTable entry when virtual
;; function elimination is enabled.

; RUN: opt %s -S -module-summary | FileCheck %s

;; These summary entries should get numbered differently.
; CHECK: ^2 = gv: (name: "_ZTS1A"
; CHECK: ^6 = typeidCompatibleVTable: (name: "_ZTS1A"
; CHECK: typeidCompatibleVTable: (name: "_ZTSM1AFivE.virtual"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS1A = dso_local constant [3 x i8] c"1A\00", align 1
@_ZTI1A = dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }

define i32 @_ZN1A3fooEv(ptr %this) {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  ret i32 1
}

!llvm.module.flags = !{!2}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
!2 = !{i32 1, !"Virtual Function Elim", i32 1}
