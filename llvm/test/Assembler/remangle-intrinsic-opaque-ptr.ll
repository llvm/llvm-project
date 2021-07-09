; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s --check-prefix=TYPED
; RUN: llvm-as --force-opaque-pointers < %s | llvm-dis --force-opaque-pointers | FileCheck %s --check-prefix=OPAQUE

; An opaque pointer type should not be accepted for an intrinsic that
; specifies a fixed pointer type, outside of --force-opaque-pointers mode.

define void @test() {
; TYPED: Intrinsic has incorrect return type!
; OPAQUE: call ptr @llvm.stacksave()
  call ptr @llvm.stacksave()

; TYPED: Intrinsic has incorrect argument type!
; OPAQUE: call <2 x i64> @llvm.masked.expandload.v2i64(ptr null, <2 x i1> zeroinitializer, <2 x i64> zeroinitializer)
  call <2 x i64> @llvm.masked.expandload.v2i64(ptr null, <2 x i1> zeroinitializer, <2 x i64> zeroinitializer)

  ret void
}

declare ptr @llvm.stacksave()
declare <2 x i64> @llvm.masked.expandload.v2i64(ptr, <2 x i1>, <2 x i64>)
