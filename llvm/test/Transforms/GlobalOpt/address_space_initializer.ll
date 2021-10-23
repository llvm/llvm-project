; RUN: opt -passes=globalopt < %s -S | FileCheck %s
; RUN: opt -passes=globalopt --mtriple=nvptx64 < %s -S | FileCheck %s --check-prefix=GPU
; RUN: opt -passes=globalopt --mtriple=amdgcn < %s -S | FileCheck %s --check-prefix=GPU
; REQUIRES: amdgpu-registered-target, nvptx-registered-target

; Check that we don't try to set a global initializer for non AS(0) globals.

@g0 = internal global i16 undef
@g1 = internal addrspace(3) global i16 undef
@g2 = internal addrspace(1) global i16 undef
; CHECK-NOT: @g0 =
; CHECK: internal unnamed_addr addrspace(3) global i16 undef
; CHECK: internal unnamed_addr addrspace(1) global i16 undef
; GPU-NOT: @g0 =
; GPU: internal unnamed_addr addrspace(3) global i16 undef
; GPU-NOT: @g2 =

define void @a() {
  store i16 3, i16* @g0, align 8
  store i16 5, i16* addrspacecast (i16 addrspace(3)* @g1 to i16*), align 8
  store i16 7, i16* addrspacecast (i16 addrspace(1)* @g2 to i16*), align 8
  ret void
}

define i8 @get0() {
  %bc = bitcast i16* @g0 to i8*
  %gep = getelementptr i8, i8* %bc, i64 1
  %r = load i8, i8* %gep
  ret i8 %r
}
define i8 @get1() {
  %ac = addrspacecast i16 addrspace(3)* @g1 to i16*
  %bc = bitcast i16* %ac to i8*
  %gep = getelementptr i8, i8* %bc, i64 1
  %r = load i8, i8* %gep
  ret i8 %r
}
define i8 @get2() {
  %ac = addrspacecast i16 addrspace(1)* @g2 to i16*
  %bc = bitcast i16* %ac to i8*
  %gep = getelementptr i8, i8* %bc, i64 1
  %r = load i8, i8* %gep
  ret i8 %r
}
