; RUN: opt -opaque-pointers=0 -S -passes=instcombine < %s | FileCheck %s

; Instcombine should preserve metadata and alignment while
; folding a bitcast into a store.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.A = type { i32 (...)** }

@G = external constant [5 x i8*]

; CHECK-LABEL: @foo
; CHECK: store i32 %x, i32* %{{.*}}, align 16, !noalias !0, !llvm.access.group !3
define void @foo(i32 %x, float* %p) nounwind {
entry:
  %x.cast = bitcast i32 %x to float
  store float %x.cast, float* %p, align 16, !noalias !0, !llvm.access.group !3
  ret void
}

; Check instcombine doesn't try and fold the following bitcast into the store.
; This transformation would not be safe since we would need to use addrspacecast
; and addrspacecast is not guaranteed to be a no-op cast.

; CHECK-LABEL: @bar
; CHECK: %cast = bitcast i8** %b to i8 addrspace(1)**
; CHECK: store i8 addrspace(1)* %a, i8 addrspace(1)** %cast
define void @bar(i8 addrspace(1)* %a, i8** %b) nounwind {
entry:
  %cast = bitcast i8** %b to i8 addrspace(1)**
  store i8 addrspace(1)* %a, i8 addrspace(1)** %cast
  ret void
}

; Check that we don't combine the bitcast into the store. This would create a
; bitcast of the swifterror which is invalid.

; CHECK-LABEL: @swifterror_store
; CHECK: bitcast i64
; CHECK: store %swift.error

%swift.error = type opaque
define void @swifterror_store(i64* %x, %swift.error** swifterror %err) {
entry:
  %casted = bitcast i64* %x to %swift.error*
  store %swift.error* %casted, %swift.error** %err
  ret void
}

; CHECK-LABEL: @ppcf128_ones_store
; CHECK: store ppc_fp128 0xMFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, ppc_fp128* %dest, align 16
define void @ppcf128_ones_store(ppc_fp128* %dest) {
entry:
  %int = or i128 0, 340282366920938463463374607431768211455 ; 128 ones
  %val = bitcast i128 %int to ppc_fp128
  store ppc_fp128 %val, ppc_fp128* %dest, align 16
  ret void
}

!0 = !{!1}
!1 = !{!1, !2}
!2 = !{!2}
!3 = !{}
