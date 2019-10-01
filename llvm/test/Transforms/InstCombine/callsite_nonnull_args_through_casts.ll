; RUN: opt -instcombine -S < %s | FileCheck %s
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @foo(i8*)
declare void @bar(i8 addrspace(1)*)

define void @nonnullAfterBitCast() {
entry:
  %i = alloca i32, align 4
  %tmp1 = bitcast i32* %i to i8*
; CHECK: call void @foo(i8* nonnull %tmp1)
  call void @foo(i8* %tmp1)
  ret void
}

define void @nonnullAfterSExt(i8 %a) {
entry:
  %b = zext i8 %a to i32              ; <- %b is >= 0
  %c = add nsw nuw i32 %b, 2          ; <- %c is > 0
  %sext = sext i32 %c to i64          ; <- %sext cannot be 0 because %c is not 0
  %i2p = inttoptr i64 %sext to i8*    ; <- no-op int2ptr cast
; CHECK: call void @foo(i8* nonnull %i2p)
  call void @foo(i8* %i2p)
  ret void
}

define void @nonnullAfterZExt(i8 %a) {
entry:
  %b = zext i8 %a to i32              ; <- %b is >= 0
  %c = add nsw nuw i32 %b, 2          ; <- %c is > 0
  %zext = zext i32 %c to i64          ; <- %zext cannot be 0 because %c is not 0
  %i2p = inttoptr i64 %zext to i8*    ; <- no-op int2ptr cast
; CHECK: call void @foo(i8* nonnull %i2p)
  call void @foo(i8* %i2p)
  ret void
}

declare void @llvm.assume(i1 %b)

define void @nonnullAfterInt2Ptr(i32 %u, i64 %lu) {
entry:
  %nz = sdiv exact i32 100, %u         ; %nz cannot be null
  %i2p = inttoptr i32 %nz to i8*       ; extending int2ptr as sizeof(i32) < sizeof(i8*)
; CHECK:  call void @foo(i8* nonnull %i2p)
  call void @foo(i8* %i2p)

  %nz.2 = sdiv exact i64 100, %lu      ; %nz.2 cannot be null
  %i2p.2 = inttoptr i64 %nz.2 to i8*   ; no-op int2ptr as sizeof(i64) == sizeof(i8*)
; CHECK:  call void @foo(i8* nonnull %i2p.2)
  call void @foo(i8* %i2p.2)
  ret void
}

define void @nonnullAfterPtr2Int() {
entry:
  %a = alloca i32
  %p2i = ptrtoint i32* %a to i64      ; no-op ptr2int as sizeof(i32*) == sizeof(i64)
  %i2p = inttoptr i64 %p2i to i8*
; CHECK:  call void @foo(i8* nonnull %i2p)
  call void @foo(i8* %i2p)
  ret void
}

define void @maybenullAfterInt2Ptr(i128 %llu) {
entry:
  %cmp = icmp ne i128 %llu, 0
  call void @llvm.assume(i1 %cmp)          ; %llu != 0
  %i2p = inttoptr i128 %llu to i8*    ; truncating int2ptr as sizeof(i128) > sizeof(i8*)
; CHECK:  call void @foo(i8* %i2p)
  call void @foo(i8* %i2p)
  ret void
}

define void @maybenullAfterPtr2Int() {
entry:
  %a = alloca i32
  %p2i = ptrtoint i32* %a to i32      ; truncating ptr2int as sizeof(i32*) > sizeof(i32)
  %i2p = inttoptr i32 %p2i to i8*
; CHECK:  call void @foo(i8* %i2p)
  call void @foo(i8* %i2p)
  ret void
}

define void @maybenullAfterAddrspacecast(i8* nonnull %p) {
entry:
  %addrspcast = addrspacecast i8* %p to i8 addrspace(1)*

; An address space cast can be "a no-op cast or a complex value modification,
; depending on the target and the address space pair". As a consequence, we
; cannot simply assume non-nullness of %p is preserved by the cast.
;
; CHECK:  call void @bar(i8 addrspace(1)* %addrspcast)
  call void @bar(i8 addrspace(1)* %addrspcast)

; CHECK:  call void @foo(i8* nonnull %p)
  call void @foo(i8* %p)
  ret void
}
