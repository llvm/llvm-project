; RUN: opt -S -analyze -stack-safety-local -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -S -analyze -stack-safety < %s -enable-new-pm=0 | FileCheck %s
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 %len, i1 %isvolatile)
declare void @llvm.memset.p0i8.i32(i8* %dest, i8 %val, i32 %len, i1 %isvolatile)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 %isvolatile)
declare void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 %isvolatile)

define void @MemsetInBounds() {
; CHECK-LABEL: MemsetInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i32(i8* %x1, i8 42, i32 4, i1 false)
  ret void
}

; Volatile does not matter for access bounds.
define void @VolatileMemsetInBounds() {
; CHECK-LABEL: VolatileMemsetInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i32(i8* %x1, i8 42, i32 4, i1 true)
  ret void
}

define void @MemsetOutOfBounds() {
; CHECK-LABEL: MemsetOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,5){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i32(i8* %x1, i8 42, i32 5, i1 false)
  ret void
}

define void @MemsetNonConst(i32 %size) {
; CHECK-LABEL: MemsetNonConst dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4294967295){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @llvm.memset.p0i8.i32(i8* %x1, i8 42, i32 %size, i1 false)
  ret void
}

; FIXME: memintrinsics should look at size range when possible
; Right now we refuse any non-constant size.
define void @MemsetNonConstInBounds(i1 zeroext %z) {
; CHECK-LABEL: MemsetNonConstInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,7){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %size = select i1 %z, i32 3, i32 4
  call void @llvm.memset.p0i8.i32(i8* %x1, i8 42, i32 %size, i1 false)
  ret void
}

define void @MemsetNonConstSize() {
; CHECK-LABEL: MemsetNonConstSize dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4294967295){{$}}
; CHECK-NEXT: y[4]: empty-set{{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %xint = ptrtoint i32* %x to i32
  %yint = ptrtoint i32* %y to i32
  %d = sub i32 %xint, %yint
  call void @llvm.memset.p0i8.i32(i8* %x1, i8 42, i32 %d, i1 false)
  ret void
}

define void @MemcpyInBounds() {
; CHECK-LABEL: MemcpyInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-NEXT: y[4]: [0,4){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %y1 = bitcast i32* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x1, i8* %y1, i32 4, i1 false)
  ret void
}

define void @MemcpySrcOutOfBounds() {
; CHECK-LABEL: MemcpySrcOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,5){{$}}
; CHECK-NEXT: y[4]: [0,5){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %y = alloca i32, align 4
  %x1 = bitcast i64* %x to i8*
  %y1 = bitcast i32* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x1, i8* %y1, i32 5, i1 false)
  ret void
}

define void @MemcpyDstOutOfBounds() {
; CHECK-LABEL: MemcpyDstOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,5){{$}}
; CHECK-NEXT: y[8]: [0,5){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i64, align 4
  %x1 = bitcast i32* %x to i8*
  %y1 = bitcast i64* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x1, i8* %y1, i32 5, i1 false)
  ret void
}

define void @MemcpyBothOutOfBounds() {
; CHECK-LABEL: MemcpyBothOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,9){{$}}
; CHECK-NEXT: y[8]: [0,9){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i64, align 4
  %x1 = bitcast i32* %x to i8*
  %y1 = bitcast i64* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x1, i8* %y1, i32 9, i1 false)
  ret void
}

define void @MemcpySelfInBounds() {
; CHECK-LABEL: MemcpySelfInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,8){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 5
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x1, i8* %x2, i32 3, i1 false)
  ret void
}

define void @MemcpySelfSrcOutOfBounds() {
; CHECK-LABEL: MemcpySelfSrcOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,9){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 5
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x1, i8* %x2, i32 4, i1 false)
  ret void
}

define void @MemcpySelfDstOutOfBounds() {
; CHECK-LABEL: MemcpySelfDstOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,9){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 5
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x2, i8* %x1, i32 4, i1 false)
  ret void
}

define void @MemmoveSelfBothOutOfBounds() {
; CHECK-LABEL: MemmoveSelfBothOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,14){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 5
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %x1, i8* %x2, i32 9, i1 false)
  ret void
}
