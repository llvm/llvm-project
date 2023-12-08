; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memset.p0.i64(ptr %dest, i8 %val, i64 %len, i1 %isvolatile)
declare void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 %len, i1 %isvolatile)
declare void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 %isvolatile)
declare void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 %isvolatile)

define void @MemsetInBounds() {
; CHECK-LABEL: MemsetInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 4, i1 false)
  ret void
}

; Volatile does not matter for access bounds.
define void @VolatileMemsetInBounds() {
; CHECK-LABEL: VolatileMemsetInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 4, i1 true)
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 4, i1 true)
  ret void
}

define void @MemsetOutOfBounds() {
; CHECK-LABEL: MemsetOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,5){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 5, i1 false)
  ret void
}

define void @MemsetNonConst(i32 %size) {
; CHECK-LABEL: MemsetNonConst dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4294967295){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 %size, i1 false)
  ret void
}

; FIXME: memintrinsics should look at size range when possible
; Right now we refuse any non-constant size.
define void @MemsetNonConstInBounds(i1 zeroext %z) {
; CHECK-LABEL: MemsetNonConstInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,7){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %size = select i1 %z, i32 3, i32 4
  call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 %size, i1 false)
  ret void
}

define void @MemsetNonConstSize() {
; CHECK-LABEL: MemsetNonConstSize dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4294967295){{$}}
; CHECK-NEXT: y[4]: empty-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %xint = ptrtoint ptr %x to i32
  %yint = ptrtoint ptr %y to i32
  %d = sub i32 %xint, %yint
  call void @llvm.memset.p0.i32(ptr %x, i8 42, i32 %d, i1 false)
  ret void
}

define void @MemcpyInBounds() {
; CHECK-LABEL: MemcpyInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-NEXT: y[4]: [0,4){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 4, i1 false)
  ret void
}

define void @MemcpySrcOutOfBounds() {
; CHECK-LABEL: MemcpySrcOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,5){{$}}
; CHECK-NEXT: y[4]: [0,5){{$}}
; GLOBAL-NEXT: safe accesses
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %y = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 5, i1 false)
  ret void
}

define void @MemcpyDstOutOfBounds() {
; CHECK-LABEL: MemcpyDstOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,5){{$}}
; CHECK-NEXT: y[8]: [0,5){{$}}
; GLOBAL-NEXT: safe accesses
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i64, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 5, i1 false)
  ret void
}

define void @MemcpyBothOutOfBounds() {
; CHECK-LABEL: MemcpyBothOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,9){{$}}
; CHECK-NEXT: y[8]: [0,9){{$}}
; GLOBAL-NEXT: safe accesses
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i64, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 9, i1 false)
  ret void
}

define void @MemcpySelfInBounds() {
; CHECK-LABEL: MemcpySelfInBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,8){{$}}
; GLOBAL-NEXT: safe accesses
; GLOBAL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %x2, i32 3, i1 false)
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 5
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %x2, i32 3, i1 false)
  ret void
}

define void @MemcpySelfSrcOutOfBounds() {
; CHECK-LABEL: MemcpySelfSrcOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,9){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 5
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %x2, i32 4, i1 false)
  ret void
}

define void @MemcpySelfDstOutOfBounds() {
; CHECK-LABEL: MemcpySelfDstOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,9){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 5
  call void @llvm.memcpy.p0.p0.i32(ptr %x2, ptr %x, i32 4, i1 false)
  ret void
}

define void @MemmoveSelfBothOutOfBounds() {
; CHECK-LABEL: MemmoveSelfBothOutOfBounds dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: [0,14){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 5
  call void @llvm.memmove.p0.p0.i32(ptr %x, ptr %x2, i32 9, i1 false)
  ret void
}

define void @MemsetInBoundsCast() {
; CHECK-LABEL: MemsetInBoundsCast dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: [0,4){{$}}
; CHECK-NEXT: y[1]: empty-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memset.p0.i32(ptr %x, i8 %yint, i32 4, i1 false)
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %y = alloca i8, align 1
  %yint = ptrtoint ptr %y to i8
  call void @llvm.memset.p0.i32(ptr %x, i8 %yint, i32 4, i1 false)
  ret void
}

define void @MemcpyInBoundsCast2(i8 %zint8) {
; CHECK-LABEL: MemcpyInBoundsCast2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[256]: [0,255){{$}}
; CHECK-NEXT: y[256]: [0,255){{$}}
; CHECK-NEXT: z[1]: empty-set{{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 %zint32, i1 false)
; CHECK-EMPTY:
entry:
  %x = alloca [256 x i8], align 4
  %y = alloca [256 x i8], align 4
  %z = alloca i8, align 1
  %zint32 = zext i8 %zint8 to i32
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 %zint32, i1 false)
  ret void
}
