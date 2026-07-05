; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

declare ptr @random.i32(ptr %ptr)
declare ptr @random.i8(ptr %ptr)

; CHECK-LABEL: Function: arr:
; CHECK-DAG: MayAlias: i32* %alloca, i32* %p0
; CHECK-DAG: NoAlias:  i32* %alloca, i32* %p1
define void @arr() {
  %alloca = alloca i32, i32 4
  %random = call ptr @random.i32(ptr %alloca)
  %p0 = getelementptr inbounds i32, ptr %random, i32 0
  %p1 = getelementptr inbounds i32, ptr %random, i32 1
  load i32, ptr %alloca
  load i32, ptr %p0
  load i32, ptr %p1
  ret void
}

; CHECK-LABEL: Function: arg:
; CHECK-DAG: MayAlias: i32* %arg, i32* %p0
; CHECK-DAG: MayAlias: i32* %arg, i32* %p1
define void @arg(ptr %arg) {
  %random = call ptr @random.i32(ptr %arg)
  %p0 = getelementptr inbounds i32, ptr %random, i32 0
  %p1 = getelementptr inbounds i32, ptr %random, i32 1
  load i32, ptr %arg
  load i32, ptr %p0
  load i32, ptr %p1
  ret void
}

@gv = global i32 1
; CHECK-LABEL: Function: global:
; CHECK-DAG: MayAlias: i32* %random, i32* @gv
; CHECK-DAG: NoAlias:  i32* %p1, i32* @gv
define void @global() {
  %random = call ptr @random.i32(ptr @gv)
  %p1 = getelementptr inbounds i32, ptr %random, i32 1
  load i32, ptr @gv
  load i32, ptr %random
  load i32, ptr %p1
  ret void
}

; CHECK-LABEL: Function: struct:
; CHECK-DAG:  MayAlias: i32* %alloca, i32* %p0
; CHECK-DAG:  MayAlias: i32* %f1, i32* %p0
; CHECK-DAG:  NoAlias:  i32* %alloca, i32* %p1
; CHECK-DAG:  MayAlias: i32* %f1, i32* %p1
%struct = type { i32, i32, i32 }
define void @struct() {
  %alloca = alloca %struct
  %random = call ptr @random.i32(ptr %alloca)
  %f1 = getelementptr inbounds %struct, ptr %alloca, i32 0, i32 1
  %p0 = getelementptr inbounds i32, ptr %random, i32 0
  %p1 = getelementptr inbounds i32, ptr %random, i32 1
  load i32, ptr %alloca
  load i32, ptr %f1
  load i32, ptr %p0
  load i32, ptr %p1
  ret void
}

; CHECK-LABEL: Function: complex1:
; CHECK-DAG:  MayAlias:     i32* %a2.0, i32* %r2.0
; CHECK-DAG:  NoAlias:      i32* %a2.0, i32* %r2.1
; CHECK-DAG:  MayAlias:     i32* %a2.0, i32* %r2.i
; CHECK-DAG:  MayAlias:     i32* %a2.0, i32* %r2.1i
; CHECK-DAG:  NoAlias:      i32* %a1, i32* %r2.0
; CHECK-DAG:  NoAlias:      i32* %a1, i32* %r2.1
; CHECK-DAG:  MayAlias:     i32* %a1, i32* %r2.i
; CHECK-DAG:  MayAlias:     i32* %a1, i32* %r2.1i
%complex = type { i32, i32, [4 x i32] }
define void @complex1(i32 %i) {
  %alloca = alloca %complex
  %r.i32 = call ptr @random.i32(ptr %alloca)
  %a1 = getelementptr inbounds %complex, ptr %alloca, i32 0, i32 1
  %a2.0 = getelementptr inbounds %complex, ptr %alloca, i32 0, i32 2, i32 0
  %r2.0 = getelementptr inbounds %complex, ptr %r.i32, i32 0, i32 2, i32 0
  %r2.1 = getelementptr inbounds %complex, ptr %r.i32, i32 0, i32 2, i32 1
  %r2.i = getelementptr inbounds %complex, ptr %r.i32, i32 0, i32 2, i32 %i
  %r2.1i = getelementptr inbounds i32, ptr %r2.1, i32 %i
  load i32, ptr %a2.0
  load i32, ptr %a1
  load i32, ptr %r2.0
  load i32, ptr %r2.1
  load i32, ptr %r2.i
  load i32, ptr %r2.1i
  ret void
}

; CHECK-LABEL: Function: complex2:
; CHECK-DAG: NoAlias:  i32* %alloca, i32* %p120
; CHECK-DAG: MayAlias: i32* %alloca, i32* %pi20
; CHECK-DAG: MayAlias: i32* %alloca, i32* %pij1
; CHECK-DAG: MayAlias: i32* %a3, i32* %pij1
%inner = type { i32, i32 }
%outer = type { i32, i32, [10 x %inner] }
declare ptr @rand_outer(ptr %p)
define void @complex2(i32 %i, i32 %j) {
  %alloca = alloca i32, i32 128
  %a3 = getelementptr inbounds i32, ptr %alloca, i32 3
  %random = call ptr @rand_outer(ptr %alloca)
  %p120 = getelementptr inbounds %outer, ptr %random, i32 1, i32 2, i32 2, i32 0
  %pi20 = getelementptr inbounds %outer, ptr %random, i32 %i, i32 2, i32 2, i32 0
  %pij1 = getelementptr inbounds %outer, ptr %random, i32 %i, i32 2, i32 %j, i32 1
  load i32, ptr %alloca
  load i32, ptr %a3
  load i32, ptr %p120
  load i32, ptr %pi20
  load i32, ptr %pij1
  ret void
}

; CHECK-LABEL: Function: pointer_offset:
; CHECK-DAG: MayAlias: ptr* %add.ptr, ptr* %x
; CHECK-DAG: MayAlias: ptr* %add.ptr, ptr* %q2
%struct.X = type { ptr, ptr }
define i32 @pointer_offset(i32 signext %i, i32 signext %j, i32 zeroext %off) {
entry:
  %i.addr = alloca i32
  %j.addr = alloca i32
  %x = alloca %struct.X
  store i32 %i, ptr %i.addr
  store i32 %j, ptr %j.addr
  store ptr %i.addr, ptr %x
  %q2 = getelementptr inbounds %struct.X, ptr %x, i32 0, i32 1
  store ptr %j.addr, ptr %q2
  %add.ptr = getelementptr inbounds ptr, ptr %q2, i32 %off
  %0 = load ptr, ptr %add.ptr
  %1 = load i32, ptr %0
  ret i32 %1
}

; CHECK-LABEL: Function: one_size_unknown:
; CHECK: NoModRef:  Ptr: i8* %p.minus1	<->  call void @llvm.memset.p0.i32(ptr %p, i8 0, i32 %size, i1 false)
define void @one_size_unknown(ptr %p, i32 %size) {
  %p.minus1 = getelementptr inbounds i8, ptr %p, i32 -1
  call void @llvm.memset.p0.i32(ptr %p, i8 0, i32 %size, i1 false)
  load i8, ptr %p.minus1
  ret void
}


; If part of the addressing is done with non-inbounds GEPs, we can't use
; properties implied by the last gep w/the whole offset. In this case,
; %random = %alloc - 4 bytes is well defined, and results in %step == %alloca,
; leaving %p as an entirely inbounds gep pointing inside %alloca
; CHECK-LABEL: Function: all_inbounds:
; CHECK: MayAlias: i32* %alloca, i8* %random
; CHECK: MayAlias:  i32* %alloca, i8* %p1
define void @all_inbounds() {
  %alloca = alloca i32, i32 4
  %random = call ptr @random.i8(ptr %alloca)
  %step = getelementptr i8, ptr %random, i8 4
  %p1 = getelementptr inbounds i8, ptr %step, i8 2
  load i32, ptr %alloca
  load i8, ptr %random
  load i8, ptr %p1
  ret void
}


; For all values of %x, %random and %p1 can't alias because %random would
; have to be out of bounds (and thus a contradiction) for them to be equal.
; CHECK-LABEL: Function: common_factor:
; CHECK: NoAlias:  i32* %p0, i32* %p1
define void @common_factor(i32 %x) {
  %alloca = alloca i32, i32 4
  %random = call ptr @random.i8(ptr %alloca)
  %p0 = getelementptr inbounds i32, ptr %alloca, i32 %x
  %step = getelementptr inbounds i8, ptr %random, i8 4
  %p1 = getelementptr inbounds i32, ptr %step, i32 %x
  load i32, ptr %p0
  load i32, ptr %p1
  ret void
}


declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)
