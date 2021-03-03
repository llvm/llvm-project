; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

declare i32* @random.i32(i32* %ptr)
declare i8* @random.i8(i32* %ptr)

; CHECK-LABEL: Function: arr:
; CHECK-DAG: MayAlias: i32* %alloca, i32* %p0
; CHECK-DAG: NoAlias:  i32* %alloca, i32* %p1
define void @arr() {
  %alloca = alloca i32, i32 4
  %random = call i32* @random.i32(i32* %alloca)
  %p0 = getelementptr inbounds i32, i32* %random, i32 0
  %p1 = getelementptr inbounds i32, i32* %random, i32 1
  ret void
}

; CHECK-LABEL: Function: arg:
; CHECK-DAG: MayAlias: i32* %arg, i32* %p0
; CHECK-DAG: MayAlias: i32* %arg, i32* %p1
define void @arg(i32* %arg) {
  %random = call i32* @random.i32(i32* %arg)
  %p0 = getelementptr inbounds i32, i32* %random, i32 0
  %p1 = getelementptr inbounds i32, i32* %random, i32 1
  ret void
}

@gv = global i32 1
; CHECK-LABEL: Function: global:
; CHECK-DAG: MayAlias: i32* %p0, i32* @gv
; CHECK-DAG: NoAlias:  i32* %p1, i32* @gv
define void @global() {
  %random = call i32* @random.i32(i32* @gv)
  %p0 = getelementptr inbounds i32, i32* %random, i32 0
  %p1 = getelementptr inbounds i32, i32* %random, i32 1
  ret void
}

; CHECK-LABEL: Function: struct:
; CHECK-DAG:  MayAlias: i32* %f0, i32* %p0
; CHECK-DAG:  MayAlias: i32* %f1, i32* %p0
; CHECK-DAG:  NoAlias:  i32* %f0, i32* %p1
; CHECK-DAG:  MayAlias: i32* %f1, i32* %p1
%struct = type { i32, i32, i32 }
define void @struct() {
  %alloca = alloca %struct
  %alloca.i32 = bitcast %struct* %alloca to i32*
  %random = call i32* @random.i32(i32* %alloca.i32)
  %f0 = getelementptr inbounds %struct, %struct* %alloca, i32 0, i32 0
  %f1 = getelementptr inbounds %struct, %struct* %alloca, i32 0, i32 1
  %p0 = getelementptr inbounds i32, i32* %random, i32 0
  %p1 = getelementptr inbounds i32, i32* %random, i32 1
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
  %alloca.i32 = bitcast %complex* %alloca to i32*
  %r.i32 = call i32* @random.i32(i32* %alloca.i32)
  %random = bitcast i32* %r.i32 to %complex*
  %a1 = getelementptr inbounds %complex, %complex* %alloca, i32 0, i32 1
  %a2.0 = getelementptr inbounds %complex, %complex* %alloca, i32 0, i32 2, i32 0
  %r2.0 = getelementptr inbounds %complex, %complex* %random, i32 0, i32 2, i32 0
  %r2.1 = getelementptr inbounds %complex, %complex* %random, i32 0, i32 2, i32 1
  %r2.i = getelementptr inbounds %complex, %complex* %random, i32 0, i32 2, i32 %i
  %r2.1i = getelementptr inbounds i32, i32* %r2.1, i32 %i
  ret void
}

; CHECK-LABEL: Function: complex2:
; CHECK-DAG: NoAlias:  i32* %alloca, i32* %p120
; CHECK-DAG: MayAlias: i32* %alloca, i32* %pi20
; CHECK-DAG: MayAlias: i32* %alloca, i32* %pij1
; CHECK-DAG: MayAlias: i32* %a3, i32* %pij1
%inner = type { i32, i32 }
%outer = type { i32, i32, [10 x %inner] }
declare %outer* @rand_outer(i32* %p)
define void @complex2(i32 %i, i32 %j) {
  %alloca = alloca i32, i32 128
  %a3 = getelementptr inbounds i32, i32* %alloca, i32 3
  %random = call %outer* @rand_outer(i32* %alloca)
  %p120 = getelementptr inbounds %outer, %outer* %random, i32 1, i32 2, i32 2, i32 0
  %pi20 = getelementptr inbounds %outer, %outer* %random, i32 %i, i32 2, i32 2, i32 0
  %pij1 = getelementptr inbounds %outer, %outer* %random, i32 %i, i32 2, i32 %j, i32 1
  ret void
}

; CHECK-LABEL: Function: pointer_offset:
; CHECK-DAG: MayAlias: i32** %add.ptr, i32** %p1
; CHECK-DAG: MayAlias: i32** %add.ptr, i32** %q2
%struct.X = type { i32*, i32* }
define i32 @pointer_offset(i32 signext %i, i32 signext %j, i32 zeroext %off) {
entry:
  %i.addr = alloca i32
  %j.addr = alloca i32
  %x = alloca %struct.X
  store i32 %i, i32* %i.addr
  store i32 %j, i32* %j.addr
  %0 = bitcast %struct.X* %x to i8*
  %p1 = getelementptr inbounds %struct.X, %struct.X* %x, i32 0, i32 0
  store i32* %i.addr, i32** %p1
  %q2 = getelementptr inbounds %struct.X, %struct.X* %x, i32 0, i32 1
  store i32* %j.addr, i32** %q2
  %add.ptr = getelementptr inbounds i32*, i32** %q2, i32 %off
  %1 = load i32*, i32** %add.ptr
  %2 = load i32, i32* %1
  ret i32 %2
}

; CHECK-LABEL: Function: one_size_unknown:
; CHECK: NoModRef:  Ptr: i8* %p.minus1	<->  call void @llvm.memset.p0i8.i32(i8* %p, i8 0, i32 %size, i1 false)
define void @one_size_unknown(i8* %p, i32 %size) {
  %p.minus1 = getelementptr inbounds i8, i8* %p, i32 -1
  call void @llvm.memset.p0i8.i32(i8* %p, i8 0, i32 %size, i1 false)
  ret void
}


; If part of the addressing is done with non-inbounds GEPs, we can't use
; properties implied by the last gep w/the whole offset. In this case,
; %random = %alloc - 4 bytes is well defined, and results in %step == %alloca,
; leaving %p as an entirely inbounds gep pointing inside %alloca
; CHECK-LABEL: Function: all_inbounds:
; CHECK: MayAlias: i32* %alloca, i8* %p0
; CHECK: MayAlias:  i32* %alloca, i8* %p1
define void @all_inbounds() {
  %alloca = alloca i32, i32 4
  %random = call i8* @random.i8(i32* %alloca)
  %p0 = getelementptr inbounds i8, i8* %random, i8 0
  %step = getelementptr i8, i8* %random, i8 4
  %p1 = getelementptr inbounds i8, i8* %step, i8 2
  ret void
}


; For all values of %x, %p0 and %p1 can't alias because %random would
; have to be out of bounds (and thus a contradiction) for them to be equal.
; CHECK-LABEL: Function: common_factor:
; CHECK: MayAlias:  i32* %p0, i32* %p1
; TODO: Missing oppurtunity
define void @common_factor(i32 %x) {
  %alloca = alloca i32, i32 4
  %random = call i8* @random.i8(i32* %alloca)
  %p0 = getelementptr inbounds i32, i32* %alloca, i32 %x
  %step = getelementptr inbounds i8, i8* %random, i8 4
  %step.bitcast = bitcast i8* %step to i32*
  %p1 = getelementptr inbounds i32, i32* %step.bitcast, i32 %x
  ret void
}


declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)
