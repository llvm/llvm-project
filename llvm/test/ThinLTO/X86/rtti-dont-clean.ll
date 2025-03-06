; RUN: opt -thinlto-bc -o %t.o %s
;
; RUN: llvm-lto2 run %t.o -o %t2 -save-temps \
; RUN:           -r=%t.o,main,px \
; RUN:           -r=%t.o,_ZTSvt,p \
; RUN:           -r=%t.o,_ZTIvt,p \
; RUN:           -r=%t.o,_ZTVvt,p \
; RUN:           -r=%t.o,_ZTSvt1,p \
; RUN:           -r=%t.o,_ZTIvt1,p \
; RUN:           -r=%t.o,_ZTVvt1,p \
; RUN:           -r=%t.o,use,p \
; RUN:           -r=%t.o,dyncast,p \
; RUN:           -r=%t.o,__dynamic_cast,px \
; RUN:           -whole-program-visibility
; RUN: llvm-dis %t2.1.1.promote.bc -o - | FileCheck %s
;

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTSvt = external constant ptr
@_ZTIvt = weak_odr constant { ptr } { ptr @_ZTSvt }

@_ZTSvt1 = external constant ptr
@_ZTIvt1 = weak_odr constant { ptr } { ptr @_ZTSvt1 }

%vtTy = type { [3 x ptr] }

; CHECK: @_ZTVvt = weak_odr constant %vtTy { [3 x ptr] [ptr null, ptr @_ZTIvt, ptr @vf] }, !type !0, !vcall_visitbiliy !1
@_ZTVvt = weak_odr constant %vtTy { [3 x ptr] [ptr null, ptr @_ZTIvt, ptr @vf] }, !type !0, !vcall_visitbiliy !1
; CHECK: @_ZTVvt1 = weak_odr constant %vtTy { [3 x ptr] [ptr null, ptr @_ZTIvt1, ptr @vf] }, !type !2, !vcall_visitbiliy !1
@_ZTVvt1 = weak_odr constant %vtTy { [3 x ptr] [ptr null, ptr @_ZTIvt1, ptr @vf] }, !type !2, !vcall_visitbiliy !1

define internal void @vf() {
  ret void
}

define ptr @use(ptr %p) {
  %vtable = load ptr, ptr %p
  %x = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTSvt")
  call void @llvm.assume(i1 %x)
  %rttiptr = getelementptr i8, ptr %vtable, i64 -16
  %rtti = load ptr, ptr %rttiptr
  ret ptr %rtti
}

define ptr @dyncast(ptr %p) {
  %r = call ptr @__dynamic_cast(ptr %p, ptr @_ZTIvt1, ptr @_ZTIvt1, i64 0)
  ret ptr %r
}

; Make symbol _ZTVvt and _ZTVvt1 alive.
define void @main() {
  %vfunc = load ptr, ptr @_ZTVvt
  %vfunc1 = load ptr, ptr @_ZTVvt1
  ret void
}

declare ptr @__dynamic_cast(ptr, ptr, ptr, i64)
declare void @llvm.assume(i1)
declare i1 @llvm.type.test(ptr %ptr, metadata %type) nounwind readnone
!0 = !{i32 16, !"_ZTSvt"}
!1 = !{i64 1}
!2 = !{i32 16, !"_ZTSvt1"}
