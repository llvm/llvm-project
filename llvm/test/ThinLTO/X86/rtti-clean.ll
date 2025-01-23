; RUN: opt -thinlto-bc -o %t.o %s
;
; RUN: llvm-lto2 run %t.o -o %t2 -save-temps \
; RUN:           -r=%t.o,main,px \
; RUN:           -r=%t.o,_ZTSvt,p \
; RUN:           -r=%t.o,_ZTIvt,p \
; RUN:           -r=%t.o,_ZTVvt,p \
; RUN:           -whole-program-visibility
; RUN: llvm-dis %t2.1.1.promote.bc -o - | FileCheck %s
;

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTSvt = external constant ptr
@_ZTIvt = weak_odr constant { ptr } { ptr @_ZTSvt }

%vtTy = type { [3 x ptr] }

; CHECK: @_ZTVvt = weak_odr constant %vtTy { [3 x ptr] [ptr null, ptr null, ptr @vf] }, !type !0, !vcall_visitbiliy !1
@_ZTVvt = weak_odr constant %vtTy { [3 x ptr] [ptr null, ptr @_ZTIvt, ptr @vf] }, !type !0, !vcall_visitbiliy !1

define internal void @vf() {
  ret void
}

define void @main() {
  %vfunc = load ptr, ptr @_ZTVvt
  ret void
}

!0 = !{i32 16, !"_ZTSvt"}
!1 = !{i64 1}
!2 = !{i32 16, !"_ZTSvt1"}
