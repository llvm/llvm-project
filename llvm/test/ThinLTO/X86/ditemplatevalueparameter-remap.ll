; https://github.com/llvm/llvm-project/pull/110064
; This test case checks if thinLTO correctly links metadata values in a specific
; situation. Assume we are linking module B into module A, where an extern
; function used in A is defined in B, but the function body has a
; DITemplateValueParameter referring to another function back in A. The
; compiler must check this other function is actually coming from A, thus
; already materialized and does not require remapping. The IR here is modified
; from the following source code.
;
; // A.h
; template <void (*Func)()>
; struct S {
;   void Impl() {
;     Func();
;   }
; };
;
; void func1();
;
; // A.cpp
; #include "A.h"
; __attribute__((weak)) void func1() {}
; extern void thinlto1();
; void bar() {
;   S<func1> s; // Force instantiation of S<func1> in this compilation unit.
;   s.Impl();
;   thinlto1();
; }
;
; // B.cpp
; #include "A.h"
; void thinlto1() {
;   S<func1> s;
; }
;
; RUN: opt -module-summary -o %t1.bc %s
; RUN: opt -module-summary -o %t2.bc %S/Inputs/ditemplatevalueparameter-remap.ll
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t3 -save-temps \
; RUN:   -r=%t1.bc,_Z5func1v,p    \
; RUN:   -r=%t1.bc,_Z3bazv,px     \
; RUN:   -r=%t1.bc,_Z8thinlto1v,x \
; RUN:   -r=%t1.bc,_Z3barv,px     \
; RUN:   -r=%t2.bc,_Z8thinlto1v,px
; RUN: llvm-dis %t3.1.4.opt.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z5func1v = comdat any

define linkonce_odr void @_Z5func1v() unnamed_addr !dbg !10 {
  ret void
}

; Dummy function to use _Z5func1v so that it is not treated as dead symbol.
define void @_Z3bazv() {
  tail call void @_Z5func1v()
  ret void
}

declare void @_Z8thinlto1v() unnamed_addr

; Check _Z8thinlto1v is inlined after thinLTO.
; CHECK: void @_Z3barv()
; CHECK-NOT: @_Z8thinlto1v()
; CHECK-NEXT: ret void
define void @_Z3barv() unnamed_addr !dbg !14 {
  tail call void @_Z8thinlto1v(), !dbg !25
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "A.cpp", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!10 = distinct !DISubprogram(name: "func1", linkageName: "_Z5func1v", scope: !11, file: !11, line: 6, type: !12, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DIFile(filename: "a.h", directory: ".")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !11, file: !11, line: 15, type: !12, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!16 = !{!17}
!17 = !DILocalVariable(name: "s", scope: !14, file: !11, line: 10, type: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<&func1>", file: !11, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !19, templateParams: !20, identifier: "_ZTS1SIXadL_Z5func1vEEE")
!19 = !{}
!20 = !{!21}
!21 = !DITemplateValueParameter(name: "Func", type: !22, value: ptr @_Z5func1v)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!25 = !DILocation(line: 16, column: 5, scope: !14)
