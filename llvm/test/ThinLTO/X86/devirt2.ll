; REQUIRES: x86-registered-target

; Test devirtualization requiring promotion of local targets.

; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t1.o %s
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t2.o %p/Inputs/devirt2.ll

; Check that we have module flag showing splitting enabled, and that we don't
; generate summary information needed for index-based WPD.
; RUN: llvm-modextract -b -n=0 %t2.o -o %t2.o.0
; RUN: llvm-dis -o - %t2.o.0 | FileCheck %s --check-prefix=ENABLESPLITFLAG --implicit-check-not=vTableFuncs --implicit-check-not=typeidCompatibleVTable
; RUN: llvm-modextract -b -n=1 %t2.o -o %t2.o.1
; RUN: llvm-dis -o - %t2.o.1 | FileCheck %s --check-prefix=ENABLESPLITFLAG --implicit-check-not=vTableFuncs --implicit-check-not=typeidCompatibleVTable
; ENABLESPLITFLAG: !{i32 1, !"EnableSplitLTOUnit", i32 1}

; Generate unsplit module with summary for ThinLTO index-based WPD.
; Force generation of the bitcode index so that we also test lazy metadata
; loader handling of the type metadata.
; RUN: opt -bitcode-mdindex-threshold=0 -thinlto-bc -o %t3.o %s
; RUN: opt -bitcode-mdindex-threshold=0 -thinlto-bc -o %t4.o %p/Inputs/devirt2.ll

; Check that we don't have module flag when splitting not enabled for ThinLTO,
; and that we generate summary information needed for index-based WPD.
; RUN: llvm-dis -o - %t4.o | FileCheck %s --check-prefix=NOENABLESPLITFLAG
; NOENABLESPLITFLAG-DAG: !{i32 1, !"EnableSplitLTOUnit", i32 0}
; NOENABLESPLITFLAG-DAG: [[An:\^[0-9]+]] = gv: (name: "_ZN1A1nEi"
; NOENABLESPLITFLAG-DAG: [[Bf:\^[0-9]+]] = gv: (name: "_ZN1B1fEi"
; NOENABLESPLITFLAG-DAG: [[Cf:\^[0-9]+]] = gv: (name: "_ZN1C1fEi"
; NOENABLESPLITFLAG-DAG: [[Dm:\^[0-9]+]] = gv: (name: "_ZN1D1mEi"
; NOENABLESPLITFLAG-DAG: [[B:\^[0-9]+]] = gv: (name: "_ZTV1B", {{.*}} vTableFuncs: ((virtFunc: [[Bf]], offset: 16), (virtFunc: [[An]], offset: 24)), refs: ([[Bf]], [[An]])
; NOENABLESPLITFLAG-DAG: [[C:\^[0-9]+]] = gv: (name: "_ZTV1C", {{.*}} vTableFuncs: ((virtFunc: [[Cf]], offset: 16), (virtFunc: [[An]], offset: 24)), refs: ([[An]], [[Cf]])
; NOENABLESPLITFLAG-DAG: [[D:\^[0-9]+]] = gv: (name: "_ZTV1D", {{.*}} vTableFuncs: ((virtFunc: [[Dm]], offset: 16)), refs: ([[Dm]])
; NOENABLESPLITFLAG-DAG: typeidCompatibleVTable: (name: "_ZTS1A", summary: ((offset: 16, [[B]]), (offset: 16, [[C]])))
; NOENABLESPLITFLAG-DAG: typeidCompatibleVTable: (name: "_ZTS1B", summary: ((offset: 16, [[B]])))
; NOENABLESPLITFLAG-DAG: typeidCompatibleVTable: (name: "_ZTS1C", summary: ((offset: 16, [[C]])))
; NOENABLESPLITFLAG-DAG: typeidCompatibleVTable: (name: "_ZTS1D", summary: ((offset: 16, [[D]])))

; Index based WPD
; RUN: llvm-lto2 run %t3.o %t4.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t5 \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t3.o,_ZTV1B, \
; RUN:   -r=%t3.o,_ZTV1C, \
; RUN:   -r=%t3.o,_ZTV1D, \
; RUN:   -r=%t3.o,_ZN1D1mEi, \
; RUN:   -r=%t3.o,test2, \
; RUN:   -r=%t4.o,_ZN1B1fEi,p \
; RUN:   -r=%t4.o,_ZN1C1fEi,p \
; RUN:   -r=%t4.o,_ZN1D1mEi,p \
; RUN:   -r=%t4.o,test2,px \
; RUN:   -r=%t4.o,_ZTV1B,px \
; RUN:   -r=%t4.o,_ZTV1C,px \
; RUN:   -r=%t4.o,_ZTV1D,px \
; RUN:   -r=%t4.o,_ZTV1E,px 2>&1 | FileCheck %s --check-prefix=REMARK --check-prefix=PRINT
; RUN: llvm-dis %t5.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1
; RUN: llvm-dis %t5.2.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR2
; RUN: llvm-nm %t5.1 | FileCheck %s --check-prefix=NM-INDEX1
; RUN: llvm-nm %t5.2 | FileCheck %s --check-prefix=NM-INDEX2

; NM-INDEX1-DAG: U _ZN1A1nEi.llvm.
; NM-INDEX1-DAG: U _ZN1E1mEi.llvm.
; NM-INDEX1-DAG: U _ZN1D1mEi

; NM-INDEX2-DAG: T _ZN1A1nEi.llvm.
; NM-INDEX2-DAG: T _ZN1E1mEi.llvm.
; NM-INDEX2-DAG: W _ZN1D1mEi
; NM-INDEX2-DAG: t _ZN1B1fEi
; NM-INDEX2-DAG: t _ZN1C1fEi

; Index based WPD, distributed backends
; RUN: llvm-lto2 run %t3.o %t4.o -save-temps \
; RUN:   -whole-program-visibility \
; RUN:   -thinlto-distributed-indexes -wholeprogramdevirt-print-index-based \
; RUN:   -o %t5 \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t3.o,_ZTV1B, \
; RUN:   -r=%t3.o,_ZTV1C, \
; RUN:   -r=%t3.o,_ZTV1D, \
; RUN:   -r=%t3.o,_ZN1D1mEi, \
; RUN:   -r=%t3.o,test2, \
; RUN:   -r=%t4.o,_ZN1B1fEi,p \
; RUN:   -r=%t4.o,_ZN1C1fEi,p \
; RUN:   -r=%t4.o,_ZN1D1mEi,p \
; RUN:   -r=%t4.o,test2,px \
; RUN:   -r=%t4.o,_ZTV1B,px \
; RUN:   -r=%t4.o,_ZTV1C,px \
; RUN:   -r=%t4.o,_ZTV1D,px \
; RUN:   -r=%t4.o,_ZTV1E,px 2>&1 | FileCheck %s --check-prefix=PRINT

; PRINT-DAG: Devirtualized call to {{.*}} (_ZN1A1nEi)
; PRINT-DAG: Devirtualized call to {{.*}} (_ZN1E1mEi)
; PRINT-DAG: Devirtualized call to {{.*}} (_ZN1D1mEi)

; New PM
; RUN: llvm-lto2 run %t1.o %t2.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t5 \
; RUN:   -r=%t1.o,test,px \
; RUN:   -r=%t1.o,_ZTV1B, \
; RUN:   -r=%t1.o,_ZTV1C, \
; RUN:   -r=%t1.o,_ZTV1D, \
; RUN:   -r=%t1.o,_ZTV1D, \
; RUN:   -r=%t1.o,_ZN1D1mEi, \
; RUN:   -r=%t1.o,_ZN1D1mEi, \
; RUN:   -r=%t1.o,test2, \
; RUN:   -r=%t2.o,_ZN1A1nEi,p \
; RUN:   -r=%t2.o,_ZN1B1fEi,p \
; RUN:   -r=%t2.o,_ZN1C1fEi,p \
; RUN:   -r=%t2.o,_ZN1D1mEi,p \
; RUN:   -r=%t2.o,_ZN1E1mEi,p \
; RUN:   -r=%t2.o,_ZTV1B, \
; RUN:   -r=%t2.o,_ZTV1C, \
; RUN:   -r=%t2.o,_ZTV1D, \
; RUN:   -r=%t2.o,_ZTV1E, \
; RUN:   -r=%t2.o,test2,px \
; RUN:   -r=%t2.o,_ZN1A1nEi, \
; RUN:   -r=%t2.o,_ZN1B1fEi, \
; RUN:   -r=%t2.o,_ZN1C1fEi, \
; RUN:   -r=%t2.o,_ZN1D1mEi, \
; RUN:   -r=%t2.o,_ZN1E1mEi, \
; RUN:   -r=%t2.o,_ZTV1B,px \
; RUN:   -r=%t2.o,_ZTV1C,px \
; RUN:   -r=%t2.o,_ZTV1D,px \
; RUN:   -r=%t2.o,_ZTV1E,px 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t5.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1
; RUN: llvm-dis %t5.2.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR2
; RUN: llvm-nm %t5.1 | FileCheck %s --check-prefix=NM-HYBRID1
; RUN: llvm-nm %t5.2 | FileCheck %s --check-prefix=NM-HYBRID2

; NM-HYBRID1-DAG: U _ZN1A1nEi.{{[0-9a-f]*}}
; NM-HYBRID1-DAG: U _ZN1E1mEi.{{[0-9a-f]*}}
; NM-HYBRID1-DAG: U _ZN1D1mEi

; NM-HYBRID2-DAG: T _ZN1A1nEi.{{[0-9a-f]*}}
; NM-HYBRID2-DAG: T _ZN1E1mEi.{{[0-9a-f]*}}
; NM-HYBRID2-DAG: W _ZN1D1mEi
; NM-HYBRID2-DAG: T _ZN1B1fEi
; NM-HYBRID2-DAG: T _ZN1C1fEi

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi
; REMARK-DAG: single-impl: devirtualized a call to _ZN1D1mEi
; We should devirt call to _ZN1E1mEi once in importing module and once
; in original (exporting) module.
; REMARK-DAG: single-impl: devirtualized a call to _ZN1E1mEi
; REMARK-DAG: single-impl: devirtualized a call to _ZN1E1mEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { i32 (...)** }
%struct.E = type { i32 (...)** }

@_ZTV1B = external constant [4 x i8*]
@_ZTV1C = external constant [4 x i8*]
;@_ZTV1D = external constant [3 x i8*]
@_ZTV1D = linkonce_odr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.D*, i32)* @_ZN1D1mEi to i8*)] }, !type !3

define linkonce_odr i32 @_ZN1D1mEi(%struct.D* %this, i32 %a) #0 {
   ret i32 0
}

; CHECK-IR1-LABEL: define i32 @test
define i32 @test(%struct.A* %obj, %struct.D* %obj2, %struct.E* %obj3, i32 %a) {
entry:
  %0 = bitcast %struct.A* %obj to i8***
  %vtable = load i8**, i8*** %0
  %1 = bitcast i8** %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8*, i8** %vtable, i32 1
  %2 = bitcast i8** %fptrptr to i32 (%struct.A*, i32)**
  %fptr1 = load i32 (%struct.A*, i32)*, i32 (%struct.A*, i32)** %2, align 8

  ; Check that the call was devirtualized. Ignore extra character before
  ; symbol name which would happen if it was promoted during module
  ; splitting for hybrid WPD.
  ; CHECK-IR1: %call = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)

  %3 = bitcast i8** %vtable to i32 (%struct.A*, i32)**
  %fptr22 = load i32 (%struct.A*, i32)*, i32 (%struct.A*, i32)** %3, align 8

  ; We still have to call it as virtual.
  ; CHECK-IR1: %call3 = tail call i32 %fptr22
  %call3 = tail call i32 %fptr22(%struct.A* nonnull %obj, i32 %call)

  %4 = bitcast %struct.D* %obj2 to i8***
  %vtable2 = load i8**, i8*** %4
  %5 = bitcast i8** %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %5, metadata !"_ZTS1D")
  call void @llvm.assume(i1 %p2)

  %6 = bitcast i8** %vtable2 to i32 (%struct.D*, i32)**
  %fptr33 = load i32 (%struct.D*, i32)*, i32 (%struct.D*, i32)** %6, align 8

  ; Check that the call was devirtualized.
  ; CHECK-IR1: %call4 = tail call i32 @_ZN1D1mEi
  %call4 = tail call i32 %fptr33(%struct.D* nonnull %obj2, i32 %call3)

  %call5 = tail call i32 @test2(%struct.E* nonnull %obj3, i32 %call4)
  ret i32 %call5
}
; CHECK-IR1-LABEL: ret i32
; CHECK-IR1-LABEL: }

; CHECK-IR2: define i32 @test2
; CHECK-IR2-NEXT: entry:
; Check that the call was devirtualized. Ignore extra character before
; symbol name which would happen if it was promoted during module
; splitting for hybrid WPD.
; CHECK-IR2-NEXT:   %call4 = tail call i32 @{{.*}}_ZN1E1mEi

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)
declare i32 @test2(%struct.E* %obj, i32 %a)

attributes #0 = { noinline optnone }

!3 = !{i64 16, !"_ZTS1D"}
