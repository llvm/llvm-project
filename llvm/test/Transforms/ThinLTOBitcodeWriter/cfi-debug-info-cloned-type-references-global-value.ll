; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck %s

; Crash test for CloneModule when there's a retained DICompositeType that
; transitively references a global value.

; CHECK: declare !type !{{[0-9]+}} !type !{{[0-9]+}} void @_Z1gIM1iKFivEEvT_(i64, i64)
; CHECK: !llvm.dbg.cu
; CHECK-DAG: distinct !DICompositeType({{.*}}, identifier: "_ZTS1oI1iiXadL_ZNKS0_5m_fn1EvEEE"
; CHECK-DAG: distinct !DICompositeType({{.*}}, identifier: "_ZTS1i"
; CHECK-DAG: !{i32 4, !"CFI Canonical Jump Tables", i32 0}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZN1i1pE = dso_local constant [1 x i8] zeroinitializer, align 1
@_ZNK1i5m_fn1Ev = external global i32

declare !type !17 !type !18 void @_Z1gIM1iKFivEEvT_(i64, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0 (git@github.com:llvm/llvm-project.git 51bf4c0e6d4cbc6dfa57857fc78003413cbeb17f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "o<i, int, &i::m_fn1>", file: !5, line: 22, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !2, templateParams: !6, identifier: "_ZTS1oI1iiXadL_ZNKS0_5m_fn1EvEEE")
!5 = !DIFile(filename: "t.ii", directory: "/tmp")
!6 = !{!7}
!7 = !DITemplateValueParameter(type: !8, value: i64 ptrtoint (i32* @_ZNK1i5m_fn1Ev to i64))
!8 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !9, size: 128, extraData: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "i", file: !5, line: 13, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !2, identifier: "_ZTS1i")
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!16 = !{i64 ptrtoint (i32* @_ZNK1i5m_fn1Ev to i64)}
!17 = !{i64 0, !"_ZTSFvM1iKFivEE"}
!18 = !{i64 0, !"_ZTSFvM1iKFivEE.generalized"}
