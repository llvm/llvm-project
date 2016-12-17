; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A { int a; };
; struct B { int b; };
; struct C : A, B { int c; };
; struct D : virtual C { int d; };
; struct E;
; int A::*pmd_a;
; int C::*pmd_b;
; int D::*pmd_c;
; int E::*pmd_d;
; void (A::*pmf_a)();
; void (C::*pmf_b)();
; void (D::*pmf_c)();
; void (E::*pmf_d)();
; struct Incomplete;
; int Incomplete::**ppmd;
; void (Incomplete::**ppmf)();
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [

; Pointer to data member

; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PointerAttributes: 0x804C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 4
; CHECK:     ClassType: A
; CHECK:     Representation: SingleInheritanceData (0x1)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PointerAttributes: 0x804C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 4
; CHECK:     ClassType: C
; CHECK:     Representation: MultipleInheritanceData (0x2)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PointerAttributes: 0x1004C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:     ClassType: D
; CHECK:     Representation: VirtualInheritanceData (0x3)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PointerAttributes: 0x1804C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 12
; CHECK:     ClassType: E
; CHECK:     Representation: GeneralData (0x4)
; CHECK:   }

; Pointer to member function

; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: void A::()
; CHECK:     PointerAttributes: 0x1006C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:     ClassType: A
; CHECK:     Representation: SingleInheritanceFunction (0x5)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: void C::()
; CHECK:     PointerAttributes: 0x2006C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 16
; CHECK:     ClassType: C
; CHECK:     Representation: MultipleInheritanceFunction (0x6)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: void D::()
; CHECK:     PointerAttributes: 0x2006C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 16
; CHECK:     ClassType: D
; CHECK:     Representation: VirtualInheritanceFunction (0x7)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: void E::()
; CHECK:     PointerAttributes: 0x3006C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 24
; CHECK:     ClassType: E
; CHECK:     Representation: GeneralFunction (0x8)
; CHECK:   }

; Unknown inheritance model MPT
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int
; CHECK:     PointerAttributes: 0x4C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 0
; CHECK:     ClassType: Incomplete
; CHECK:     Representation: Unknown (0x0)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: void Incomplete::()
; CHECK:     PointerAttributes: 0x6C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 0
; CHECK:     ClassType: Incomplete
; CHECK:     Representation: Unknown (0x0)
; CHECK:   }
; CHECK:   Pointer ({{.*}}) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)

; CHECK: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%0 = type opaque
%1 = type opaque

@"\01?pmd_a@@3PEQA@@HEQ1@" = global i32 -1, align 8, !dbg !4
@"\01?pmd_b@@3PEQC@@HEQ1@" = global i32 -1, align 8, !dbg !10
@"\01?pmd_c@@3PEQD@@HEQ1@" = global { i32, i32 } { i32 0, i32 -1 }, align 8, !dbg !20
@"\01?pmd_d@@3PEQE@@HEQ1@" = global { i32, i32, i32 } { i32 0, i32 0, i32 -1 }, align 8, !dbg !23
@"\01?pmf_a@@3P8A@@EAAXXZEQ1@" = global i8* null, align 8, !dbg !26
@"\01?pmf_b@@3P8C@@EAAXXZEQ1@" = global { i8*, i32 } zeroinitializer, align 8, !dbg !31
@"\01?pmf_c@@3P8D@@EAAXXZEQ1@" = global { i8*, i32, i32 } zeroinitializer, align 8, !dbg !36
@"\01?pmf_d@@3P8E@@EAAXXZEQ1@" = global { i8*, i32, i32, i32 } zeroinitializer, align 8, !dbg !41
@"\01?ppmd@@3PEAPEQIncomplete@@HEA" = global %0* null, align 8, !dbg !46
@"\01?ppmf@@3PEAP8Incomplete@@EAAXXZEA" = global %1* null, align 8, !dbg !50

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!56, !57, !58}
!llvm.ident = !{!59}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 273036) (llvm/trunk 273053)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4, !10, !20, !23, !26, !31, !36, !41, !46, !50}
!4 = distinct !DIGlobalVariable(name: "pmd_a", linkageName: "\01?pmd_a@@3PEQA@@HEQ1@", scope: !0, file: !1, line: 6, type: !5, isLocal: false, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 32, flags: DIFlagSingleInheritance, extraData: !7)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 32, align: 32, elements: !8, identifier: ".?AUA@@")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !1, line: 1, baseType: !6, size: 32, align: 32)
!10 = distinct !DIGlobalVariable(name: "pmd_b", linkageName: "\01?pmd_b@@3PEQC@@HEQ1@", scope: !0, file: !1, line: 7, type: !11, isLocal: false, isDefinition: true)
!11 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 32, flags: DIFlagMultipleInheritance, extraData: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 3, size: 96, align: 32, elements: !13, identifier: ".?AUC@@")
!13 = !{!14, !15, !19}
!14 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !12, baseType: !7)
!15 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !12, baseType: !16, offset: 32)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 2, size: 32, align: 32, elements: !17, identifier: ".?AUB@@")
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !16, file: !1, line: 2, baseType: !6, size: 32, align: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !12, file: !1, line: 3, baseType: !6, size: 32, align: 32, offset: 64)
!20 = distinct !DIGlobalVariable(name: "pmd_c", linkageName: "\01?pmd_c@@3PEQD@@HEQ1@", scope: !0, file: !1, line: 8, type: !21, isLocal: false, isDefinition: true)
!21 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 64, flags: DIFlagVirtualInheritance, extraData: !22)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !1, line: 4, size: 256, align: 64, flags: DIFlagFwdDecl, identifier: ".?AUD@@")
!23 = distinct !DIGlobalVariable(name: "pmd_d", linkageName: "\01?pmd_d@@3PEQE@@HEQ1@", scope: !0, file: !1, line: 9, type: !24, isLocal: false, isDefinition: true)
!24 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, size: 96, extraData: !25)
!25 = !DICompositeType(tag: DW_TAG_structure_type, name: "E", file: !1, line: 5, flags: DIFlagFwdDecl, identifier: ".?AUE@@")
!26 = distinct !DIGlobalVariable(name: "pmf_a", linkageName: "\01?pmf_a@@3P8A@@EAAXXZEQ1@", scope: !0, file: !1, line: 10, type: !27, isLocal: false, isDefinition: true)
!27 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !28, size: 64, flags: DIFlagSingleInheritance, extraData: !7)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = distinct !DIGlobalVariable(name: "pmf_b", linkageName: "\01?pmf_b@@3P8C@@EAAXXZEQ1@", scope: !0, file: !1, line: 11, type: !32, isLocal: false, isDefinition: true)
!32 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !33, size: 128, flags: DIFlagMultipleInheritance, extraData: !12)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = distinct !DIGlobalVariable(name: "pmf_c", linkageName: "\01?pmf_c@@3P8D@@EAAXXZEQ1@", scope: !0, file: !1, line: 12, type: !37, isLocal: false, isDefinition: true)
!37 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !38, size: 128, flags: DIFlagVirtualInheritance, extraData: !22)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!41 = distinct !DIGlobalVariable(name: "pmf_d", linkageName: "\01?pmf_d@@3P8E@@EAAXXZEQ1@", scope: !0, file: !1, line: 13, type: !42, isLocal: false, isDefinition: true)
!42 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !43, size: 192, extraData: !25)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !45}
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!46 = distinct !DIGlobalVariable(name: "ppmd", linkageName: "\01?ppmd@@3PEAPEQIncomplete@@HEA", scope: !0, file: !1, line: 15, type: !47, isLocal: false, isDefinition: true)
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !48, size: 64, align: 64)
!48 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !6, extraData: !49)
!49 = !DICompositeType(tag: DW_TAG_structure_type, name: "Incomplete", file: !1, line: 14, flags: DIFlagFwdDecl, identifier: ".?AUIncomplete@@")
!50 = distinct !DIGlobalVariable(name: "ppmf", linkageName: "\01?ppmf@@3PEAP8Incomplete@@EAAXXZEA", scope: !0, file: !1, line: 16, type: !51, isLocal: false, isDefinition: true)
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !52, size: 64, align: 64)
!52 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !53, extraData: !49)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !55}
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !49, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!56 = !{i32 2, !"CodeView", i32 1}
!57 = !{i32 2, !"Debug Info Version", i32 3}
!58 = !{i32 1, !"PIC Level", i32 2}
!59 = !{!"clang version 3.9.0 (trunk 273036) (llvm/trunk 273053)"}
