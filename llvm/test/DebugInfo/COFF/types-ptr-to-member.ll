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

@"\01?pmd_a@@3PEQA@@HEQ1@" = global i32 -1, align 8, !dbg !0
@"\01?pmd_b@@3PEQC@@HEQ1@" = global i32 -1, align 8, !dbg !6
@"\01?pmd_c@@3PEQD@@HEQ1@" = global { i32, i32 } { i32 0, i32 -1 }, align 8, !dbg !21
@"\01?pmd_d@@3PEQE@@HEQ1@" = global { i32, i32, i32 } { i32 0, i32 0, i32 -1 }, align 8, !dbg !25
@"\01?pmf_a@@3P8A@@EAAXXZEQ1@" = global i8* null, align 8, !dbg !29
@"\01?pmf_b@@3P8C@@EAAXXZEQ1@" = global { i8*, i32 } zeroinitializer, align 8, !dbg !35
@"\01?pmf_c@@3P8D@@EAAXXZEQ1@" = global { i8*, i32, i32 } zeroinitializer, align 8, !dbg !41
@"\01?pmf_d@@3P8E@@EAAXXZEQ1@" = global { i8*, i32, i32, i32 } zeroinitializer, align 8, !dbg !47
@"\01?ppmd@@3PEAPEQIncomplete@@HEA" = global %0* null, align 8, !dbg !53
@"\01?ppmf@@3PEAP8Incomplete@@EAAXXZEA" = global %1* null, align 8, !dbg !58

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!66, !67, !68}
!llvm.ident = !{!69}

!0 = distinct !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "pmd_a", linkageName: "\01?pmd_a@@3PEQA@@HEQ1@", scope: !2, file: !3, line: 6, type: !65, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 273036) (llvm/trunk 273053)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0, !6, !21, !25, !29, !35, !41, !47, !53, !58}
!6 = distinct !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "pmd_b", linkageName: "\01?pmd_b@@3PEQC@@HEQ1@", scope: !2, file: !3, line: 7, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !9, size: 32, flags: DIFlagMultipleInheritance, extraData: !10)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 3, size: 96, align: 32, elements: !11, identifier: ".?AUC@@")
!11 = !{!12, !16, !20}
!12 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 32, align: 32, elements: !14, identifier: ".?AUA@@")
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !3, line: 1, baseType: !9, size: 32, align: 32)
!16 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !10, baseType: !17, offset: 32)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !3, line: 2, size: 32, align: 32, elements: !18, identifier: ".?AUB@@")
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !17, file: !3, line: 2, baseType: !9, size: 32, align: 32)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !10, file: !3, line: 3, baseType: !9, size: 32, align: 32, offset: 64)
!21 = distinct !DIGlobalVariableExpression(var: !22)
!22 = !DIGlobalVariable(name: "pmd_c", linkageName: "\01?pmd_c@@3PEQD@@HEQ1@", scope: !2, file: !3, line: 8, type: !23, isLocal: false, isDefinition: true)
!23 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !9, size: 64, flags: DIFlagVirtualInheritance, extraData: !24)
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !3, line: 4, size: 256, align: 64, flags: DIFlagFwdDecl, identifier: ".?AUD@@")
!25 = distinct !DIGlobalVariableExpression(var: !26)
!26 = !DIGlobalVariable(name: "pmd_d", linkageName: "\01?pmd_d@@3PEQE@@HEQ1@", scope: !2, file: !3, line: 9, type: !27, isLocal: false, isDefinition: true)
!27 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !9, size: 96, extraData: !28)
!28 = !DICompositeType(tag: DW_TAG_structure_type, name: "E", file: !3, line: 5, flags: DIFlagFwdDecl, identifier: ".?AUE@@")
!29 = distinct !DIGlobalVariableExpression(var: !30)
!30 = !DIGlobalVariable(name: "pmf_a", linkageName: "\01?pmf_a@@3P8A@@EAAXXZEQ1@", scope: !2, file: !3, line: 10, type: !31, isLocal: false, isDefinition: true)
!31 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !32, size: 64, flags: DIFlagSingleInheritance, extraData: !13)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!35 = distinct !DIGlobalVariableExpression(var: !36)
!36 = !DIGlobalVariable(name: "pmf_b", linkageName: "\01?pmf_b@@3P8C@@EAAXXZEQ1@", scope: !2, file: !3, line: 11, type: !37, isLocal: false, isDefinition: true)
!37 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !38, size: 128, flags: DIFlagMultipleInheritance, extraData: !10)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!41 = distinct !DIGlobalVariableExpression(var: !42)
!42 = !DIGlobalVariable(name: "pmf_c", linkageName: "\01?pmf_c@@3P8D@@EAAXXZEQ1@", scope: !2, file: !3, line: 12, type: !43, isLocal: false, isDefinition: true)
!43 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !44, size: 128, flags: DIFlagVirtualInheritance, extraData: !24)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !46}
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = distinct !DIGlobalVariableExpression(var: !48)
!48 = !DIGlobalVariable(name: "pmf_d", linkageName: "\01?pmf_d@@3P8E@@EAAXXZEQ1@", scope: !2, file: !3, line: 13, type: !49, isLocal: false, isDefinition: true)
!49 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !50, size: 192, extraData: !28)
!50 = !DISubroutineType(types: !51)
!51 = !{null, !52}
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = distinct !DIGlobalVariableExpression(var: !54)
!54 = !DIGlobalVariable(name: "ppmd", linkageName: "\01?ppmd@@3PEAPEQIncomplete@@HEA", scope: !2, file: !3, line: 15, type: !55, isLocal: false, isDefinition: true)
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !56, size: 64, align: 64)
!56 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !9, extraData: !57)
!57 = !DICompositeType(tag: DW_TAG_structure_type, name: "Incomplete", file: !3, line: 14, flags: DIFlagFwdDecl, identifier: ".?AUIncomplete@@")
!58 = distinct !DIGlobalVariableExpression(var: !59)
!59 = !DIGlobalVariable(name: "ppmf", linkageName: "\01?ppmf@@3PEAP8Incomplete@@EAAXXZEA", scope: !2, file: !3, line: 16, type: !60, isLocal: false, isDefinition: true)
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !61, size: 64, align: 64)
!61 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !62, extraData: !57)
!62 = !DISubroutineType(types: !63)
!63 = !{null, !64}
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !57, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!65 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !9, size: 32, flags: DIFlagSingleInheritance, extraData: !13)
!66 = !{i32 2, !"CodeView", i32 1}
!67 = !{i32 2, !"Debug Info Version", i32 3}
!68 = !{i32 1, !"PIC Level", i32 2}
!69 = !{!"clang version 3.9.0 (trunk 273036) (llvm/trunk 273053)"}

