; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A {
;   void f_default_public();
; private:
;   void f_private();
; protected:
;   void f_protected();
; public:
;   void f_public();
; };
;
; class B {
;  void f_default_private();
; public:
;   void f(float);
;   void f(int);
; };
;
; void foo() {
;   A a;
;   B b;
; }


; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (5)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1001)
; CHECK:     Name: foo
; CHECK:   }
; CHECK:   Struct (0x1003) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x80)
; CHECK:       ForwardReference (0x80)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: A
; CHECK:   }
; CHECK:   Pointer (0x1004) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1003)
; CHECK:     PtrType: Near32 (0xA)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:   }
; CHECK:   MemberFunction (0x1005) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: A (0x1003)
; CHECK:     ThisType: A* const (0x1004)
; CHECK:     CallingConvention: ThisCall (0xB)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList (0x1006) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void A::() (0x1005)
; CHECK:       Name: A::f_default_public
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: void A::() (0x1005)
; CHECK:       Name: A::f_private
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Protected (0x2)
; CHECK:       Type: void A::() (0x1005)
; CHECK:       Name: A::f_protected
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void A::() (0x1005)
; CHECK:       Name: A::f_public
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x1007) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 4
; CHECK:     Properties [ (0x0)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1006)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: A
; CHECK:   }
; CHECK:   StringId (0x1008) {
; CHECK:     TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK:     Id: 0x0
; CHECK:     StringData: /t.cpp
; CHECK:   }
; CHECK:   UdtSourceLine (0x1009) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: A (0x1007)
; CHECK:     SourceFile: /t.cpp (0x1008)
; CHECK:     LineNumber: 1
; CHECK:   }
; CHECK:   Class (0x100A) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x80)
; CHECK:       ForwardReference (0x80)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: B
; CHECK:   }
; CHECK:   Pointer (0x100B) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: B (0x100A)
; CHECK:     PtrType: Near32 (0xA)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 4
; CHECK:   }
; CHECK:   MemberFunction (0x100C) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: B (0x100A)
; CHECK:     ThisType: B* const (0x100B)
; CHECK:     CallingConvention: ThisCall (0xB)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   ArgList (0x100D) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: float (0x40)
; CHECK:     ]
; CHECK:   }
; CHECK:   MemberFunction (0x100E) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: B (0x100A)
; CHECK:     ThisType: B* const (0x100B)
; CHECK:     CallingConvention: ThisCall (0xB)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (float) (0x100D)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   ArgList (0x100F) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: int (0x74)
; CHECK:     ]
; CHECK:   }
; CHECK:   MemberFunction (0x1010) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: B (0x100A)
; CHECK:     ThisType: B* const (0x100B)
; CHECK:     CallingConvention: ThisCall (0xB)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (int) (0x100F)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   MethodOverloadList (0x1011) {
; CHECK:     TypeLeafKind: LF_METHODLIST (0x1206)
; CHECK:     Method [
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void B::(float) (0x100E)
; CHECK:     ]
; CHECK:     Method [
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void B::(int) (0x1010)
; CHECK:     ]
; CHECK:   }
; CHECK:   FieldList (0x1012) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Private (0x1)
; CHECK:       Type: void B::() (0x100C)
; CHECK:       Name: B::f_default_private
; CHECK:     }
; CHECK:     OverloadedMethod {
; CHECK:       MethodCount: 0x2
; CHECK:       MethodListIndex: 0x1011
; CHECK:       Name: B::f
; CHECK:     }
; CHECK:   }
; CHECK:   Class (0x1013) {
; CHECK:     TypeLeafKind: LF_CLASS (0x1504)
; CHECK:     MemberCount: 3
; CHECK:     Properties [ (0x0)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1012)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: B
; CHECK:   }
; CHECK:   UdtSourceLine (0x1014) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: B (0x1013)
; CHECK:     SourceFile: /t.cpp (0x1008)
; CHECK:     LineNumber: 11
; CHECK:   }
; CHECK: ]


target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

%struct.A = type { i8 }
%class.B = type { i8 }

; Function Attrs: nounwind
define void @"\01?foo@@YAXXZ"() #0 !dbg !6 {
entry:
  %a = alloca %struct.A, align 1
  %b = alloca %class.B, align 1
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !9, metadata !19), !dbg !20
  call void @llvm.dbg.declare(metadata %class.B* %b, metadata !21, metadata !19), !dbg !36
  ret void, !dbg !37
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272316)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 272316)"}
!6 = distinct !DISubprogram(name: "foo", linkageName: "\01?foo@@YAXXZ", scope: !1, file: !1, line: 18, type: !7, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 19, type: !10)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 8, align: 8, elements: !11)
!11 = !{!12, !16, !17, !18}
!12 = !DISubprogram(name: "A::f_default_public", linkageName: "\01?f_default_public@A@@QAEXXZ", scope: !10, file: !1, line: 2, type: !13, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!13 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !DISubprogram(name: "A::f_private", linkageName: "\01?f_private@A@@AAEXXZ", scope: !10, file: !1, line: 4, type: !13, isLocal: false, isDefinition: false, scopeLine: 4, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false)
!17 = !DISubprogram(name: "A::f_protected", linkageName: "\01?f_protected@A@@IAEXXZ", scope: !10, file: !1, line: 6, type: !13, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: false)
!18 = !DISubprogram(name: "A::f_public", linkageName: "\01?f_public@A@@QAEXXZ", scope: !10, file: !1, line: 8, type: !13, isLocal: false, isDefinition: false, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false)
!19 = !DIExpression()
!20 = !DILocation(line: 19, scope: !6)
!21 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 20, type: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !1, line: 11, size: 8, align: 8, elements: !23)
!23 = !{!24, !28, !32}
!24 = !DISubprogram(name: "B::f_default_private", linkageName: "\01?f_default_private@B@@AAEXXZ", scope: !22, file: !1, line: 12, type: !25, isLocal: false, isDefinition: false, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false)
!25 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !26)
!26 = !{null, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 32, align: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!28 = !DISubprogram(name: "B::f", linkageName: "\01?f@B@@QAEXM@Z", scope: !22, file: !1, line: 14, type: !29, isLocal: false, isDefinition: false, scopeLine: 14, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!29 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !30)
!30 = !{null, !27, !31}
!31 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!32 = !DISubprogram(name: "B::f", linkageName: "\01?f@B@@QAEXH@Z", scope: !22, file: !1, line: 15, type: !33, isLocal: false, isDefinition: false, scopeLine: 15, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!33 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !34)
!34 = !{null, !27, !35}
!35 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!36 = !DILocation(line: 20, scope: !6)
!37 = !DILocation(line: 21, scope: !6)
