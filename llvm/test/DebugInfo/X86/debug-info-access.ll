; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; Test the DW_AT_accessibility DWARF attribute.
;
;
; Regenerate me:
; clang++ -g tools/clang/test/CodeGenCXX/debug-info-access.cpp -S -emit-llvm -o -
;
;   struct A {
;     void pub_default();
;     static int pub_default_static;
;   };
;
;   class B : public A {
;   public:
;     void pub();
;     static int public_static;
;   protected:
;     void prot();
;   private:
;     void priv_default();
;   };
;
;   union U {
;     void union_pub_default();
;   private:
;     int union_priv;
;   };
;
;   void free() {}
;
;   A a;
;   B b;
;   U u;

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"pub_default_static")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_inheritance
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"public_static")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"prot")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_protected)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"priv_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"union_priv")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_private)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"union_pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"free")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG
;
; ModuleID = '/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp'
source_filename = "test/DebugInfo/X86/debug-info-access.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.A = type { i8 }
%class.B = type { i8 }
%union.U = type { i32 }

@a = global %struct.A zeroinitializer, align 1, !dbg !0
@b = global %class.B zeroinitializer, align 1, !dbg !11
@u = global %union.U zeroinitializer, align 4, !dbg !23

; Function Attrs: nounwind ssp uwtable
define void @_Z4freev() #0 !dbg !39 {
  ret void, !dbg !42
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!32}
!llvm.module.flags = !{!36, !37}
!llvm.ident = !{!38}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 37, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !2, line: 3, size: 8, align: 8, elements: !4, identifier: "_ZTS1A")
!4 = !{!5, !7}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "pub_default_static", scope: !3, file: !2, line: 7, baseType: !6, flags: DIFlagStaticMember)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DISubprogram(name: "pub_default", linkageName: "_ZN1A11pub_defaultEv", scope: !3, file: !2, line: 5, type: !8, isLocal: false, isDefinition: false, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = !DIGlobalVariableExpression(var: !12)
!12 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 38, type: !13, isLocal: false, isDefinition: true)
!13 = !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !2, line: 11, size: 8, align: 8, elements: !14, identifier: "_ZTS1B")
!14 = !{!15, !16, !17, !21, !22}
!15 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !13, baseType: !3, flags: DIFlagPublic)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "public_static", scope: !13, file: !2, line: 16, baseType: !6, flags: DIFlagPublic | DIFlagStaticMember)
!17 = !DISubprogram(name: "pub", linkageName: "_ZN1B3pubEv", scope: !13, file: !2, line: 14, type: !18, isLocal: false, isDefinition: false, scopeLine: 14, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DISubprogram(name: "prot", linkageName: "_ZN1B4protEv", scope: !13, file: !2, line: 19, type: !18, isLocal: false, isDefinition: false, scopeLine: 19, virtualIndex: 6, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: false)
!22 = !DISubprogram(name: "priv_default", linkageName: "_ZN1B12priv_defaultEv", scope: !13, file: !2, line: 22, type: !18, isLocal: false, isDefinition: false, scopeLine: 22, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!23 = !DIGlobalVariableExpression(var: !24)
!24 = !DIGlobalVariable(name: "u", scope: null, file: !2, line: 39, type: !25, isLocal: false, isDefinition: true)
!25 = !DICompositeType(tag: DW_TAG_union_type, name: "U", file: !2, line: 25, size: 32, align: 32, elements: !26, identifier: "_ZTS1U")
!26 = !{!27, !28}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "union_priv", scope: !25, file: !2, line: 30, baseType: !6, size: 32, align: 32, flags: DIFlagPrivate)
!28 = !DISubprogram(name: "union_pub_default", linkageName: "_ZN1U17union_pub_defaultEv", scope: !25, file: !2, line: 27, type: !29, isLocal: false, isDefinition: false, scopeLine: 27, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.6.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !33, retainedTypes: !34, globals: !35, imports: !33)
!33 = !{}
!34 = !{!3, !13, !25}
!35 = !{!0, !11, !23}
!36 = !{i32 2, !"Dwarf Version", i32 2}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{!"clang version 3.6.0 "}
!39 = distinct !DISubprogram(name: "free", linkageName: "_Z4freev", scope: !2, file: !2, line: 35, type: !40, isLocal: false, isDefinition: true, scopeLine: 35, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !32, variables: !33)
!40 = !DISubroutineType(types: !41)
!41 = !{null}
!42 = !DILocation(line: 35, column: 14, scope: !39)

