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
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.A = type { i8 }
%class.B = type { i8 }
%union.U = type { i32 }

@a = global %struct.A zeroinitializer, align 1
@b = global %class.B zeroinitializer, align 1
@u = global %union.U zeroinitializer, align 4

; Function Attrs: nounwind ssp uwtable
define void @_Z4freev() #0 !dbg !30 {
  ret void, !dbg !41
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38, !39}
!llvm.ident = !{!40}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !34, imports: !2)
!1 = !DIFile(filename: "/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp", directory: "")
!2 = !{}
!3 = !{!4, !12, !22}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 3, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "pub_default_static", line: 7, flags: DIFlagStaticMember, file: !1, scope: !4, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "pub_default", linkageName: "_ZN1A11pub_defaultEv", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !1, scope: !4, type: !9)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!12 = !DICompositeType(tag: DW_TAG_class_type, name: "B", line: 11, size: 8, align: 8, file: !1, elements: !13, identifier: "_ZTS1B")
!13 = !{!14, !15, !16, !20, !21}
!14 = !DIDerivedType(tag: DW_TAG_inheritance, flags: DIFlagPublic, scope: !12, baseType: !4)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "public_static", line: 16, flags: DIFlagPublic | DIFlagStaticMember, file: !1, scope: !12, baseType: !7)
!16 = !DISubprogram(name: "pub", linkageName: "_ZN1B3pubEv", line: 14, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 14, file: !1, scope: !12, type: !17)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !12)
!20 = !DISubprogram(name: "prot", linkageName: "_ZN1B4protEv", line: 19, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: false, scopeLine: 19, file: !1, scope: !12, type: !17)
!21 = !DISubprogram(name: "priv_default", linkageName: "_ZN1B12priv_defaultEv", line: 22, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 22, file: !1, scope: !12, type: !17)
!22 = !DICompositeType(tag: DW_TAG_union_type, name: "U", line: 25, size: 32, align: 32, file: !1, elements: !23, identifier: "_ZTS1U")
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "union_priv", line: 30, size: 32, align: 32, flags: DIFlagPrivate, file: !1, scope: !22, baseType: !7)
!25 = !DISubprogram(name: "union_pub_default", linkageName: "_ZN1U17union_pub_defaultEv", line: 27, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 27, file: !1, scope: !22, type: !26)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !22)
!30 = distinct !DISubprogram(name: "free", linkageName: "_Z4freev", line: 35, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 35, file: !1, scope: !31, type: !32, variables: !2)
!31 = !DIFile(filename: "/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp", directory: "")
!32 = !DISubroutineType(types: !33)
!33 = !{null}
!34 = !{!35, !36, !37}
!35 = !DIGlobalVariable(name: "a", line: 37, isLocal: false, isDefinition: true, scope: null, file: !31, type: !4, variable: %struct.A* @a)
!36 = !DIGlobalVariable(name: "b", line: 38, isLocal: false, isDefinition: true, scope: null, file: !31, type: !12, variable: %class.B* @b)
!37 = !DIGlobalVariable(name: "u", line: 39, isLocal: false, isDefinition: true, scope: null, file: !31, type: !22, variable: %union.U* @u)
!38 = !{i32 2, !"Dwarf Version", i32 2}
!39 = !{i32 2, !"Debug Info Version", i32 3}
!40 = !{!"clang version 3.6.0 "}
!41 = !DILocation(line: 35, column: 14, scope: !30)
