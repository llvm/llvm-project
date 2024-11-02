; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; Test the DW_AT_accessibility DWARF attribute.

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
;
;   protected:
;     typedef int prot_typedef;
;     using prot_using = prot_typedef;
;     prot_using prot_member;
;
;     void prot();
;
;   private:
;     void priv_default();
;   };
;
;   class C {
;   public:
;     struct D {
;     };
;   protected:
;     union E {
;     };
;   private:
;     struct J {
;     };
;   public:
;     D d;
;     E e;
;     J j;
;   };
;
;   struct F {
;   private:
;     union G {
;     };
;   public:
;     G g;
;   };
;
;   union H {
;   private:
;     class I {
;     };
;   public:
;     I i;
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
;   U u;
;   A a;
;   B b;
;   C c;
;   F f;
;   H h;

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"union_priv")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_private)

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"union_pub_default")
; CHECK-NOT: DW_AT_accessibility

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"pub_default_static")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG

; CHECK: DW_TAG_inheritance
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"public_static")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"prot_member")

; CHECK: DW_TAG_typedef
; CHECK:     DW_AT_name {{.*}}"prot_using")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_protected)

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"prot")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_protected)

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"priv_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG

; CHECK: DW_TAG_structure_type
; CHECK:     DW_AT_name ("D")
; CHECK:     DW_AT_accessibility (DW_ACCESS_public)

; CHECK: DW_TAG_union_type
; CHECK:     DW_AT_name ("E")
; CHECK:     DW_AT_byte_size (0x01)
; CHECK:     DW_AT_accessibility (DW_ACCESS_protected)

; CHECK: DW_TAG_union_type
; CHECK:     DW_AT_name ("G")
; CHECK:     DW_AT_accessibility (DW_ACCESS_private)

; CHECK: DW_TAG_class_type
; CHECK:     DW_AT_name ("I")
; CHECK:     DW_AT_accessibility (DW_ACCESS_private)

; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"free")
; CHECK-NOT: DW_AT_accessibility

%union.U = type { i32 }
%struct.A = type { i8 }
%class.B = type { i32 }
%class.C = type { %"struct.C::D", %"union.C::E", %"struct.C::J" }
%"struct.C::D" = type { i8 }
%"union.C::E" = type { i8 }
%"struct.C::J" = type { i8 }
%struct.F = type { %"union.F::G" }
%"union.F::G" = type { i8 }
%union.H = type { %"class.H::I" }
%"class.H::I" = type { i8 }

@u = dso_local global %union.U zeroinitializer, align 4, !dbg !0
@a = dso_local global %struct.A zeroinitializer, align 1, !dbg !5
@b = dso_local global %class.B zeroinitializer, align 4, !dbg !16
@c = dso_local global %class.C zeroinitializer, align 1, !dbg !31
@f = dso_local global %struct.F zeroinitializer, align 1, !dbg !42
@h = dso_local global %union.H zeroinitializer, align 1, !dbg !48

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z4freev() #0 !dbg !69 {
entry:
  ret void, !dbg !72
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!61, !62, !63, !64, !65, !66, !67}
!llvm.ident = !{!68}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "u", scope: !2, file: !7, line: 86, type: !54, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 16.0.0 (https://github.com/llvm/llvm-project.git 113a643a597b6a8f68099fedbeb7509449d4bd50)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "/home/jon/sources/llvm-project/clang/test/CodeGenCXX/debug-info-access.cpp", directory: "/home/jon/sources/llvm-project/build/testing", checksumkind: CSK_MD5, checksum: "98644ed3fc3955a9b5fefee27d5c16ef")
!4 = !{!0, !5, !16, !31, !42, !48}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !7, line: 87, type: !8, isLocal: false, isDefinition: true)
!7 = !DIFile(filename: "clang/test/CodeGenCXX/debug-info-access.cpp", directory: "/home/jon/sources/llvm-project", checksumkind: CSK_MD5, checksum: "98644ed3fc3955a9b5fefee27d5c16ef")
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !7, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !9, identifier: "_ZTS1A")
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "pub_default_static", scope: !8, file: !7, line: 9, baseType: !11, flags: DIFlagStaticMember)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DISubprogram(name: "pub_default", linkageName: "_ZN1A11pub_defaultEv", scope: !8, file: !7, line: 7, type: !13, scopeLine: 7, flags: DIFlagPrototyped, spFlags: 0)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !7, line: 88, type: !18, isLocal: false, isDefinition: true)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !7, line: 13, size: 32, flags: DIFlagTypePassByValue, elements: !19, identifier: "_ZTS1B")
!19 = !{!20, !21, !22, !25, !29, !30}
!20 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !18, baseType: !8, flags: DIFlagPublic, extraData: i32 0)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "public_static", scope: !18, file: !7, line: 18, baseType: !11, flags: DIFlagPublic | DIFlagStaticMember)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "prot_member", scope: !18, file: !7, line: 25, baseType: !23, size: 32, flags: DIFlagProtected)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "prot_using", scope: !18, file: !7, line: 24, baseType: !24, flags: DIFlagProtected)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "prot_typedef", scope: !18, file: !7, line: 22, baseType: !11, flags: DIFlagProtected)
!25 = !DISubprogram(name: "pub", linkageName: "_ZN1B3pubEv", scope: !18, file: !7, line: 16, type: !26, scopeLine: 16, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = !DISubprogram(name: "prot", linkageName: "_ZN1B4protEv", scope: !18, file: !7, line: 29, type: !26, scopeLine: 29, flags: DIFlagProtected | DIFlagPrototyped, spFlags: 0)
!30 = !DISubprogram(name: "priv_default", linkageName: "_ZN1B12priv_defaultEv", scope: !18, file: !7, line: 33, type: !26, scopeLine: 33, flags: DIFlagPrototyped, spFlags: 0)
!31 = !DIGlobalVariableExpression(var: !32, expr: !DIExpression())
!32 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !7, line: 89, type: !33, isLocal: false, isDefinition: true)
!33 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !7, line: 36, size: 24, flags: DIFlagTypePassByValue, elements: !34, identifier: "_ZTS1C")
!34 = !{!35, !38, !40}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !33, file: !7, line: 50, baseType: !36, size: 8, flags: DIFlagPublic)
!36 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", scope: !33, file: !7, line: 39, size: 8, flags: DIFlagPublic | DIFlagTypePassByValue, elements: !37, identifier: "_ZTSN1C1DE")
!37 = !{}
!38 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !33, file: !7, line: 51, baseType: !39, size: 8, offset: 8, flags: DIFlagPublic)
!39 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "E", scope: !33, file: !7, line: 43, size: 8, flags: DIFlagProtected | DIFlagTypePassByValue, elements: !37, identifier: "_ZTSN1C1EE")
!40 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !33, file: !7, line: 52, baseType: !41, size: 8, offset: 16, flags: DIFlagPublic)
!41 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "J", scope: !33, file: !7, line: 47, size: 8, flags: DIFlagTypePassByValue, elements: !37, identifier: "_ZTSN1C1JE")
!42 = !DIGlobalVariableExpression(var: !43, expr: !DIExpression())
!43 = distinct !DIGlobalVariable(name: "f", scope: !2, file: !7, line: 90, type: !44, isLocal: false, isDefinition: true)
!44 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "F", file: !7, line: 55, size: 8, flags: DIFlagTypePassByValue, elements: !45, identifier: "_ZTS1F")
!45 = !{!46}
!46 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !44, file: !7, line: 61, baseType: !47, size: 8)
!47 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "G", scope: !44, file: !7, line: 58, size: 8, flags: DIFlagPrivate | DIFlagTypePassByValue, elements: !37, identifier: "_ZTSN1F1GE")
!48 = !DIGlobalVariableExpression(var: !49, expr: !DIExpression())
!49 = distinct !DIGlobalVariable(name: "h", scope: !2, file: !7, line: 91, type: !50, isLocal: false, isDefinition: true)
!50 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "H", file: !7, line: 64, size: 8, flags: DIFlagTypePassByValue, elements: !51, identifier: "_ZTS1H")
!51 = !{!52}
!52 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !50, file: !7, line: 70, baseType: !53, size: 8)
!53 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "I", scope: !50, file: !7, line: 67, size: 8, flags: DIFlagPrivate | DIFlagTypePassByValue, elements: !37, identifier: "_ZTSN1H1IE")
!54 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "U", file: !7, line: 73, size: 32, flags: DIFlagTypePassByValue, elements: !55, identifier: "_ZTS1U")
!55 = !{!56, !57}
!56 = !DIDerivedType(tag: DW_TAG_member, name: "union_priv", scope: !54, file: !7, line: 78, baseType: !11, size: 32, flags: DIFlagPrivate)
!57 = !DISubprogram(name: "union_pub_default", linkageName: "_ZN1U17union_pub_defaultEv", scope: !54, file: !7, line: 75, type: !58, scopeLine: 75, flags: DIFlagPrototyped, spFlags: 0)
!58 = !DISubroutineType(types: !59)
!59 = !{null, !60}
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !54, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!61 = !{i32 7, !"Dwarf Version", i32 5}
!62 = !{i32 2, !"Debug Info Version", i32 3}
!63 = !{i32 1, !"wchar_size", i32 4}
!64 = !{i32 8, !"PIC Level", i32 2}
!65 = !{i32 7, !"PIE Level", i32 2}
!66 = !{i32 7, !"uwtable", i32 2}
!67 = !{i32 7, !"frame-pointer", i32 2}
!68 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 113a643a597b6a8f68099fedbeb7509449d4bd50)"}
!69 = distinct !DISubprogram(name: "free", linkageName: "_Z4freev", scope: !7, file: !7, line: 84, type: !70, scopeLine: 84, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
!70 = !DISubroutineType(types: !71)
!71 = !{null}
!72 = !DILocation(line: 84, column: 14, scope: !69)
