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
;   protected:
;     void prot();
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
;   public:
;     D d;
;     E e;
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
;   A a;
;   B b;
;   U u;
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
%class.B = type { i8 }
%class.C = type { %"struct.C::D", %"union.C::E" }
%"struct.C::D" = type { i8 }
%"union.C::E" = type { i8 }
%struct.F = type { %"union.F::G" }
%"union.F::G" = type { i8 }
%union.H = type { %"class.H::I" }
%"class.H::I" = type { i8 }

@u = global %union.U zeroinitializer, align 4, !dbg !0
@a = global %struct.A zeroinitializer, align 1, !dbg !5
@b = global %class.B zeroinitializer, align 1, !dbg !16
@c = global %class.C zeroinitializer, align 1, !dbg !28
@f = global %struct.F zeroinitializer, align 1, !dbg !37
@h = global %union.H zeroinitializer, align 1, !dbg !43

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z4freev() #0 !dbg !59 {
entry:
  ret void, !dbg !62
}

attributes #0 = { mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!56, !57}
!llvm.ident = !{!58}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "u", scope: !2, file: !7, line: 73, type: !49, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "clang/test/CodeGenCXX/<stdin>", directory: "")
!4 = !{!0, !5, !16, !28, !37, !43}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !7, line: 74, type: !8, isLocal: false, isDefinition: true)
!7 = !DIFile(filename: "clang/test/CodeGenCXX/debug-info-access.cpp", directory: "")
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !7, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !9, identifier: "_ZTS1A")
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "pub_default_static", scope: !8, file: !7, line: 9, baseType: !11, flags: DIFlagStaticMember)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DISubprogram(name: "pub_default", linkageName: "_ZN1A11pub_defaultEv", scope: !8, file: !7, line: 7, type: !13, scopeLine: 7, flags: DIFlagPrototyped, spFlags: 0)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !7, line: 75, type: !18, isLocal: false, isDefinition: true)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !7, line: 14, size: 8, flags: DIFlagTypePassByValue, elements: !19, identifier: "_ZTS1B")
!19 = !{!20, !21, !22, !26, !27}
!20 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !18, baseType: !8, flags: DIFlagPublic, extraData: i32 0)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "public_static", scope: !18, file: !7, line: 19, baseType: !11, flags: DIFlagPublic | DIFlagStaticMember)
!22 = !DISubprogram(name: "pub", linkageName: "_ZN1B3pubEv", scope: !18, file: !7, line: 17, type: !23, scopeLine: 17, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!26 = !DISubprogram(name: "prot", linkageName: "_ZN1B4protEv", scope: !18, file: !7, line: 22, type: !23, scopeLine: 22, flags: DIFlagProtected | DIFlagPrototyped, spFlags: 0)
!27 = !DISubprogram(name: "priv_default", linkageName: "_ZN1B12priv_defaultEv", scope: !18, file: !7, line: 25, type: !23, scopeLine: 25, flags: DIFlagPrototyped, spFlags: 0)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !7, line: 76, type: !30, isLocal: false, isDefinition: true)
!30 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !7, line: 28, size: 16, flags: DIFlagTypePassByValue, elements: !31, identifier: "_ZTS1C")
!31 = !{!32, !35}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !30, file: !7, line: 38, baseType: !33, size: 8, flags: DIFlagPublic)
!33 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", scope: !30, file: !7, line: 31, size: 8, flags: DIFlagPublic | DIFlagTypePassByValue, elements: !34, identifier: "_ZTSN1C1DE")
!34 = !{}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !30, file: !7, line: 39, baseType: !36, size: 8, offset: 8, flags: DIFlagPublic)
!36 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "E", scope: !30, file: !7, line: 35, size: 8, flags: DIFlagProtected | DIFlagTypePassByValue, elements: !34, identifier: "_ZTSN1C1EE")
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = distinct !DIGlobalVariable(name: "f", scope: !2, file: !7, line: 77, type: !39, isLocal: false, isDefinition: true)
!39 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "F", file: !7, line: 42, size: 8, flags: DIFlagTypePassByValue, elements: !40, identifier: "_ZTS1F")
!40 = !{!41}
!41 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !39, file: !7, line: 48, baseType: !42, size: 8)
!42 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "G", scope: !39, file: !7, line: 45, size: 8, flags: DIFlagPrivate | DIFlagTypePassByValue, elements: !34, identifier: "_ZTSN1F1GE")
!43 = !DIGlobalVariableExpression(var: !44, expr: !DIExpression())
!44 = distinct !DIGlobalVariable(name: "h", scope: !2, file: !7, line: 78, type: !45, isLocal: false, isDefinition: true)
!45 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "H", file: !7, line: 51, size: 8, flags: DIFlagTypePassByValue, elements: !46, identifier: "_ZTS1H")
!46 = !{!47}
!47 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !45, file: !7, line: 57, baseType: !48, size: 8)
!48 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "I", scope: !45, file: !7, line: 54, size: 8, flags: DIFlagPrivate | DIFlagTypePassByValue, elements: !34, identifier: "_ZTSN1H1IE")
!49 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "U", file: !7, line: 60, size: 32, flags: DIFlagTypePassByValue, elements: !50, identifier: "_ZTS1U")
!50 = !{!51, !52}
!51 = !DIDerivedType(tag: DW_TAG_member, name: "union_priv", scope: !49, file: !7, line: 65, baseType: !11, size: 32, flags: DIFlagPrivate)
!52 = !DISubprogram(name: "union_pub_default", linkageName: "_ZN1U17union_pub_defaultEv", scope: !49, file: !7, line: 62, type: !53, scopeLine: 62, flags: DIFlagPrototyped, spFlags: 0)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !55}
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !49, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!56 = !{i32 2, !"Debug Info Version", i32 3}
!57 = !{i32 1, !"wchar_size", i32 4}
!58 = !{!"clang version 14.0.0"}
!59 = distinct !DISubprogram(name: "free", linkageName: "_Z4freev", scope: !7, file: !7, line: 71, type: !60, scopeLine: 71, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
!60 = !DISubroutineType(types: !61)
!61 = !{null}
!62 = !DILocation(line: 71, column: 14, scope: !59)
