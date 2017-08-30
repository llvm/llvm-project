; RUN: opt -run-twice -verify -disable-debug-info-type-map -S -o - %s | FileCheck %s

; Generated using:
; $ cat p.cpp
; void sink(void *);
; class A {
; public:
;   template <typename> void m_fn2() { static int a; }
;   virtual void m_fn1();
; };
; void foo() {
;   class B : public A {
;   public:
;     B() { m_fn2<B>(); }
;   };
;   sink(new B);
; }
; $ clang++ -target x86_64-unknown-linux -fvisibility=hidden -O2 -g2 -flto -S p.cpp -o p.ll
; # then manually removed function/gv definitions

; Test that when the module is cloned it does not contain a reference to
; the original DICompileUnit as a result of a collision between the cloned
; DISubprogram for m_fn2<B> (which refers to the non-ODR entity B via
; template parameters) and the original DISubprogram.

; CHECK: DICompileUnit
; CHECK-NOT: DICompileUnit

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29}
!llvm.ident = !{!30}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "p.cpp", directory: "/usr/local/google/home/pcc/b682773-2-repro/small2")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "a", scope: !6, file: !1, line: 5, type: !27, isLocal: true, isDefinition: true)
!6 = distinct !DISubprogram(name: "m_fn2<B>", linkageName: "_ZN1A5m_fn2IZ3foovE1BEEvv", scope: !7, file: !1, line: 5, type: !8, isLocal: true, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !11, declaration: !23, variables: !24)
!7 = !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !1, line: 3, flags: DIFlagFwdDecl, identifier: "_ZTS1A")
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = !{!12}
!12 = !DITemplateTypeParameter(type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", scope: !14, file: !1, line: 10, size: 64, elements: !17, vtableHolder: !7)
!14 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 9, type: !15, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !13, baseType: !7, flags: DIFlagPublic)
!19 = !DISubprogram(name: "B", scope: !13, file: !1, line: 12, type: !20, isLocal: false, isDefinition: false, scopeLine: 12, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!23 = !DISubprogram(name: "m_fn2<B>", linkageName: "_ZN1A5m_fn2IZ3foovE1BEEvv", scope: !7, file: !1, line: 5, type: !8, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true, templateParams: !11)
!24 = !{!25}
!25 = !DILocalVariable(name: "this", arg: 1, scope: !6, type: !26, flags: DIFlagArtificial | DIFlagObjectPointer)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!28 = !{i32 2, !"Dwarf Version", i32 4}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{!"clang version 5.0.0 "}
