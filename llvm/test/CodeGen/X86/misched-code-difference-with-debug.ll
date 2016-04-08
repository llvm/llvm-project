; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-unknown -mcpu=generic | FileCheck %s
; Both functions should produce the same code. The presence of debug values
; should not affect the scheduling strategy.
; Generated from:
; char argc;
; class C {
; public:
;   int test(char ,char ,char ,...);
; };
; void foo() {
;   C c;
;   char lc = argc;
;   c.test(0,argc,0,lc);
;   c.test(0,argc,0,lc);
; }
;
; with
; clang -O2 -c test.cpp -emit-llvm -S
; clang -O2 -c test.cpp -emit-llvm -S -g
;


%class.C = type { i8 }

@argc = global i8 0, align 1

declare i32 @test_function(%class.C*, i8 signext, i8 signext, i8 signext, ...)

; CHECK-LABEL: test_without_debug
; CHECK: movl [[A:%[a-z]+]], [[B:%[a-z]+]]
; CHECK-NEXT: movl [[A]], [[C:%[a-z]+]]
define void @test_without_debug() {
entry:
  %c = alloca %class.C, align 1
  %0 = load i8, i8* @argc, align 1
  %conv = sext i8 %0 to i32
  %call = call i32 (%class.C*, i8, i8, i8, ...) @test_function(%class.C* %c, i8 signext 0, i8 signext %0, i8 signext 0, i32 %conv)
  %1 = load i8, i8* @argc, align 1
  %call2 = call i32 (%class.C*, i8, i8, i8, ...) @test_function(%class.C* %c, i8 signext 0, i8 signext %1, i8 signext 0, i32 %conv)
  ret void
}

; CHECK-LABEL: test_with_debug
; CHECK: movl [[A]], [[B]]
; CHECK-NEXT: movl [[A]], [[C]]
define void @test_with_debug() !dbg !13 {
entry:
  %c = alloca %class.C, align 1
  %0 = load i8, i8* @argc, align 1
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !19, metadata !29), !dbg !DILocation(scope: !13)
  %conv = sext i8 %0 to i32
  tail call void @llvm.dbg.value(metadata %class.C* %c, i64 0, metadata !18, metadata !DIExpression(DW_OP_deref)), !dbg !DILocation(scope: !13)
  %call = call i32 (%class.C*, i8, i8, i8, ...) @test_function(%class.C* %c, i8 signext 0, i8 signext %0, i8 signext 0, i32 %conv)
  %1 = load i8, i8* @argc, align 1
  call void @llvm.dbg.value(metadata %class.C* %c, i64 0, metadata !18, metadata !DIExpression(DW_OP_deref)), !dbg !DILocation(scope: !13)
  %call2 = call i32 (%class.C*, i8, i8, i8, ...) @test_function(%class.C* %c, i8 signext 0, i8 signext %1, i8 signext 0, i32 %conv)
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, enums: !2, retainedTypes: !3, subprograms: !12, globals: !20, imports: !2, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "C", line: 2, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1C")
!5 = !{!6}
!6 = !DISubprogram(name: "test", file: !1, scope: !"_ZTS1C", type: !7, isDefinition: false)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11, !11, !11, null}
!9 = !DIBasicType(encoding: DW_ATE_signed, size: 32, align: 32, name: "int")
!10 = !DIDerivedType(baseType: !"_ZTS1C", tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!12 = !{!13}
!13 = distinct !DISubprogram(name: "test_with_debug", linkageName: "test_with_debug", line: 6, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 6, file: !1, scope: !14, type: !15, variables: !17)
!14 = !DIFile(filename: "test.cpp", directory: "")
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18, !19}
!18 = !DILocalVariable(name: "c", line: 7, scope: !13, file: !14, type: !"_ZTS1C")
!19 = !DILocalVariable(name: "lc", line: 8, scope: !13, file: !14, type: !11)
!20 = !{!21}
!21 = !DIGlobalVariable(name: "argc", line: 1, isLocal: false, isDefinition: true, scope: null, file: !14, type: !11, variable: i8* @argc)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !DILocation(line: 8, column: 3, scope: !13)
!29 = !DIExpression()
