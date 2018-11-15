; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=generic | FileCheck %s
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

source_filename = "test/CodeGen/X86/misched-code-difference-with-debug.ll"

%class.C = type { i8 }

@argc = global i8 0, align 1, !dbg !0

declare i32 @test_function(%class.C*, i8 signext, i8 signext, i8 signext, ...)
; CHECK-LABEL: test_without_debug
; CHECK: movl [[A:%[a-z]+]], [[B:%[a-z]+]]
; CHECK: movl [[A]], [[C:%[a-z]+]]

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
; CHECK: movl [[A]], [[C]]

define void @test_with_debug() !dbg !17 {
entry:
  %c = alloca %class.C, align 1
  %0 = load i8, i8* @argc, align 1
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !22, metadata !23), !dbg !24
  %conv = sext i8 %0 to i32
  tail call void @llvm.dbg.value(metadata %class.C* %c, i64 0, metadata !21, metadata !25), !dbg !24
  %call = call i32 (%class.C*, i8, i8, i8, ...) @test_function(%class.C* %c, i8 signext 0, i8 signext %0, i8 signext 0, i32 %conv)
  %1 = load i8, i8* @argc, align 1
  call void @llvm.dbg.value(metadata %class.C* %c, i64 0, metadata !21, metadata !25), !dbg !24
  %call2 = call i32 (%class.C*, i8, i8, i8, ...) @test_function(%class.C* %c, i8 signext 0, i8 signext %1, i8 signext 0, i32 %conv)
  ret void
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "argc", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test.cpp", directory: "")
!3 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !6, globals: !14, imports: !5)
!5 = !{}
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !2, line: 2, size: 8, align: 8, elements: !8, identifier: "_ZTS1C")
!8 = !{!9}
!9 = !DISubprogram(name: "test", scope: !7, file: !2, type: !10, isLocal: false, isDefinition: false, isOptimized: false)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13, !3, !3, !3, null}
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial)
!14 = !{!0}
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = distinct !DISubprogram(name: "test_with_debug", linkageName: "test_with_debug", scope: !2, file: !2, line: 6, type: !18, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !4, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{!21, !22}
!21 = !DILocalVariable(name: "c", scope: !17, file: !2, line: 7, type: !7)
!22 = !DILocalVariable(name: "lc", scope: !17, file: !2, line: 8, type: !3)
!23 = !DIExpression()
!24 = !DILocation(line: 0, scope: !17)
!25 = !DIExpression(DW_OP_deref)

