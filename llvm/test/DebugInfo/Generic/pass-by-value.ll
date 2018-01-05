; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
;
; // S is not trivially copyable.
; struct S {
;    ~S() {}
; };
;
; // T is a POD.
; struct T {
;    ~T() = default;
; };
;  
; S s;
; T t;
;
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_calling_convention	(DW_CC_pass_by_reference)
; CHECK-NEXT: DW_AT_name	("S")
;
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_calling_convention	(DW_CC_pass_by_value)
; CHECK-NEXT: DW_AT_name	("T")

source_filename = "pass.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

%struct.S = type { i8 }
%struct.T = type { i8 }

@s = global %struct.S zeroinitializer, align 1, !dbg !0
@__dso_handle = external hidden global i8
@t = global %struct.T zeroinitializer, align 1, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 9, type: !14, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 321763) (llvm/trunk 321758)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "pass.cpp", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "t", scope: !2, file: !3, line: 10, type: !8, isLocal: false, isDefinition: true)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", file: !3, line: 5, size: 8, elements: !9, identifier: "_ZTS1T", flags: DIFlagTypePassByValue)
!9 = !{!10}
!10 = !DISubprogram(name: "~T", scope: !8, file: !3, line: 6, type: !11, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 8, elements: !15, identifier: "_ZTS1S", flags: DIFlagTypePassByReference)
!15 = !{!16}
!16 = !DISubprogram(name: "~S", scope: !14, file: !3, line: 2, type: !17, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{i32 7, !"PIC Level", i32 2}
!24 = !{!"clang version 7.0.0 (trunk 321763) (llvm/trunk 321758)"}
