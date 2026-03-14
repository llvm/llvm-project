; RUN: llc -O2 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; Check that Machine CSE correctly handles during the transformation, the
; debug location information for variables.

; Generated with clang -c -g -O2

; typedef float __attribute__((__vector_size__(16))) f4;
; f4 get();
; int main() {
;   float MyVar = get()[0];
;   if (MyVar)
;     return 1;
; }

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @main() !dbg !7 {
entry:
  %call = tail call <4 x float> @_Z3getv(), !dbg !14
  %vecext = extractelement <4 x float> %call, i32 0, !dbg !14
  call void @llvm.dbg.value(metadata float %vecext, metadata !12, metadata !DIExpression()), !dbg !15
  %tobool = fcmp une float %vecext, 0.000000e+00, !dbg !16
  %. = zext i1 %tobool to i32, !dbg !18
  ret i32 %., !dbg !19
}

declare dso_local <4 x float> @_Z3getv()

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 339665)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 339665)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "MyVar", scope: !7, file: !1, line: 4, type: !13)
!13 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!14 = !DILocation(line: 4, column: 18, scope: !7)
!15 = !DILocation(line: 4, column: 9, scope: !7)
!16 = !DILocation(line: 5, column: 7, scope: !17)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 5, column: 7)
!18 = !DILocation(line: 6, column: 5, scope: !17)
!19 = !DILocation(line: 7, column: 1, scope: !7)

; Look at the debug location information for variable 'MyVar'.
; Verify that we see a sequence of DI entries, that looks like:
; DW_TAG_variable
;   DW_AT_location        (0x00000000
;     [0x0000000000000009,  0x0000000000000012): DW_OP_reg17 XMM0)
;   DW_AT_name    ("MyVar")

; CHECK-LABEL: DW_TAG_variable
; CHECK-NEXT: DW_AT_location{{.*}}
; CHECK-NEXT: {{.*}}DW_OP_reg17 XMM0
; CHECK-NEXT: DW_AT_name{{.*}}("MyVar")
