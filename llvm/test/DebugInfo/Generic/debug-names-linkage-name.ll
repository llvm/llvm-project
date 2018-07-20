; REQUIRES: object-emission

; Generate one file with all linkage names, and another with only abstract ones.
; Then test both.
; RUN: %llc_dwarf -accel-tables=Dwarf -dwarf-linkage-names=All -filetype=obj -o %t.All < %s
; RUN: %llc_dwarf -accel-tables=Dwarf -dwarf-linkage-names=Abstract -filetype=obj -o %t.Abstract < %s
; RUN: llvm-dwarfdump -debug-info -debug-names %t.All | FileCheck %s --check-prefix=ALL
; RUN: llvm-dwarfdump -debug-info -debug-names %t.Abstract \
; RUN:   | FileCheck %s --check-prefix=ABSTRACT --implicit-check-not=_Z1gi --implicit-check-not=_ZN1n1vE 
; RUN: llvm-dwarfdump -debug-names -verify %t.All | FileCheck --check-prefix=VERIFY %s
; RUN: llvm-dwarfdump -debug-names -verify %t.Abstract | FileCheck --check-prefix=VERIFY %s

; We should have all three linkage names in the .debug_info and .debug_names
; ALL: .debug_info contents:
; ALL: DW_AT_linkage_name	("_ZN1n1vE")
; ALL: DW_AT_linkage_name	("_Z1fi")
; ALL: DW_AT_linkage_name	("_Z1gi")
; ALL: .debug_names contents:
; ALL: String: {{.*}} "_Z1fi"
; ALL: String: {{.*}} "_Z1gi"
; ALL: String: {{.*}} "_ZN1n1vE"

; Only _Z1fi should be present in both sections
; ABSTRACT: .debug_info contents:
; ABSTRACT: DW_AT_linkage_name	("_Z1fi")
; ABSTRACT: .debug_names contents:
; ABSTRACT: String: {{.*}} "_Z1fi"

; There should be no verification errors for both files.
; VERIFY: No errors.

; Input generated from the following C code using
; clang -g -O2 -S -emit-llvm

; int e(int);
; inline int f(int a) { return e(a); }
; int g(int a) { return f(a); }
; 
; namespace n {
; int v;
; }

@_ZN1n1vE = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local i32 @_Z1gi(i32 %a) local_unnamed_addr !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %a, metadata !18, metadata !DIExpression()), !dbg !21
  %call.i = tail call i32 @_Z1ei(i32 %a), !dbg !23
  ret i32 %call.i, !dbg !24
}

declare dso_local i32 @_Z1ei(i32) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "v", linkageName: "_ZN1n1vE", scope: !2, file: !3, line: 6, type: !4, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "n", scope: null)
!3 = !DIFile(filename: "/tmp/linkage-name.cc", directory: "/usr/local/google/home/labath/ll/build/dbg")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 331861) (llvm/trunk 331884)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, globals: !7)
!6 = !{}
!7 = !{!0}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 7.0.0 (trunk 331861) (llvm/trunk 331884)"}
!12 = distinct !DISubprogram(name: "g", linkageName: "_Z1gi", scope: !3, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !5, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!4, !4}
!15 = !{!16}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !3, line: 3, type: !4)
!17 = !DILocation(line: 3, column: 11, scope: !12)
!18 = !DILocalVariable(name: "a", arg: 1, scope: !19, file: !3, line: 2, type: !4)
!19 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !3, file: !3, line: 2, type: !13, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !5, retainedNodes: !20)
!20 = !{!18}
!21 = !DILocation(line: 2, column: 18, scope: !19, inlinedAt: !22)
!22 = distinct !DILocation(line: 3, column: 23, scope: !12)
!23 = !DILocation(line: 2, column: 30, scope: !19, inlinedAt: !22)
!24 = !DILocation(line: 3, column: 16, scope: !12)
