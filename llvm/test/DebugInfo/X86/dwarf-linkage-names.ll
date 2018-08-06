; DWARF linkage name attributes are optional; verify they are missing for
; PS4 triple or when tuning for SCE.

; RUN: llc -O0 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s -check-prefix LINKAGE1
; RUN: llc -O0 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s -check-prefix LINKAGE2
; RUN: llc -O0 -mtriple=x86_64-scei-ps4 < %s | FileCheck %s -check-prefix NOLINKAGE
; RUN: llc -O0 -mtriple=x86_64-unknown-unknown -debugger-tune=sce < %s | FileCheck %s -check-prefix NOLINKAGE

; $ clang++ -emit-llvm -S -g dwarf-linkage-names.cpp
; namespace test {
;  int global_var;
;  int bar() { return global_var; }
;};

; With linkage names, we get an attribute for the declaration (first) entry
; for the global variable, and one for the function.

; This assumes the variable will appear before the function.
; LINKAGE1: .section .debug_info
; LINKAGE1: DW_TAG_variable
; LINKAGE1-NOT: DW_TAG
; LINKAGE1: {{DW_AT_(MIPS_)?linkage_name}}
; LINKAGE1: DW_TAG_subprogram
; LINKAGE1-NOT: DW_TAG
; LINKAGE1: {{DW_AT_(MIPS_)?linkage_name}}
; LINKAGE1: .section

; Also verify we see the mangled names. We do this as a separate pass to
; avoid depending on the order of .debug_info and .debug_str sections.

; LINKAGE2-DAG: .asciz   "_ZN4test10global_varE"
; LINKAGE2-DAG: .asciz   "_ZN4test3barEv"

; Without linkage names, verify there aren't any linkage-name attributes,
; and no mangled names.

; NOLINKAGE-NOT: {{DW_AT_(MIPS_)?linkage_name}}
; NOLINKAGE-NOT: .asciz   "_ZN4test10global_varE"
; NOLINKAGE-NOT: .asciz   "_ZN4test3barEv"

source_filename = "test/DebugInfo/X86/dwarf-linkage-names.ll"

@_ZN4test10global_varE = global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define i32 @_ZN4test3barEv() #0 !dbg !11 {
entry:
  %0 = load i32, i32* @_ZN4test10global_varE, align 4, !dbg !14
  ret i32 %0, !dbg !15
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "global_var", linkageName: "_ZN4test10global_varE", scope: !2, file: !3, line: 2, type: !4, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "test", scope: null)
!3 = !DIFile(filename: "dwarf-linkage-names.cpp", directory: "/home/probinson/projects/scratch")
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.8.0 (trunk 244662)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, globals: !7)
!6 = !{}
!7 = !{!0}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 244662)"}
!11 = distinct !DISubprogram(name: "bar", linkageName: "_ZN4test3barEv", scope: !2, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !5, retainedNodes: !6)
!12 = !DISubroutineType(types: !13)
!13 = !{!4}
!14 = !DILocation(line: 3, column: 21, scope: !11)
!15 = !DILocation(line: 3, column: 14, scope: !11)

