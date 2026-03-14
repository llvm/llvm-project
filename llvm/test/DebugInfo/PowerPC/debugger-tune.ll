; Verify target-based defaults for "debugger tuning," and the ability to
; override defaults.
; We use the use of DW_FORM_string (rather than DW_FORM_strp) to distinguish the debugger tuning.

; Verify defaults for various targets.
; RUN: llc -mtriple=powerpc64le-unknown-linux -filetype=obj < %s | llvm-dwarfdump -debug-info --show-form - | FileCheck --check-prefix=GDB %s --implicit-check-not DW_FORM_string
; TODO: Use -filetype-obj and llvm-dwarfdump when the obj mode is supported.
; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff < %s | FileCheck --check-prefix=DBX --implicit-check-not DW_FROM_strp %s

; We can override defaults.
; RUN: llc -mtriple=powerpc64le-unknown-linux -filetype=obj -debugger-tune=dbx < %s | llvm-dwarfdump -debug-info --show-form - | FileCheck --check-prefix=DBX %s --implicit-check-not DW_FROM_strp
; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff -debugger-tune=gdb < %s | FileCheck --check-prefix=GDB %s --implicit-check-not DW_FORM_string

; GDB: DW_FORM_strp
; DBX: DW_FORM_string

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "1.c", directory: "debug")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 12.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !8)
