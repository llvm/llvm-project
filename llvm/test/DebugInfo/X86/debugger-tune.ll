; Verify target-based defaults for "debugger tuning," and the ability to
; override defaults.
; We use existence of the debug_pubnames section to distinguish the GDB case,
; and the apple_names section to distinguish the LLDB case. SCE has neither.

; Verify defaults for various targets.
; RUN: llc -mtriple=x86_64-scei-ps4 -filetype=obj < %s | llvm-readobj -sections - | FileCheck --check-prefix=SCE %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s | llvm-readobj -sections - | FileCheck --check-prefix=LLDB %s
; RUN: llc -mtriple=x86_64-pc-freebsd -filetype=obj < %s | llvm-readobj -sections - | FileCheck --check-prefix=GDB %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj < %s | llvm-readobj -sections - | FileCheck --check-prefix=GDB %s

; We can override defaults.
; RUN: llc -mtriple=x86_64-scei-ps4 -filetype=obj -debugger-tune=gdb < %s | llvm-readobj -sections - | FileCheck --check-prefix=GDB %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -debugger-tune=lldb < %s | llvm-readobj -sections - | FileCheck --check-prefix=LLDB %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj -debugger-tune=sce < %s | llvm-readobj -sections - | FileCheck --check-prefix=SCE %s

; GDB-NOT: apple_names
; GDB: debug_pubnames
; GDB-NOT: apple_names

; LLDB-NOT: debug_pubnames
; LLDB: apple_names
; LLDB-NOT: debug_pubnames

; SCE-NOT: debug_pubnames
; SCE-NOT: apple_names

source_filename = "test/DebugInfo/X86/debugger-tune.ll"

@globalvar = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "globalvar", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.7.0 (trunk 238808)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "debugger-tune.cpp", directory: "/home/probinson/projects/scratch")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.7.0 (trunk 238808)"}

