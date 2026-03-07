; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Verify that without the "BTF" module flag, no .BTF sections are emitted
; for non-BPF targets, even when debug info is present.
;
; Source:
;   int a;
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -S -emit-llvm t.c  (no -gbtf)

@a = common dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; CHECK-NOT:         .section        .BTF
; CHECK-NOT:         .section        .BTF.ext

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
