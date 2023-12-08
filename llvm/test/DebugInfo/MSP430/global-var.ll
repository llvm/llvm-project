; RUN: llc --filetype=obj -o %t < %s
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
; RUN: llvm-dwarfdump --verify %t

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name      ("global_var")
; CHECK-NEXT:   DW_AT_type      ({{0x[0-9]+}} "char")
; CHECK-NEXT:   DW_AT_external  (true)
; CHECK-NEXT:   DW_AT_decl_file ("/tmp{{[/\\]}}global-var.c")
; CHECK-NEXT:   DW_AT_decl_line (1)
; CHECK-NEXT:   DW_AT_location  (DW_OP_addr 0x0)

; ModuleID = 'global-var.c'
source_filename = "global-var.c"
target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

@global_var = dso_local global i8 42, align 1, !dbg !0

; Function Attrs: noinline nounwind optnone
define dso_local i16 @main() #0 !dbg !10 {
entry:
  ret i16 0, !dbg !15
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global_var", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project ...)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "global-var.c", directory: "/tmp")
!4 = !{!0}
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project ...)"}
!10 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 2, type: !11, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocation(line: 2, column: 13, scope: !10)
