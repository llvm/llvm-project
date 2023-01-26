; RUN: llc -filetype=obj -mtriple=aarch64--linux-gnu -o - %s | llvm-dwarfdump -v - | FileCheck %s
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_AT_location

@var = thread_local global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind optnone
define i32 @foo() #0 !dbg !11 {
entry:
  %0 = load i32, ptr @var, align 4, !dbg !14
  ret i32 %0, !dbg !15
}

attributes #0 = { noinline nounwind optnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "tls-at-location.c", directory: "/home/lliu0/llvm/tls-at-location/DebugInfo/AArch64")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0"}
!11 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !DILocation(line: 4, column: 10, scope: !11)
!15 = !DILocation(line: 4, column: 3, scope: !11)
