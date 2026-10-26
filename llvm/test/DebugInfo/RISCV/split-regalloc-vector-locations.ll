; RUN: llc -mtriple=riscv64 -mattr=+v -filetype=obj < %s | llvm-dwarfdump --name=s - | FileCheck %s

; Vector registers are allocated, and rewritten by a VirtRegRewriter, before
; scalar registers. Check that a variable whose value lives in a vector
; register still gets a location when debug values are emitted after scalar
; register allocation.

; CHECK: DW_AT_location
; CHECK-NEXT: DW_OP_regx V{{[0-9]+}}
; CHECK: DW_AT_name ("s")

define <vscale x 2 x i32> @f(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, ptr %p) !dbg !5 {
  %s = add <vscale x 2 x i32> %a, %b, !dbg !8
    #dbg_value(<vscale x 2 x i32> %s, !9, !DIExpression(), !8)
  store <vscale x 2 x i32> %s, ptr %p, !dbg !8
  %t = mul <vscale x 2 x i32> %s, %a, !dbg !8
  ret <vscale x 2 x i32> %t, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 2, column: 1, scope: !5)
!9 = !DILocalVariable(name: "s", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
