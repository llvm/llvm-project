; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mattr=+ptx72 | FileCheck %s
;
;; The comments next to .debug_str entries must escape the string: ptxas
;; (before CUDA 13.1) rejects non-ASCII bytes anywhere in its input, even
;; in comments, and function names can contain arbitrary UTF-8.

; CHECK: .loc {{[0-9]+}} 2 0, function_name [[NAME:\$L__info_string[0-9]+]], inlined_at
; CHECK: .section .debug_str
; CHECK: [[NAME]]:
; CHECK-NEXT: .b8 206 // string offset={{[0-9]+}} ; \xCE\x94x
; CHECK-NOT: Δ

target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @kernel(ptr %a) !dbg !4 {
  store i64 1, ptr %a, !dbg !7
  ret void, !dbg !8
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "test.cu", directory: ".")
!4 = distinct !DISubprogram(name: "kernel", linkageName: "kernel", scope: null, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 2, scope: !9, inlinedAt: !8)
!8 = !DILocation(line: 3, scope: !4)
!9 = distinct !DISubprogram(name: "Δx", linkageName: "Δx", scope: null, file: !3, line: 10, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !2)
