; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple x86_64-pc-linux-gnu < %t.ll | FileCheck %s

; CHECK:      .long   .Lline_table_start0          # DW_AT_stmt_list

; CHECK:      .section        .debug_line,"",@progbits
; CHECK-NEXT: .Lline_table_start0:

define void @f() !dbg !0 {
entry:
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7}
!5 = !{!0}

!0 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 1, file: !6, scope: !1, type: !3)
!1 = !DIFile(filename: "test2.c", directory: "/home/espindola/llvm")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 ()", isOptimized: true, emissionKind: FullDebug, file: !6, enums: !{}, retainedTypes: !{})
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!6 = !DIFile(filename: "test2.c", directory: "/home/espindola/llvm")
!7 = !{i32 1, !"Debug Info Version", i32 3}
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
