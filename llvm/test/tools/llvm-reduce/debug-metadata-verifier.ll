; RUN: llvm-reduce %s -o %t --delta-passes=metadata --test %python --test-arg %p/Inputs/remove-metadata.py --abort-on-invalid-reduction
; RUN: FileCheck %s --input-file %t

; CHECK: call void @llvm.dbg.declare{{.*}}, !dbg
; CHECK: !llvm.dbg.cu = !{!0}
; CHECK-NOT: uninteresting

define i32 @main() !dbg !4 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %i, metadata !10, metadata !DIExpression()), !dbg !11
  ret i32 0
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!interesting = !{}
!uninteresting = !{}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 16.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "3b0a4b024d464b367033485450c7a5f9")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!5 = !DIFile(filename: "/tmp/a.c", directory: "", checksumkind: CSK_MD5, checksum: "3b0a4b024d464b367033485450c7a5f9")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{}
!10 = !DILocalVariable(name: "i", scope: !4, file: !5, line: 2, type: !8)
!11 = !DILocation(line: 2, column: 6, scope: !4)
