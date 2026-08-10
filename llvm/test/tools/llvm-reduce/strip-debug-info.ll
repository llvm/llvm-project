; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=strip-debug-info --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s --input-file=%t

; CHECK-INTERESTINGNESS: define i32 @main
; CHECK-FINAL: define i32 @main
; CHECK-FINAL-NOT: !dbg
; CHECK-FINAL-NOT: call {{.*}}llvm.dbg
; CHECK-FINAL-NOT: !llvm.dbg
; CHECK-FINAL-NOT: = !DI

define i32 @main() !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i8, align 1
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %a, metadata !10, metadata !DIExpression()), !dbg !12
  store i8 0, ptr %a, align 1, !dbg !12
  ret i32 0, !dbg !13
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 16.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2ed6e287521b82331926229153026511")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 4, type: !6, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!5 = !DIFile(filename: "/tmp/a.c", directory: "", checksumkind: CSK_MD5, checksum: "2ed6e287521b82331926229153026511")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{}
!10 = !DILocalVariable(name: "a", scope: !4, file: !5, line: 6, type: !11)
!11 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!12 = !DILocation(line: 6, column: 8, scope: !4)
!13 = !DILocation(line: 7, column: 2, scope: !4)
