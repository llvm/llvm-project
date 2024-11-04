; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t --try-experimental-debuginfo-iterators
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s --implicit-check-not=dbg.value

; Test that we can, in RemoveDIs mode / DPValues mode (where variable location
; information isn't an instruction), remove one variable location assignment
; but not another.

; CHECK-INTERESTINGNESS:     call void @llvm.dbg.value(metadata i32 %added,

; CHECK-FINAL:     declare void @llvm.dbg.value(metadata,
; CHECK-FINAL:     %added = add
; CHECK-FINAL-NEXT: call void @llvm.dbg.value(metadata i32 %added,

declare void @llvm.dbg.value(metadata, metadata, metadata)

define i32 @main() !dbg !7 {
entry:
  %uninteresting1 = alloca i32, align 4
  %interesting = alloca i32, align 4
  %uninteresting2 = alloca i32, align 4
  store i32 0, ptr %uninteresting1, align 4
  store i32 0, ptr %interesting, align 4
  %0 = load i32, ptr %interesting, align 4
  %added = add nsw i32 %0, 1
  tail call void @llvm.dbg.value(metadata i32 %added, metadata !13, metadata !DIExpression()), !dbg !14
  store i32 %added, ptr %interesting, align 4
  %alsoloaded = load i32, ptr %interesting, align 4
  tail call void @llvm.dbg.value(metadata i32 %alsoloaded, metadata !13, metadata !DIExpression()), !dbg !14
  store i32 %alsoloaded, ptr %uninteresting2, align 4
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!8 = !DIFile(filename: "/tmp/a.c", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{}
!13 = !DILocalVariable(name: "a", scope: !7, file: !8, line: 2, type: !11)
!14 = !DILocation(line: 2, column: 7, scope: !7)

