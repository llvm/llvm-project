; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; When removing the llvm.dbg.value intrinsic in the unreachable block
; InstCombine would incorrectly return a false Modified status.

; CHECK: cond.true:
; CHECK-NEXT: br label %cond.end

define i32 @foo() !dbg !7 {
entry:
  br i1 false, label %cond.true, label %cond.end

cond.true:
  call void @llvm.dbg.value(metadata i32 undef, metadata !12, metadata !DIExpression()), !dbg !13
  br label %cond.end

cond.end:
  ret i32 undef
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !12}
!12 = !DILocalVariable(name: "bar", scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocation(line: 0, scope: !7)
