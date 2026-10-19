;; Tests that an !flmd attachment other than !MDFunctionLocalContext is rejected
;; by the verifier.
; RUN: llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Unexpected !flmd attachment to function

define i32 @foo(i32 noundef %i) #0 !dbg !9 !flmd !11 {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4, !dbg !12
  %add = add nsw i32 %0, 1, !dbg !12
  store i32 %add, ptr %j, align 4, !dbg !12
  %1 = load i32, ptr %j, align 4, !dbg !13
  %mul = mul nsw i32 %1, 2, !dbg !13
  store i32 %mul, ptr %k, align 4, !dbg !13
  %2 = load i32, ptr %k, align 4, !dbg !14
  %sub = sub nsw i32 %2, 1, !dbg !14
  ret i32 %sub, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 24.0.0git"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 3, scope: !9)
!13 = !DILocation(line: 4, scope: !9)
!14 = !DILocation(line: 5, scope: !9)