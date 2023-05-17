; RUN: llc -mtriple=x86_64-unknown-linux-gnu --basic-block-sections=all < %s | FileCheck %s
;
; Verify that AsmPrinter does not optimize debug line directives
; across two distinct sections. Such optimization will lead
; to incorrect debug line number if the memory layout of those
; sections does not strictly mirror the order in the Asm file.
;
;    int G;
;    void func(int x) {
;      if (x) G=0;
;    }
;
; Reduced from the above example, the IR has been modified so that
; the store and return instructions have the same DebugLoc; in this
; context, we should see 2 different directives for line #4
;
; CHECK-COUNT-2: .loc    0 4 1

define void @func(i1 %0) !dbg !8 {
  br i1 %0, label %3, label %2

2:                                                ; preds = %1
  store i32 0, ptr null, align 4, !dbg !12
  br label %3

3:                                                ; preds = %2, %1
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 16.0.2", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "fd2935e6157b382ad81da17a61f24bb1")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "G", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{}
!12 = !DILocation(line: 4, column: 1, scope: !8)
