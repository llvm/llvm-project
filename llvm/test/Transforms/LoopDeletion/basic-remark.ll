; RUN: opt -loop-deletion %s -o /dev/null --pass-remarks-output=%t --pass-remarks-filter=loop-delete
; RUN: cat %t | FileCheck %s

; Check that we use the right debug location: the loop header.
; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-delete
; CHECK-NEXT: Name:            Invariant
; CHECK-NEXT: DebugLoc:        { File: loop.c, Line: 2, Column: 3 }
; CHECK-NEXT: Function:        main
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop deleted because it is invariant
; CHECK-NEXT: ...
define i32 @main() local_unnamed_addr #0 {
entry:
  br label %for.cond, !dbg !9

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.cond ]
  %cmp = icmp ult i32 %i.0, 1000
  %inc = add nuw nsw i32 %i.0, 1
  br i1 %cmp, label %for.cond, label %for.cond.cleanup

for.cond.cleanup:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "loop.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "loop.c", directory: "")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 3, scope: !6)
