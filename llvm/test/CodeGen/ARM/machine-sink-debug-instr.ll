; RUN: llc < %s | FileCheck %s
; RUN: opt -passes=strip < %s | llc | FileCheck %s

; A debug record between the subtraction and compare must not allow MachineSink
; to move the subtraction into a successor and prevent compare folding.
; CHECK-LABEL: f:
; CHECK-NOT: cmp
; CHECK: subs r0, r0, r1
; CHECK-NOT: cmp
; CHECK: {{beq|bne}}
; CHECK-NOT: cmp
; CHECK: bx lr

target triple = "thumbv7"

define i32 @f(i32 %a, i32 %b, i32 %c) !dbg !5 {
entry:
  %sub = sub nsw i32 %a, %b, !dbg !10
    #dbg_value(i32 %sub, !9, !DIExpression(), !10)
  %cmp = icmp eq i32 %a, %b, !dbg !10
  br i1 %cmp, label %t, label %e, !dbg !10

t:
  ret i32 %c, !dbg !10

e:
  call void @g(i32 %sub), !dbg !10
  ret i32 0, !dbg !10
}

declare void @g(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{!11, !11, !11, !11}
!8 = !{!9}
!9 = !DILocalVariable(name: "d", scope: !5, file: !1, line: 1, type: !11)
!10 = !DILocation(line: 2, column: 1, scope: !5)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
