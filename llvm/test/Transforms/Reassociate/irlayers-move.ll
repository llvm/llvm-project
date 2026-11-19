; RUN: opt -S -passes=reassociate < %s | FileCheck %s

;; NegateValue moves the existing negation %sub2 out of the loop body, next to
;; its operand's definition. It gets a line 0 location in the function's scope
;; that keeps its layers.

define void @fn1(i32 %a, i1 %c, ptr %ptr) !dbg !4 {
; CHECK-LABEL: define void @fn1(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK:       for.cond:
; CHECK:         {{%.*}} = sub i32 0, %d.0, !dbg ![[NEG:[0-9]+]]
; CHECK:       for.body:
entry:
  br label %for.cond

for.cond:
  %d.0 = phi i32 [ 1, %entry ], [ 2, %for.body ]
  br i1 %c, label %for.end, label %for.body

for.body:
  %sub1 = sub i32 %a, %d.0
  %dead1 = add i32 %sub1, 1
  %dead2 = mul i32 %dead1, 3
  %dead3 = mul i32 %dead2, %sub1
  %sub2 = sub nsw i32 0, %d.0, !dbg !8
  store i32 %sub2, ptr %ptr, align 4
  br label %for.cond

for.end:
  ret void
}

; CHECK:      ![[NEG]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
; CHECK-NEXT: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-NEXT: ![[LAYER]] = !DILayerLoc(line: 100,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!6 = !DILayerLoc(line: 100, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!8 = !DILocation(line: 8, column: 1, scope: !4, irlayers: !7)
