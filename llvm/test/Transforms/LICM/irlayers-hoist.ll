; RUN: opt -S -passes=licm < %s | FileCheck %s

;; An instruction hoisted to the preheader gets a line 0 location in the
;; function's scope that keeps its layers.

define void @f(i32 %x, i32 %n, ptr %p) !dbg !4 {
; CHECK-LABEL: define void @f(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK:       entry:
; CHECK-NEXT:    %inv = mul i32 %x, %x, !dbg ![[INV:[0-9]+]]
; CHECK-NEXT:    br label %loop
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %inv = mul i32 %x, %x, !dbg !8
  %gep = getelementptr i32, ptr %p, i32 %i
  store i32 %inv, ptr %gep
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK:      ![[INV]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
; CHECK-NEXT: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-NEXT: ![[LAYER]] = !DILayerLoc(line: 100,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!6 = !DILayerLoc(line: 100, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!8 = !DILocation(line: 5, column: 9, scope: !4, irlayers: !7)
