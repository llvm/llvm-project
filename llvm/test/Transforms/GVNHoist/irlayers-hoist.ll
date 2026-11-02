; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

;; The hoisted instruction replaces both copies and gets a line 0 location in
;; the function's scope that keeps its layers.

define i32 @same_layers(i1 %c, i32 %x) !dbg !4 {
; CHECK-LABEL: define i32 @same_layers(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK:       entry:
; CHECK-NEXT:    %{{.*}} = add i32 %x, 1, !dbg ![[SAME:[0-9]+]]
entry:
  br i1 %c, label %a, label %b

a:
  %y1 = add i32 %x, 1, !dbg !10
  br label %end

b:
  %y2 = add i32 %x, 1, !dbg !11
  br label %end

end:
  %r = phi i32 [ %y1, %a ], [ %y2, %b ]
  ret i32 %r
}

; CHECK:      ![[SAME]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
; CHECK-NEXT: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-NEXT: ![[LAYER]] = !DILayerLoc(line: 100,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "same_layers", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!6 = !DILayerLoc(line: 100, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!10 = !DILocation(line: 3, column: 5, scope: !4, irlayers: !7)
!11 = !DILocation(line: 5, column: 5, scope: !4, irlayers: !7)
