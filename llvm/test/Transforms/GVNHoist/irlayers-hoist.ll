; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

;; The hoisted instruction replaces both copies, so it keeps only the layers
;; they share, on a line 0 location in the function's scope.

;; Both copies have layer 100, so it is kept.
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

;; The copies have different layers, so nothing is shared and nothing is kept.
define i32 @different_layers(i1 %c, i32 %x) !dbg !20 {
; CHECK-LABEL: define i32 @different_layers(
; CHECK:       entry:
; CHECK-NEXT:    %{{.*}} = add i32 %x, 1{{$}}
entry:
  br i1 %c, label %a, label %b

a:
  %y1 = add i32 %x, 1, !dbg !22
  br label %end

b:
  %y2 = add i32 %x, 1, !dbg !23
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
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!6 = !DILayerLoc(line: 100, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!8 = !DILayerLoc(line: 200, column: 1, file: !5, kind: "IntermediateIR")
!9 = !DILayerLocList(!8)

!4 = distinct !DISubprogram(name: "same_layers", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DILocation(line: 3, column: 5, scope: !4, irlayers: !7)
!11 = !DILocation(line: 5, column: 5, scope: !4, irlayers: !7)

!20 = distinct !DISubprogram(name: "different_layers", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!22 = !DILocation(line: 12, column: 5, scope: !20, irlayers: !7)
!23 = !DILocation(line: 14, column: 5, scope: !20, irlayers: !9)
