; RUN: opt -S -passes=gvn < %s | FileCheck %s

;; Load PRE materializes the load's address in the predecessor where the load
;; is unavailable, here by phi-translating %gep into %left. The new GEP gets a
;; line 0 location in the function's scope that keeps the original's layers.

define i32 @f(ptr %p, ptr %q, i1 %c) !dbg !4 {
; CHECK-LABEL: define i32 @f(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK:       left:
; CHECK-NEXT:    %gep.phi.trans.insert = getelementptr i32, ptr %q, i32 1, !dbg ![[GEP:[0-9]+]]
; CHECK-NEXT:    %v.pre = load i32, ptr %gep.phi.trans.insert
entry:
  br i1 %c, label %left, label %right

left:
  br label %merge

right:
  %rgep = getelementptr i32, ptr %p, i32 1
  store i32 0, ptr %rgep
  br label %merge

merge:
  %ptr = phi ptr [ %p, %right ], [ %q, %left ]
  %gep = getelementptr i32, ptr %ptr, i32 1, !dbg !8
  %v = load i32, ptr %gep
  ret i32 %v
}

; CHECK:      ![[GEP]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
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
