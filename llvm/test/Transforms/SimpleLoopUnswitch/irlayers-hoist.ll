; RUN: opt -passes='simple-loop-unswitch<nontrivial>' -S < %s | FileCheck %s

;; The unswitched branch, hoisted out of the loop, gets a line 0 location in the
;; function's scope that keeps its layers.

declare void @a()
declare void @b()

define void @f(i32 %n, i1 %c) !dbg !4 {
; CHECK-LABEL: define void @f(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK:       entry:
; CHECK-NEXT:    %c.fr = freeze i1 %c{{$}}
; CHECK-NEXT:    br i1 %c.fr, label %{{.*}}, label %{{.*}}, !dbg ![[HOISTED:[0-9]+]]
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %body, label %exit

body:
  br i1 %c, label %then, label %else, !dbg !8

then:
  call void @a()
  br label %latch

else:
  call void @b()
  br label %latch

latch:
  %inc = add nuw nsw i32 %i, 1
  br label %loop

exit:
  ret void
}

; CHECK:      ![[HOISTED]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
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
!8 = !DILocation(line: 5, column: 3, scope: !4, irlayers: !7)
