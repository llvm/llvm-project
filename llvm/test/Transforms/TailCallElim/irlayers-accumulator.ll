; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

;; The accumulator copy inserted before the return does the accumulation's work
;; one more time, so it gets a line 0 location in the function's scope that
;; keeps the accumulation's layers.

define i32 @factorial(i32 %x) !dbg !4 {
; CHECK-LABEL: define i32 @factorial(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK:       then:
; CHECK:         %{{.*}} = mul i32 %accumulator.tr, %x.tr, !dbg ![[ACC:[0-9]+]]
; CHECK:       else:
; CHECK-NEXT:    %accumulator.ret.tr = mul i32 %accumulator.tr, 1, !dbg ![[RET:[0-9]+]]
; CHECK-NEXT:    ret i32 %accumulator.ret.tr
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %then, label %else

then:
  %dec = add i32 %x, -1
  %recurse = call i32 @factorial(i32 %dec), !dbg !9
  %accumulate = mul i32 %recurse, %x, !dbg !8
  ret i32 %accumulate

else:
  ret i32 1
}

; CHECK:      ![[ACC]] = !DILocation(line: 3, column: 10, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
; CHECK-NEXT: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-NEXT: ![[LAYER]] = !DILayerLoc(line: 100,
; CHECK:      ![[RET]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "factorial", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!6 = !DILayerLoc(line: 100, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!8 = !DILocation(line: 3, column: 10, scope: !4, irlayers: !7)
!9 = !DILocation(line: 3, column: 14, scope: !4)
