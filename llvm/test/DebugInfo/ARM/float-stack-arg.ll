; RUN: llc -mtriple=armv4t-unknown-unknown -start-after=codegenprepare -stop-before=finalize-isel -o - %s | FileCheck %s

; Verify that a stack-referencing DBG_VALUE is emitted for p5 at the start of
; the function.
;
; Reproducer for PR40777.
;
; Based on the following C reproducer:
;
;   float fn1(int p1, int p2, int p3, int p4, float p5) {
;    return p5;
;   }
;
; that was compiled using -O1 -g -S -emit-llvm.
;
; Irrelevant metadata, e.g. information about %p[1-4], has been stripped.

; CHECK: ![[P5:[0-9]*]] = !DILocalVariable(name: "p5"

define arm_aapcscc float @fn1(i32 %p1, i32 %p2, i32 %p3, i32 %p4, float returned %p5) #0 !dbg !7 {
; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: DBG_VALUE %fixed-stack.0, 0, ![[P5]]
entry:
  call void @llvm.dbg.value(metadata float %p5, metadata !17, metadata !DIExpression()), !dbg !18
  ret float %p5, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind readnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "float.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"min_enum_size", i32 4}
!6 = !{!"clang version 9.0.0"}
!7 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !11, !11, !11, !10}
!10 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!17}
!17 = !DILocalVariable(name: "p5", arg: 5, scope: !7, file: !1, line: 1, type: !10)
!18 = !DILocation(line: 1, scope: !7)
!19 = !DILocation(line: 2, scope: !7)
