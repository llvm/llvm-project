; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

; The instructions the involution (parity) accumulator inserts carry no debug
; location: attributing them to the eliminated fneg's line would step backwards
; into the recursive branch. The `{{$}}` anchors assert no trailing !dbg.

define float @test_involution_debugloc(float %x) !dbg !5 {
; CHECK-LABEL: define float @test_involution_debugloc(
; CHECK:       neg:
; CHECK:         [[PARITY_FLIP_TR:%.*]] = xor i1 %parity.tr, true{{$}}
; CHECK:       base:
; CHECK:         [[INVOLUTION_VAL_TR:%.*]] = fneg float %x.tr{{$}}
; CHECK:         [[ACCUMULATOR_RET_TR:%.*]] = select i1 %parity.tr, float [[INVOLUTION_VAL_TR]], float %x.tr{{$}}
;
entry:
  %isneg = fcmp olt float %x, 0.000000e+00, !dbg !8
  br i1 %isneg, label %neg, label %base, !dbg !8

neg:
  %nx = fneg float %x, !dbg !9
  %r = call float @test_involution_debugloc(float %nx), !dbg !9
  %rn = fneg float %r, !dbg !9
  ret float %rn, !dbg !9

base:
  ret float %x, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "test_involution_debugloc", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 2, column: 1, scope: !5)
!9 = !DILocation(line: 3, column: 1, scope: !5)
!10 = !DILocation(line: 4, column: 1, scope: !5)
