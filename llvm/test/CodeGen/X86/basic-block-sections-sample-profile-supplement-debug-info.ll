; In every function below, block #1 (containing `call i32 @baz()`) has 0
; Propeller samples, so it is split into a `.cold` section unless the
; supplemental sample profile gives block #1 a non-zero weight.
;
; `sample.prof` places supplemental samples at the exact offsets (including
; offsets wrapped mod 65536 such as `(0 - 10) & 0xffff = 65526`) that an
; unguarded lookup for block #1's debug locations would match:
;   - `no_subprogram` and `line_zero_subprogram`: function has no `DISubprogram`
;     or has `line: 0` on its `DISubprogram` -> block #1 must not be
;     supplemented and stays in `.cold`.
;   - `line_zero_inst`: block #1's instructions have `line: 0` on their
;     `DILocation` -> block #1 must not be supplemented and stays in `.cold`.

; RUN: rm -rf %t && split-file %s %t
; RUN: llc < %t/test.ll -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t/propeller.prof -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t/sample.prof | \
; RUN: FileCheck %s

;--- propeller.prof
v1
f no_subprogram
g 0:100,1:0,2:100 1:0,2:0 2:100
f line_zero_subprogram
g 0:100,1:0,2:100 1:0,2:0 2:100
f line_zero_inst
g 0:100,1:0,2:100 1:0,2:0 2:100

;--- sample.prof
no_subprogram:2000:1000
 3: 5
line_zero_subprogram:2000:1000
 0: 5
 4: 5
line_zero_inst:2000:1000
 65526: 5

;--- test.ll
define void @no_subprogram(i1 zeroext %0) nounwind {
  br i1 %0, label %4, label %2

2:
  %3 = call i32 @baz(), !dbg !7
  br label %4

4:
  ret void
}

define void @line_zero_subprogram(i1 zeroext %0) nounwind !dbg !8 {
  br i1 %0, label %4, label %2, !dbg !9

2:
  %3 = call i32 @baz(), !dbg !10
  br label %4, !dbg !9

4:
  ret void, !dbg !11
}

define void @line_zero_inst(i1 zeroext %0) nounwind !dbg !12 {
  br i1 %0, label %4, label %2, !dbg !13

2:
  %3 = call i32 @baz(), !dbg !14
  br label %4, !dbg !14

4:
  ret void, !dbg !15
}

declare i32 @baz()

; CHECK-LABEL: no_subprogram:
; CHECK:       .section        .text.split.no_subprogram,"ax",@progbits
; CHECK-LABEL: no_subprogram.cold:

; CHECK-LABEL: line_zero_subprogram:
; CHECK:       .section        .text.split.line_zero_subprogram,"ax",@progbits
; CHECK-LABEL: line_zero_subprogram.cold:

; CHECK-LABEL: line_zero_inst:
; CHECK:       .section        .text.split.line_zero_inst,"ax",@progbits
; CHECK-LABEL: line_zero_inst.cold:

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test.cc", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "other", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 4, column: 1, scope: !6)
!8 = distinct !DISubprogram(name: "line_zero_subprogram", scope: !1, file: !1, line: 0, type: !4, scopeLine: 0, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocation(line: 0, column: 0, scope: !8)
!10 = !DILocation(line: 4, column: 1, scope: !8)
!11 = !DILocation(line: 5, column: 1, scope: !8)
!12 = distinct !DISubprogram(name: "line_zero_inst", scope: !1, file: !1, line: 10, type: !4, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 10, column: 1, scope: !12)
!14 = !DILocation(line: 0, column: 0, scope: !12)
!15 = !DILocation(line: 15, column: 1, scope: !12)
