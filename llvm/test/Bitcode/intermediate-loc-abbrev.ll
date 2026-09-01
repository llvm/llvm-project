; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

;; A DILocation with no `irlayers` should not pay for the operand. The writer
;; has two abbrevs per record kind and picks per record, so an unlayered
;; location comes out one operand shorter.
;;
;; The trailing "/>" is the real check: it proves the operand is absent, not
;; just zero. A VBR field costs its full width even for zero, so always using
;; the wide abbrev would still round-trip while growing every record.

;; Layers sit on the location with no inlinedAt -- the root of the inline chain.
;; Not inlined here, so that is the instruction's own location.
define void @layered() !dbg !8 {
  ret void, !dbg !20
}

;; Inlined, so the layers move to the chain root: !21 and !22 carry none and
;; their inlinedAt targets do. Those targets are enumerated as nodes, which is
;; what puts a layered location into the metadata block.
define void @inlined(i32 %x) !dbg !9 {
  %a = add i32 %x, 1, !dbg !21
  ret void, !dbg !22
}

;; DEBUG_LOC, in the function block: 8 operands layered, 7 unlayered. op3 is the
;; inlinedAt id, so it is 0 on the layered one and set on the other two.
; CHECK-DAG: <DEBUG_LOC abbrevid={{[0-9]+}} op0=2 op1=1 op2={{[0-9]+}} op3=0 op4=0 op5=0 op6=0 op7={{[0-9]+}}/>
; CHECK-DAG: <DEBUG_LOC abbrevid={{[0-9]+}} op0=3 op1=1 op2={{[0-9]+}} op3={{[0-9]+}} op4=0 op5=0 op6=0/>
; CHECK-DAG: <DEBUG_LOC abbrevid={{[0-9]+}} op0=4 op1=1 op2={{[0-9]+}} op3={{[0-9]+}} op4=0 op5=0 op6=0/>

;; LOCATION, in the metadata block: 9 operands layered, 8 unlayered. Instruction
;; locations are never enumerated as nodes, so these are the inlinedAt targets
;; from @inlined -- which is how a layered location reaches this record kind.
; CHECK-DAG: <LOCATION abbrevid={{[0-9]+}} op0=0 op1=10 op2=1 op3={{[0-9]+}} op4=0 op5=0 op6=0 op7=0 op8={{[0-9]+}}/>
; CHECK-DAG: <LOCATION abbrevid={{[0-9]+}} op0=0 op1=11 op2=1 op3={{[0-9]+}} op4=0 op5=0 op6=0 op7=0/>

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "layered", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = distinct !DISubprogram(name: "inlined", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}

!14 = !DIFile(filename: "intermediate.tileir", directory: ".")
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 100, column: 10, file: !14, kind: "TileIR")

!20 = !DILocation(line: 2, column: 1, scope: !8, irlayers: !30)
!21 = !DILocation(line: 3, column: 1, scope: !9, inlinedAt: !40)
!22 = !DILocation(line: 4, column: 1, scope: !9, inlinedAt: !41)

;; Chain roots. !40 carries the layers for the instruction inlined at it; !41
;; has none, to cover the narrow LOCATION.
!40 = !DILocation(line: 10, column: 1, scope: !9, irlayers: !30)
!41 = !DILocation(line: 11, column: 1, scope: !9)
