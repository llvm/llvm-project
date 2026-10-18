; RUN: llc -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -experimental-debug-variable-locations=false \
; RUN:   -stop-before=phi-node-elimination -o - %s | FileCheck %s --check-prefix=INPUT-DBG-VALUE
; RUN: llc -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -experimental-debug-variable-locations=false \
; RUN:   -stop-after=phi-node-elimination -o - %s | FileCheck %s --check-prefix=DBG-VALUE
; RUN: llc -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -experimental-debug-variable-locations=true \
; RUN:   -stop-before=phi-node-elimination -o - %s | FileCheck %s --check-prefix=INPUT-INSTR-REF
; RUN: llc -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -experimental-debug-variable-locations=true \
; RUN:   -stop-after=phi-node-elimination -o - %s | FileCheck %s --check-prefix=INSTR-REF
; RUN: opt -strip-debug < %s | llc -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -stop-after=phi-node-elimination -o - | FileCheck %s --check-prefix=NO-DEBUG

; Debug instructions between a PHI and an EH label must not cause the
; destination copy to be inserted before the label.

; INPUT-DBG-VALUE-LABEL: bb.{{[0-9]+}}.lpad (landing-pad):
; INPUT-DBG-VALUE: PHI
; INPUT-DBG-VALUE-NEXT: DBG_VALUE
; INPUT-DBG-VALUE-NEXT: DBG_LABEL
; INPUT-DBG-VALUE-NEXT: EH_LABEL

; DBG-VALUE-LABEL: bb.{{[0-9]+}}.lpad (landing-pad):
; DBG-VALUE: EH_LABEL
; DBG-VALUE-NEXT: [[COPY:%[0-9]+]]:gr32 = COPY
; DBG-VALUE-NEXT: DBG_VALUE [[COPY]]
; DBG-VALUE-NEXT: DBG_LABEL

; INPUT-INSTR-REF-LABEL: bb.{{[0-9]+}}.lpad (landing-pad):
; INPUT-INSTR-REF: PHI
; INPUT-INSTR-REF-NEXT: DBG_INSTR_REF
; INPUT-INSTR-REF-NEXT: DBG_LABEL
; INPUT-INSTR-REF-NEXT: EH_LABEL

; INSTR-REF-LABEL: bb.{{[0-9]+}}.lpad (landing-pad):
; INSTR-REF: EH_LABEL
; INSTR-REF-NEXT: [[COPY:%[0-9]+]]:gr32 = COPY
; INSTR-REF-NEXT: DBG_INSTR_REF
; INSTR-REF-NEXT: DBG_LABEL

; NO-DEBUG-LABEL: bb.{{[0-9]+}}.lpad (landing-pad):
; NO-DEBUG: EH_LABEL
; NO-DEBUG-NEXT: [[COPY:%[0-9]+]]:gr32 = COPY

declare i32 @f()
declare i32 @__gxx_personality_v0(...)

define i32 @phi(i32 %x) personality ptr @__gxx_personality_v0 !dbg !4 {
entry:
  %a = invoke i32 @f()
          to label %cont unwind label %lpad

cont:
  %b = invoke i32 @f()
          to label %exit unwind label %lpad

exit:
  ret i32 %b

lpad:
  %v = phi i32 [ %x, %entry ], [ %a, %cont ]
  #dbg_label(!10, !8)
  #dbg_value(i32 %v, !7, !DIExpression(), !8)
  %exn = landingpad { ptr, i32 }
          cleanup
  ret i32 %v
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "phi-elimination-debug-invariance.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "phi", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocalVariable(name: "v", scope: !4, file: !1, line: 2, type: !9)
!8 = !DILocation(line: 2, column: 1, scope: !4)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILabel(scope: !4, name: "lpad", file: !1, line: 2)
