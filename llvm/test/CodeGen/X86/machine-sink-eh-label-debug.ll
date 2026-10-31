; RUN: llc -stop-after=machine-sink -o - %s | FileCheck %s --check-prefix=INSTRREF
; RUN: llc -experimental-debug-variable-locations=false -stop-after=machine-sink -o - %s | FileCheck %s --check-prefix=DBGVAL

; MachineSink must place the sunk address after the EH label. A debug record
; for that address must remain after its definition.

; INSTRREF-LABEL: name: f
; INSTRREF: bb.1.call:
; INSTRREF: EH_LABEL
; INSTRREF-NEXT: %[[ADDR:[0-9]+]]:gr64 = LEA64r {{.*}}debug-instr-number [[NUM:[0-9]+]]
; INSTRREF-NEXT: DBG_INSTR_REF {{.*}}dbg-instr-ref([[NUM]], 0)

; DBGVAL-LABEL: name: f
; DBGVAL: bb.1.call:
; DBGVAL: EH_LABEL
; DBGVAL-NEXT: %[[ADDR:[0-9]+]]:gr64 = LEA64r
; DBGVAL-NEXT: DBG_VALUE %[[ADDR]]

target triple = "x86_64-unknown-linux-gnu"

declare ptr @g(ptr)
declare i32 @__gxx_personality_v0(...)

define ptr @f(ptr %p, i64 %n, i1 %c) personality ptr @__gxx_personality_v0 !dbg !5 {
entry:
  %a = getelementptr inbounds i32, ptr %p, i64 %n
  br i1 %c, label %call, label %exit

call:
    #dbg_value(ptr %a, !11, !DIExpression(), !12)
  %r = invoke ptr @g(ptr %a)
          to label %exit unwind label %lp, !dbg !12

lp:
  %l = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %l

exit:
  %v = phi ptr [ null, %entry ], [ %r, %call ]
  ret ptr %v
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!11}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!11 = !DILocalVariable(name: "q", scope: !5, file: !1, line: 2, type: !10)
!12 = !DILocation(line: 2, column: 3, scope: !5)
