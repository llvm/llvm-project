; RUN: llc -start-after=codegenprepare -stop-before=finalize-isel -o - < %s \
; RUN:     -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -start-after=codegenprepare -stop-before=finalize-isel -o - < %s \
; RUN:     -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF
;
;; Test for correct placement of DBG_VALUE, which in PR40427 is placed before
;; the load instruction it refers to. The circumstance replicated here is where
;; two instructions in a row, trunc and add, begin with no-op Copy{To,From}Reg
;; SDNodes that produce no instructions.
;; The DBG_VALUE instruction should come immediately after the load instruction
;; because the truncate is optimised out, and the DBG_VALUE should be placed
;; in front of the first instruction that occurs after the dbg.value.
;

; CHECK: ![[DBGVAR:[0-9]+]] = !DILocalVariable(name: "bees",

target triple = "x86_64-unknown-linux-gnu"

define i16 @lolwat(i1 %spoons, ptr %bees, i16 %yellow, ptr %more) {
entry:
  br i1 %spoons, label %trueb, label %falseb
trueb:
  br label %block
falseb:
  br label %block
block:
; CHECK:      [[PHIREG:%[0-9]+]]:gr64 = PHI %6, %bb.2, %4, %bb.1
; CHECK-NEXT: [[LOADR:%[0-9]+]]:gr16 = MOV16rm %0,
; INSTRREF-SAME: debug-instr-number 1
; DBGVALUE-NEXT: DBG_VALUE [[LOADR]], $noreg, ![[DBGVAR]]
; INSTRREF-NEXT: DBG_INSTR_REF ![[DBGVAR]], {{.+}}, dbg-instr-ref(1, 0)
; CHECK-NEXT: %{{[0-9]+}}:gr32 = IMPLICIT_DEF
  %foo = phi ptr[%bees, %trueb], [%more, %falseb]
  %ret = load i32, ptr %foo, !dbg !6
  %cast = trunc i32 %ret to i16, !dbg !6
  call void @llvm.dbg.value(metadata i16 %cast, metadata !1, metadata !DIExpression()), !dbg !6
  %orly2 = add i16 %yellow, 1
  br label %bb1
bb1:
  %cheese = add i16 %orly2, %cast
  ret i16 %cheese, !dbg !6
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!2}
!1 = !DILocalVariable(name: "bees", scope: !5, type: null)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "beards", isOptimized: true, runtimeVersion: 4, emissionKind: FullDebug)
!3 = !DIFile(filename: "bees.cpp", directory: "")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "nope", scope: !2, file: !3, line: 1, unit: !2)
!6 = !DILocation(line: 0, scope: !5)
!7 = !DILocalVariable(name: "flannel", scope: !5, type: null)
