; Test that a local SafeStack moves to the unsafe stack keeps a frame-index
; location: its dbg.declare is anchored on a stack slot holding the unsafe
; stack's base pointer, not on the base pointer's SSA value, which would make
; it a DBG_VALUE that lasts only as long as the register holding that value.
; A local described by dbg.values of its alloca is anchored on the same slot,
; so each DBG_VALUE is of the slot rather than of a register.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -stop-after finalize-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O2 -stop-after finalize-isel %s -o - | FileCheck %s

; Built from the following source with SafeStack, and simplified:
; void capture(int *);
; void f(void) {
;   int xxx[10];
;   int yyy[10];
;   capture(xxx);
;   capture(yyy);
; }
; with yyy's dbg.declare written as a dbg.value of its alloca.

; CHECK:      ![[XXX:[0-9]+]] = !DILocalVariable(name: "xxx"
; CHECK:      ![[YYY:[0-9]+]] = !DILocalVariable(name: "yyy"
; CHECK:      stack:
; CHECK:        name: unsafe_stack_base,
; CHECK:        debug-info-variable: '![[XXX]]', debug-info-expression: '!DIExpression(DW_OP_deref, DW_OP_constu, 48, DW_OP_minus)'
; CHECK-NOT:  DBG_VALUE {{.*}}![[XXX]]
; CHECK:      DBG_VALUE %stack.{{[0-9]+}}.unsafe_stack_base, $noreg, ![[YYY]], !DIExpression(DW_OP_deref, DW_OP_constu, 96, DW_OP_minus, DW_OP_deref)
; CHECK-NOT:  DBG_VALUE {{.*}}![[XXX]]

define void @f() #0 !dbg !6 {
entry:
  %xxx = alloca [10 x i32], align 16
  %yyy = alloca [10 x i32], align 16
    #dbg_declare(ptr %xxx, !10, !DIExpression(), !15)
  call void @capture(ptr %xxx), !dbg !16
    #dbg_value(ptr %yyy, !18, !DIExpression(DW_OP_deref), !19)
  call void @capture(ptr %yyy), !dbg !19
  ret void, !dbg !17
}

declare void @capture(ptr)

attributes #0 = { nounwind safestack uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "f.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!10 = !DILocalVariable(name: "xxx", scope: !6, file: !1, line: 3, type: !11)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 320, elements: !13)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DISubrange(count: 10)
!15 = !DILocation(line: 3, column: 7, scope: !6)
!16 = !DILocation(line: 4, column: 3, scope: !6)
!17 = !DILocation(line: 5, column: 1, scope: !6)
!18 = !DILocalVariable(name: "yyy", scope: !6, file: !1, line: 4, type: !11)
!19 = !DILocation(line: 4, column: 3, scope: !6)
