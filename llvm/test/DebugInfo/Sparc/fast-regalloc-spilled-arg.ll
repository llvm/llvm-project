; RUN: llc -O0 -relocation-model=pic -mtriple=sparc64-unknown-linux-gnu \
; RUN:   -stop-after=regallocfast %s -o - | FileCheck %s

; rustc describes an argument passed by reference with #dbg_declare on the
; incoming pointer. The pointer is live out of the entry block, so the fast
; register allocator spills it right after its def. Every DBG_VALUE for the
; argument after that spill must use the stack slot: one left on %i0 would
; override the spill's DBG_VALUE, and %i0 is clobbered by the GOT setup
; before the branch, leaving a hole in the location list.

; CHECK-LABEL: name: foo
; CHECK:       DBG_VALUE $i0, 0, ![[A:[0-9]+]], !DIExpression()
; CHECK:       STXri %stack.[[SLOT:[0-9]+]], 0, killed $i0
; CHECK-NEXT:  DBG_VALUE %stack.[[SLOT]], 0, ![[A]], !DIExpression(DW_OP_deref)
; CHECK-NOT:   DBG_VALUE {{.*}}$i0, 0, ![[A]],
; CHECK:       GETPCX
; CHECK-NOT:   DBG_VALUE {{.*}}$i0, 0, ![[A]],
; CHECK:       BPXCC

@g = external global i64

define i64 @foo(ptr %a, i64 %i) !dbg !10 {
entry:
    #dbg_declare(ptr %a, !14, !DIExpression(), !16)
  %c = icmp ult i64 %i, 64, !dbg !17
  br i1 %c, label %in, label %out, !dbg !17

in:
  %p = getelementptr inbounds i64, ptr %a, i64 %i, !dbg !17
  %v = load i64, ptr %p, align 8, !dbg !17
  ret i64 %v, !dbg !18

out:
  %x = load i64, ptr @g, align 8, !dbg !17
  ret i64 %x, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "hand-written", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.c", directory: "/tmp")
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 4096, elements: !7)
!7 = !{!8}
!8 = !DISubrange(count: 64)
!9 = !DISubroutineType(types: !{!5, !6, !5})
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !9, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !{})
!14 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 3, type: !6)
!16 = !DILocation(line: 3, scope: !10)
!17 = !DILocation(line: 4, column: 9, scope: !10)
!18 = !DILocation(line: 4, column: 2, scope: !10)
