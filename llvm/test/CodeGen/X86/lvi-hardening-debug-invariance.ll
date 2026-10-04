; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown-linux-gnu -x86-lvi-load-no-cbranch < %s | FileCheck %s

; PR224484: A leading debug instruction (such as DBG_VALUE or DBG_INSTR_REF) at
; the start of a basic block must not become an ordinary vertex in the LVI gadget
; graph. When getGadgetGraph adds the first instruction in each block, it should
; skip debug-only instructions so that debug info does not alter fence placement.

define i32 @nested_lvi(ptr %pointer_slot, ptr %control, i32 %limit) #0 !dbg !4 {
; CHECK-LABEL: nested_lvi:
; CHECK:       # %bb.0:
; CHECK:         lfence
; CHECK:       .LBB0_2: # %inner.header
; CHECK:         movq (%rdi), %r{{[0-9]+}}
; CHECK:         lfence
; CHECK-NOT:   .LBB0_{{[0-9]+}}: # %inner.debug.join
; CHECK-NOT:     lfence
entry:
  br label %outer.header

outer.header:
  %outer.i = phi i32 [ 0, %entry ], [ %outer.next, %outer.latch ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %outer.latch ]
  %outer.more = icmp slt i32 %outer.i, %limit
  br i1 %outer.more, label %inner.header, label %exit

inner.header:
  %loaded.pointer = load ptr, ptr %pointer_slot, align 8
  %first.control = load volatile i32, ptr %control, align 4
  %take.first = icmp eq i32 %first.control, 1
  br i1 %take.first, label %inner.exit.a, label %inner.check

inner.check:
  %second.control = load volatile i32, ptr %control, align 4
  %take.second = icmp eq i32 %second.control, 2
  br i1 %take.second, label %inner.exit.b, label %inner.latch

inner.latch:
  call void asm sideeffect "", "~{memory}"()
  br label %inner.header

inner.exit.a:
  br label %inner.debug.join

inner.exit.b:
  br label %inner.debug.join

inner.debug.join:
  call void @llvm.dbg.value(metadata ptr %loaded.pointer, metadata !6, metadata !DIExpression()), !dbg !7
  br label %inner.merge

inner.merge:
  %value = load i32, ptr %loaded.pointer, align 4
  %sum.next = add i32 %sum, %value
  br label %outer.latch

outer.latch:
  %outer.next = add nuw nsw i32 %outer.i, 1
  br label %outer.header

exit:
  ret i32 %sum
}

attributes #0 = { "target-features"="+lvi-load-hardening" }

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "lvi-debug-empty.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "nested_lvi", scope: !1, file: !1, line: 1, type: !5, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !DILocalVariable(name: "ghost", scope: !4, file: !1, line: 1)
!7 = !DILocation(line: 1, scope: !4)
