; RUN: opt < %s -passes=asan -asan-use-after-return=never -S | \
; RUN:   llc -O0 -filetype=obj - -o - | \
; RUN:   llvm-dwarfdump - | FileCheck %s

; RUN: opt --try-experimental-debuginfo-iterators < %s -passes=asan -asan-use-after-return=never -S | \
; RUN:   llc -O0 -filetype=obj - -o - | \
; RUN:   llvm-dwarfdump - | FileCheck %s

; For this test case, ASan used to produce IR which resulted in the following
; DWARF (at -O0):
;
;   DW_TAG_subprogram
;     DW_AT_low_pc    (0x0000000000000000)
;     DW_AT_high_pc   (0x00000000000000f1)
;
;     DW_TAG_variable
;       DW_AT_location        (0x00000000
;         [0x0000000000000014,  0x000000000000006d): DW_OP_breg0 RAX+32
;         [0x000000000000006d,  0x00000000000000a4): DW_OP_breg7 RSP+16, DW_OP_deref, DW_OP_plus_uconst 0x20
;         [0x00000000000000a6,  0x00000000000000ef): DW_OP_breg7 RSP+16, DW_OP_deref, DW_OP_plus_uconst 0x20)
;
; The DWARF produced for the original ObjC code that motivated this test case
; was actually not as nice! In that example, the location list ranges didn't
; intersect with the ranges of the parent lexical scope. But recreating that
; exactly requires playing tricks to get LiveDebugValue's lexical dominance
; check to kill a variable range early, and it isn't strictly necessary to show
; the problem here.
;
; The problem is that we shouldn't get a location list at all. The instruction
; selector should recognize that we have an "alloca" in the entry block, and
; just make the fixed location available in the whole function. We now produce
; the correct DWARF, namely:

; CHECK: DW_TAG_variable
; CHECK-NEXT:  DW_AT_location (DW_OP_breg7 RSP+32, DW_OP_plus_uconst 0x20)

target triple = "x86_64-apple-macosx10.10.0"

declare void @escape(ptr)

; Function Attrs: sanitize_address
define ptr @foo(i1 %cond) #0 !dbg !6 {
entry:
  %a1 = alloca ptr, !dbg !12
  call void @escape(ptr %a1), !dbg !13
  br i1 %cond, label %l1, label %l2, !dbg !14

l1:                                               ; preds = %entry
  ret ptr null, !dbg !15

l2:                                               ; preds = %entry
  call void @llvm.dbg.declare(metadata ptr %a1, metadata !11, metadata !DIExpression()), !dbg !16
  %p = load ptr, ptr %a1, !dbg !16
  ret ptr %p, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

attributes #0 = { sanitize_address }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "redu.ll", directory: "/")
!2 = !{}
!3 = !{i32 6}
!4 = !{i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!11}
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 5, type: !10)
!12 = !DILocation(line: 1, column: 1, scope: !6)
!13 = !DILocation(line: 2, column: 1, scope: !6)
!14 = !DILocation(line: 3, column: 1, scope: !6)
!15 = !DILocation(line: 4, column: 1, scope: !6)
!16 = !DILocation(line: 5, column: 1, scope: !6)
!17 = !DILocation(line: 6, column: 1, scope: !6)
