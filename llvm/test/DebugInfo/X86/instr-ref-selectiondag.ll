; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel -experimental-debug-variable-locations=false \
; RUN:   | FileCheck %s --check-prefix=NORMAL \
; RUN:     --implicit-check-not=debug-instr-number \
; RUN:     --implicit-check-not=DBG_INSTR_REF
; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:     -experimental-debug-variable-locations -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=INSTRREF \
; RUN:     --implicit-check-not=DBG_VALUE
; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:     -experimental-debug-variable-locations -verify-machineinstrs \
; RUN:     -fast-isel \
; RUN:   | FileCheck %s --check-prefix=FASTISEL-INSTRREF \
; RUN:     --implicit-check-not=DBG_VALUE

; NORMAL: ![[SOCKS:[0-9]+]] = !DILocalVariable(name: "socks",
; NORMAL: ![[KNEES:[0-9]+]] = !DILocalVariable(name: "knees",
; INSTRREF: ![[SOCKS:[0-9]+]] = !DILocalVariable(name: "socks",
; INSTRREF: ![[KNEES:[0-9]+]] = !DILocalVariable(name: "knees",
; FASTISEL-INSTRREF: ![[SOCKS:[0-9]+]] = !DILocalVariable(name: "socks",
; FASTISEL-INSTRREF: ![[KNEES:[0-9]+]] = !DILocalVariable(name: "knees",

; Test that SelectionDAG produces DBG_VALUEs normally, but DBG_INSTR_REFs when
; asked.

; NORMAL-LABEL: name: foo

; NORMAL:      %[[REG0:[0-9]+]]:gr32 = ADD32rr
; NORMAL-NEXT: DBG_VALUE %[[REG0]]
; NORMAL-NEXT: %[[REG1:[0-9]+]]:gr32 = ADD32rr
; NORMAL-NEXT: DBG_VALUE %[[REG1]]

; Note that I'm baking in an assumption of one-based ordering here. We could
; capture and check for the instruction numbers, we'd rely on machine verifier
; ensuring there were no duplicates.

; INSTRREF-LABEL: name: foo

; INSTRREF:      ADD32rr
; INSTRREF-SAME: debug-instr-number 1
; INSTRREF-NEXT: DBG_INSTR_REF 1, 0
; INSTRREF-NEXT: ADD32rr
; INSTRREF-SAME: debug-instr-number 2
; INSTRREF-NEXT: DBG_INSTR_REF 2, 0

; Test that fast-isel will produce DBG_INSTR_REFs too.

; FASTISEL-INSTRREF-LABEL: name: foo

; FASTISEL-INSTRREF:      ADD32rr
; FASTISEL-INSTRREF-SAME: debug-instr-number 1
; FASTISEL-INSTRREF-NEXT: DBG_INSTR_REF 1, 0
; FASTISEL-INSTRREF-NEXT: ADD32rr
; FASTISEL-INSTRREF-SAME: debug-instr-number 2
; FASTISEL-INSTRREF-NEXT: DBG_INSTR_REF 2, 0

@glob32 = global i32 0
@glob16 = global i16 0
@glob8 = global i8 0

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

define i32 @foo(i32 %bar, i32 %baz, i32 %qux) !dbg !7 {
entry:
  %0 = add i32 %bar, %baz, !dbg !14
  call void @llvm.dbg.value(metadata i32 %0, metadata !13, metadata !DIExpression()), !dbg !14
  %1 = add i32 %0, %qux
  call void @llvm.dbg.value(metadata i32 %1, metadata !13, metadata !DIExpression()), !dbg !14
  ret i32 %1, !dbg !14
}

; In the code below, isel produces a large number of copies between subregisters
; to represent the gradually decreasing width of the argument. This gets
; optimized away into three stores, but it's an objective of the instruction
; referencing design that COPYs are not numbered: they move values, not define
; them. Test that nothing is numbered, and instead that appropriate
; substitutions with subregister details are recorded.

; NORMAL-LABEL: name: bar

; NORMAL:      DBG_VALUE $rdi
; NORMAL-NEXT: %0:gr64_with_sub_8bit = COPY $rdi
; NORMAL-NEXT: DBG_VALUE %0,
; NORMAL-NEXT: %1:gr32 = COPY %0.sub_32bit,
; NORMAL-NEXT: DBG_VALUE %1
; NORMAL:      %3:gr16 = COPY %0.sub_16bit,
; NORMAL-NEXT: DBG_VALUE %3
; NORMAL:      %5:gr8 = COPY %0.sub_8bit,
; NORMAL-NEXT: DBG_VALUE %5

; INSTRREF-LABEL: name: bar

;; 
; INSTRREF:      debugValueSubstitutions:
; INSTRREF-NEXT: - { srcinst: 2, srcop: 0, dstinst: 1, dstop: 0, subreg: 6 }
; INSTRREF-NEXT: - { srcinst: 4, srcop: 0, dstinst: 3, dstop: 0, subreg: 4 }
; INSTRREF-NEXT: - { srcinst: 6, srcop: 0, dstinst: 5, dstop: 0, subreg: 1 }

;; As a slight inefficiency today, multiple DBG_PHIs are created.

; INSTRREF:      DBG_PHI $rdi, 5
; INSTRREF-NEXT: DBG_PHI $rdi, 3
; INSTRREF-NEXT: DBG_PHI $rdi, 1
;; Allow arguments to be specified by physreg DBG_VALUEs.
; INSTRREF-NEXT: DBG_VALUE $rdi

;; Don't test the location of these instr-refs, only that the three non-argument
;; dbg.values become DBG_INSTR_REFs. We previously checked that these numbers
;; get substituted, with appropriate subregister qualifiers.
; INSTRREF:      DBG_INSTR_REF 2, 0
; INSTRREF:      DBG_INSTR_REF 4, 0
; INSTRREF:      DBG_INSTR_REF 6, 0

;; In fast-isel, we get four DBG_INSTR_REFs (compared to three and one
;; DBG_VALUE with normal isel). We get additional substitutions as a result:

; FASTISEL-INSTRREF:      debugValueSubstitutions:
; FASTISEL-INSTRREF-NEXT: - { srcinst: 3, srcop: 0, dstinst: 2, dstop: 0, subreg: 6 }
; FASTISEL-INSTRREF-NEXT: - { srcinst: 5, srcop: 0, dstinst: 4, dstop: 0, subreg: 6 }
; FASTISEL-INSTRREF-NEXT: - { srcinst: 6, srcop: 0, dstinst: 5, dstop: 0, subreg: 4 }
; FASTISEL-INSTRREF-NEXT  - { srcinst: 8, srcop: 0, dstinst: 7, dstop: 0, subreg: 6 }
; FASTISEL-INSTRREF-NEXT  - { srcinst: 9, srcop: 0, dstinst: 8, dstop: 0, subreg: 4 }
; FASTISEL-INSTRREF-NEXT  - { srcinst: 10, srcop: 0, dstinst: 9, dstop: 0, subreg: 1 }

;; Those substitutions are anchored against these DBG_PHIs:

; FASTISEL-INSTRREF:      DBG_PHI $rdi, 7
; FASTISEL-INSTRREF-NEXT: DBG_PHI $rdi, 4
; FASTISEL-INSTRREF-NEXT: DBG_PHI $rdi, 2
; FASTISEL-INSTRREF-NEXT: DBG_PHI $rdi, 1

; FASTISEL-INSTRREF:      DBG_INSTR_REF 1, 0
; FASTISEL-INSTRREF:      DBG_INSTR_REF 3, 0
; FASTISEL-INSTRREF:      DBG_INSTR_REF 6, 0
; FASTISEL-INSTRREF:      DBG_INSTR_REF 10, 0

define i32 @bar(i64 %bar) !dbg !20 {
entry:
  call void @llvm.dbg.value(metadata i64 %bar, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = trunc i64 %bar to i32, !dbg !22
  call void @llvm.dbg.value(metadata i32 %0, metadata !21, metadata !DIExpression()), !dbg !22
  store i32 %0, i32 *@glob32, !dbg !22
  %1 = trunc i32 %0 to i16, !dbg !22
  call void @llvm.dbg.value(metadata i16 %1, metadata !21, metadata !DIExpression()), !dbg !22
  store i16 %1, i16 *@glob16, !dbg !22
  %2 = trunc i16 %1 to i8, !dbg !22
  call void @llvm.dbg.value(metadata i8 %2, metadata !21, metadata !DIExpression()), !dbg !22
  store i8 %2, i8 *@glob8, !dbg !22
  ret i32 0, !dbg !22
}

; Ensure that we can track copies back to physreg defs, and throw in a subreg
; substitution for fun. The call to @xyzzy defines $rax, which gets copied to
; a VReg, and then truncated by a subreg copy. We should be able to track
; through the copies and walk back to the physreg def, labelling the CALL
; instruction. We should also be able to do this even when the block layout is
; crazily ordered.

; NORMAL-LABEL: name: baz

; NORMAL:      CALL64pcrel32 target-flags(x86-plt) @xyzzy
; NORMAL:      %2:gr64 = COPY $rax,
; NORMAL:      %0:gr64 = COPY %2,
; NORMAL-LABEL: bb.1.slippers:
; NORMAL:      DBG_VALUE %1
; NORMAL-LABEL: bb.2.shoes:
; NORMAL:      %1:gr16 = COPY %0.sub_16bit

; INSTRREF-LABEL: name: baz

; INSTRREF:      debugValueSubstitutions:
; INSTRREF-NEXT:  - { srcinst: 2, srcop: 0, dstinst: 1, dstop: 6, subreg: 4 }

; INSTRREF:      CALL64pcrel32 target-flags(x86-plt) @xyzzy, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, implicit-def $rax, debug-instr-number 1
; INSTRREF:      DBG_INSTR_REF 2, 0

;; Fast-isel produces the same arrangement, a DBG_INSTR_REF pointing back to
;; the call instruction. However: the operand numbers are different (6 for
;; normal isel, 4 for fast-isel). This isn't because of debug-info differences,
;; it's because normal isel implicit-defs the stack registers, and fast-isel
;; does not. The meaning is the same.

; FASTISEL-INSTRREF-LABEL: name: baz

; FASTISEL-INSTRREF:      debugValueSubstitutions:
; FASTISEL-INSTRREF-NEXT:  - { srcinst: 2, srcop: 0, dstinst: 1, dstop: 4, subreg: 4 }

; FASTISEL-INSTRREF:      CALL64pcrel32 target-flags(x86-plt) @xyzzy, csr_64, implicit $rsp, implicit $ssp, implicit-def $rax, debug-instr-number 1
; FASTISEL-INSTRREF:      DBG_INSTR_REF 2, 0

declare i64 @xyzzy()

define i32 @baz() !dbg !30 {
entry:
  %foo = call i64 @xyzzy(), !dbg !32
  br label %shoes

slippers:
  call void @llvm.dbg.value(metadata i16 %moo, metadata !31, metadata !DIExpression()), !dbg !32
  store i16 %moo, i16 *@glob16, !dbg !32
  ret i32 0, !dbg !32

shoes:
  %moo = trunc i64 %foo to i16
  br label %slippers
}

;; Test for dbg.declare of non-stack-slot Values. These turn up with NRVO and 
;; other ABI scenarios where something is technically in memory, but we don't
;; refer to it relative to the stack pointer. We refer to these either with an
;; indirect DBG_VAUE, or a DBG_INSTR_REF with DW_OP_deref prepended.
;;
;; Test an inlined dbg.declare in a different scope + block, to test behaviours
;; where the debug intrinsic isn't in the first block. The normal-mode DBG_VALUE
;; is hoisted into the entry block for that. This is fine because the variable
;; location is never re-assigned. (FIXME: do we scope-trim / fail-to-propagate
;; these hoisted locations later?).

; NORMAL-LABEL: name: qux
;
; NORMAL:      DBG_VALUE $rdi, 0, ![[SOCKS]], !DIExpression(),
; NORMAL-NEXT: %0:gr64 = COPY $rdi
; NORMAL-NEXT: DBG_VALUE %0, 0, ![[SOCKS]], !DIExpression(),
; NORMAL-NEXT: DBG_VALUE %0, 0, ![[KNEES]], !DIExpression(),

;; In instruction referencing mode, the "real" argument becomes a DBG_VALUE,
;; but the hoisted variable location from the inlined scope is a DBG_INSTR_REF.

; INSTRREF-LABEL: name: qux

; INSTRREF:      DBG_PHI $rdi, 1
; INSTRREF-NEXT: DBG_VALUE $rdi, 0, ![[SOCKS]], !DIExpression(),
; INSTRREF-NEXT: %0:gr64 = COPY $rdi
; INSTRREF-NEXT: DBG_INSTR_REF 1, 0, ![[KNEES]], !DIExpression(DW_OP_deref),

; In fast-isel mode, neither variable are hoisted or forwarded to a physreg.

; FASTISEL-INSTRREF-LABEL: name: qux

; FASTISEL-INSTRREF:      DBG_PHI $rdi, 2
; FASTISEL-INSTRREF-NEXT: DBG_PHI $rdi, 1
; FASTISEL-INSTRREF:      DBG_INSTR_REF 1, 0, ![[SOCKS]], !DIExpression(DW_OP_deref),

; FASTISEL-INSTRREF-LABEL: bb.1.lala:
; FASTISEL-INSTRREF:      DBG_INSTR_REF 2, 0, ![[KNEES]], !DIExpression(DW_OP_deref),
declare i64 @cheddar(i32 *%arg)

define void @qux(i32* noalias sret(i32) %agg.result) !dbg !40 {
entry:
  call void @llvm.dbg.declare(metadata i32 *%agg.result, metadata !41, metadata !DIExpression()), !dbg !42
  %foo = call i64 @cheddar(i32 *%agg.result), !dbg !42
  br label %lala

lala:
  call void @llvm.dbg.declare(metadata i32 *%agg.result, metadata !45, metadata !DIExpression()), !dbg !44
  ret void, !dbg !44
}



!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "exprconflict.c", directory: "/home/jmorse")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!13}
!13 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 6, type: !10)
!14 = !DILocation(line: 1, scope: !7)
!20 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!21 = !DILocalVariable(name: "xyzzy", scope: !20, file: !1, line: 6, type: !10)
!22 = !DILocation(line: 1, scope: !20)
!30 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!31 = !DILocalVariable(name: "xyzzy", scope: !30, file: !1, line: 6, type: !10)
!32 = !DILocation(line: 1, scope: !30)
!40 = distinct !DISubprogram(name: "qux", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!41 = !DILocalVariable(name: "socks", scope: !40, file: !1, line: 6, type: !10)
!42 = !DILocation(line: 1, scope: !40)
!43 = distinct !DISubprogram(name: "inlined", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!44 = !DILocation(line: 0, scope: !43, inlinedAt: !42)
!45 = !DILocalVariable(name: "knees", scope: !43, file: !1, line: 6, type: !10)
