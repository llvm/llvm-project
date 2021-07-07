; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:   | FileCheck %s --check-prefix=NORMAL \
; RUN:     --implicit-check-not=debug-instr-number \
; RUN:     --implicit-check-not=DBG_INSTR_REF
; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:     -experimental-debug-variable-locations -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=INSTRREF \
; RUN:     --implicit-check-not=DBG_VALUE

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

@glob32 = global i32 0
@glob16 = global i16 0
@glob8 = global i8 0

declare void @llvm.dbg.value(metadata, metadata, metadata)

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

; INSTRREF:      CALL64pcrel32 target-flags(x86-plt) @xyzzy, {{.*}} debug-instr-number 1
; INSTRREF:      DBG_INSTR_REF 2, 0

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
