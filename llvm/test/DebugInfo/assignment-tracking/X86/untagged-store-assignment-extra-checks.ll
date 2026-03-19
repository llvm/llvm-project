; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Similarly to untagged-store-assignment-outside-variable.ll this test checks
;; that out of bounds stores that have no DIAssignID are interpreted correctly
;; (see inline comments and checks). Hand-written IR.

target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @a(i32)

define dso_local void @b() local_unnamed_addr !dbg !14 {
entry:
  %c = alloca [4 x i16], align 8, !DIAssignID !24
  %arrayidx = getelementptr inbounds [4 x i16], ptr %c, i64 0, i64 2
  call void @llvm.dbg.assign(metadata i1 undef, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !24, metadata ptr %arrayidx, metadata !DIExpression()), !dbg !26

;; Set variable value to create a non-stack DBG_VALUE.
; CHECK: DBG_VALUE 0, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 128, 32)
  call void @llvm.dbg.assign(metadata i64 0, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !29, metadata ptr %c, metadata !DIExpression()), !dbg !26

;; Trim assignment that leaks outside alloca (upper 32 bits don't fit inside %c alloca).
; CHECK: DBG_VALUE %stack.0.c, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 128, 32)
  store i64 1, ptr %arrayidx, align 4
;; Set variable value (use a call to prevent eliminating redundant DBG_VALUEs).
; CHECK: DBG_VALUE 10, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 128, 32)
  call void @a(i32 1)
  call void @llvm.dbg.assign(metadata i64 10, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !29, metadata ptr %c, metadata !DIExpression()), !dbg !26

;; Trim assignment that doesn't align with fragment start and leaks outside
;; alloca (16 bit offset from fragment start, upper 48 bits don't fit inside %c
;; alloca).
; CHECK: DBG_VALUE %stack.0.c, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 6, DW_OP_deref, DW_OP_LLVM_fragment, 144, 16)
  %arrayidx1 = getelementptr inbounds [4 x i16], ptr %c, i64 0, i64 3
  store i64 2, ptr %arrayidx1, align 4
;; Set variable value (use a call to prevent eliminating redundant DBG_VALUEs).
; CHECK: DBG_VALUE 20, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 128, 32)
  call void @a(i32 2)
  call void @llvm.dbg.assign(metadata i64 20, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !29, metadata ptr %c, metadata !DIExpression()), !dbg !26

;; Negative accesses are skipped.
  %arrayidx2 = getelementptr inbounds [4 x i16], ptr %c, i64 0, i64 -1
  store i128 3, ptr %arrayidx2, align 4
;; Set variable value (use a call to prevent eliminating redundant DBG_VALUEs).
; CHECK: DBG_VALUE 30, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 128, 32)
  call void @a(i32 3)
  call void @llvm.dbg.assign(metadata i64 30, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !29, metadata ptr %c, metadata !DIExpression()), !dbg !26

;; Skip assignment outside base variable fragment.
  store i32 4, ptr %c, align 4
;; Set variable value (use a call to prevent eliminating redundant DBG_VALUEs).
; CHECK: DBG_VALUE 40, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 128, 32)
  call void @a(i32 4)
  call void @llvm.dbg.assign(metadata i64 40, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !29, metadata ptr %c, metadata !DIExpression()), !dbg !26

;; Trim partial overlap (lower 32 bits of store don't intersect base fragment
;; and upper 64 bits don't actually fit inside the alloca).
  store i128 5, ptr %c, align 4
; CHECK: DBG_VALUE %stack.0.c, $noreg, ![[#]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 128, 32)
;; Set variable value (use a call to prevent eliminating redundant DBG_VALUEs).
; CHECK: DBG_VALUE 50, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 128, 32)
  call void @a(i32 5)
  call void @llvm.dbg.assign(metadata i64 50, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32), metadata !29, metadata ptr %c, metadata !DIExpression()), !dbg !26

  ret void
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !12}
!llvm.ident = !{!13}

!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!13 = !{!"clang version 17.0.0"}
!14 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !15, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18}
!18 = !DILocalVariable(name: "c", scope: !14, file: !3, line: 3, type: !19)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 160, elements: !20)
!20 = !{!21}
!21 = !DISubrange(count: 2)
!24 = distinct !DIAssignID()
!26 = !DILocation(line: 0, scope: !14)
!29 = distinct !DIAssignID()
