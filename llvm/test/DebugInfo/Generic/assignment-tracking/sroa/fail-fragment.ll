; RUN: opt -passes=sroa -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Check that a dbg.assign for a promoted variable becomes a kill location if
;; it used a fragment that can't be split (the first check directive below).
;; NOTE: If createFragmentExpression gets smarter it may be necessary to create
;; a new test case.
;; In some cases a dbg.assign with a poison value is replaced with a non-poison
;; value. This needs reworking, but as a stop-gap we need to ensure this
;; doesn't cause invalid expressions to be created. Check we don't do it when
;; the expression uses more than one location operand (DW_OP_arg n).

; CHECK: if.then:
; CHECK: dbg.value(metadata i32 poison, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32))
;; FIXME: The value below should be poison. See https://reviews.llvm.org/D147431#4245260.
; CHECK: dbg.value(metadata i32 %{{.*}}, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32))

; CHECK: if.else:
; CHECK: dbg.value(metadata i32 2, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32))
; CHECK: dbg.value(metadata i32 0, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32))

; CHECK: if.inner:
; CHECK: call void @llvm.dbg.value(metadata i32 poison, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value))

; CHECK: end:
; CHECK: dbg.value(metadata i32 %{{.*}}, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32))

declare i64 @get_i64()

define internal fastcc i64 @fun() !dbg !18 {
entry:
  %codepoint = alloca i64, align 4, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i1 poison, metadata !15, metadata !DIExpression(), metadata !27, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
   %0 = call i64 @get_i64() #2
   %1 = add i64 %0, 1
   %cmp = icmp ugt i64 %1, 100
   br i1 %cmp, label %if.then, label %if.else

if.then:
  store i64 %1, ptr %codepoint, align 4, !DIAssignID !25
  call void @llvm.dbg.assign(metadata i64 %1, metadata !15, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value), metadata !25, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
  br label %end

if.else:
  store i64 2, ptr %codepoint, align 4, !DIAssignID !28
  call void @llvm.dbg.assign(metadata i32 2, metadata !15, metadata !DIExpression(), metadata !28, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
  br i1 %cmp, label %end, label %if.inner

if.inner:
  store i32 3, ptr %codepoint, align 4, !DIAssignID !29
  call void @llvm.dbg.assign(metadata i32 poison, metadata !15, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), metadata !29, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
  br label %end

end:
  %r = load i32, ptr %codepoint
  %rr = zext i32 %r to i64
  ret i64 %rr
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.cpp", directory: "/")
!2 = !{}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!15 = !DILocalVariable(name: "codepoint", scope: !18, file: !1, line: 10, type: !24)
!18 = distinct !DISubprogram(name: "fun", linkageName: "fun", scope: !1, file: !1, line: 4, type: !19, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!19 = distinct !DISubroutineType(types: !2)
!24 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!25 = distinct !DIAssignID()
!26 = !DILocation(line: 0, scope: !18)
!27 = distinct !DIAssignID()
!28 = distinct !DIAssignID()
!29 = distinct !DIAssignID()
