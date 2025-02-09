; RUN: opt -passes=sroa -S %s -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=sroa -S %s -o - | FileCheck %s

;; Check that a dbg.assign for a promoted variable becomes a kill location if
;; it used an arglist.

; CHECK: if.then:
; CHECK-NEXT: #dbg_value(i32 poison,

; CHECK: if.else:
; CHECK-NEXT: #dbg_value(i32 2,

declare i8 @get_i8()

define internal fastcc i64 @fun() !dbg !18 {
entry:
  %codepoint = alloca i32, align 4, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i1 poison, metadata !15, metadata !DIExpression(), metadata !27, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
  %0 = call i8 @get_i8() #2
  %1 = and i8 %0, 15
  %and14 = zext i8 %1 to i32
  %shl = shl nuw nsw i32 %and14, 12
  %2 = call i8 @get_i8() #2
  %3 = and i8 %2, 63
  %and21 = zext i8 %3 to i32
  %or22 = or i32 %shl, %and21
  %cmp = icmp ugt i32 %or22, 57343
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 %or22, ptr %codepoint, align 4, !DIAssignID !25
  call void @llvm.dbg.assign(metadata !DIArgList(i32 %shl, i32 %and21), metadata !15, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_or, DW_OP_stack_value), metadata !25, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
  br label %end

if.else:
  store i32 2, ptr %codepoint, align 4, !DIAssignID !28
  call void @llvm.dbg.assign(metadata i32 2, metadata !15, metadata !DIExpression(), metadata !28, metadata ptr %codepoint, metadata !DIExpression()), !dbg !26
  br label %end

end:
  %r = load i32, ptr %codepoint
  %rr = sext i32 %r to i64
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
!24 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!25 = distinct !DIAssignID()
!26 = !DILocation(line: 0, scope: !18)
!27 = distinct !DIAssignID()
!28 = distinct !DIAssignID()
