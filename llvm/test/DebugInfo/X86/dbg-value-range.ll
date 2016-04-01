; RUN: llc -mtriple=x86_64-apple-darwin10 < %s | FileCheck %s

%struct.a = type { i32 }

define i32 @bar(%struct.a* nocapture %b) nounwind ssp !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.a* %b, i64 0, metadata !6, metadata !DIExpression()), !dbg !13
  %tmp1 = getelementptr inbounds %struct.a, %struct.a* %b, i64 0, i32 0, !dbg !14
  %tmp2 = load i32, i32* %tmp1, align 4, !dbg !14
  tail call void @llvm.dbg.value(metadata i32 %tmp2, i64 0, metadata !11, metadata !DIExpression()), !dbg !14
  %call = tail call i32 (...) @foo(i32 %tmp2) nounwind , !dbg !18
  %add = add nsw i32 %tmp2, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare i32 @foo(...) 

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24}

!0 = distinct !DISubprogram(name: "bar", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !22, scope: !1, type: !3, variables: !21)
!1 = !DIFile(filename: "bar.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 122997)", isOptimized: true, emissionKind: FullDebug, file: !22, enums: !23, retainedTypes: !23, subprograms: !20, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "b", line: 5, arg: 1, scope: !0, file: !1, type: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !2, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", line: 1, size: 32, align: 32, file: !22, scope: !2, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "c", line: 2, size: 32, align: 32, file: !22, scope: !1, baseType: !5)
!11 = !DILocalVariable(name: "x", line: 6, scope: !12, file: !1, type: !5)
!12 = distinct !DILexicalBlock(line: 5, column: 22, file: !22, scope: !0)
!13 = !DILocation(line: 5, column: 19, scope: !0)
!14 = !DILocation(line: 6, column: 14, scope: !12)
!18 = !DILocation(line: 7, column: 2, scope: !12)
!19 = !DILocation(line: 8, column: 2, scope: !12)
!20 = !{!0}
!21 = !{!6, !11}
!22 = !DIFile(filename: "bar.c", directory: "/private/tmp")
!23 = !{}

; Check that variable bar:b value range is appropriately truncated in debug info.
; The variable is in %rdi which is clobbered by 'movl %ebx, %edi'
; Here Ltmp7 is the end of the location range.

;CHECK: .loc	1 7 2
;CHECK: movl
;CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]

;CHECK:Ldebug_loc0:
;CHECK-NEXT: Lset{{.*}} =
;CHECK-NEXT:	.quad
;CHECK-NEXT: [[CLOBBER_OFF:Lset.*]] = [[CLOBBER]]-{{.*}}
;CHECK-NEXT:	.quad	[[CLOBBER_OFF]]
;CHECK-NEXT:  .short 1 ## Loc expr size
;CHECK-NEXT:	.byte	85 ## DW_OP_reg
;CHECK-NEXT:	.quad	0
;CHECK-NEXT:	.quad	0
!24 = !{i32 1, !"Debug Info Version", i32 3}
