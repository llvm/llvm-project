; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10"

%struct.a = type { i32, %struct.a* }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i8* (%struct.a*)* @bar to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i8* @bar(%struct.a* %myvar) nounwind optsize noinline ssp !dbg !9 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.a* %myvar, i64 0, metadata !8, metadata !DIExpression()), !dbg !DILocation(scope: !9)
  %0 = getelementptr inbounds %struct.a, %struct.a* %myvar, i64 0, i32 0, !dbg !28 ; <i32*> [#uses=1]
  %1 = load i32, i32* %0, align 8, !dbg !28            ; <i32> [#uses=1]
  tail call void @foo(i32 %1) nounwind optsize noinline ssp, !dbg !28
  %2 = bitcast %struct.a* %myvar to i8*, !dbg !30 ; <i8*> [#uses=1]
  ret i8* %2, !dbg !30
}

declare void @foo(i32) nounwind optsize noinline ssp

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!38}

!0 = !DIGlobalVariable(name: "ret", line: 7, isLocal: false, isDefinition: true, scope: !1, file: !1, type: !3)
!1 = !DIFile(filename: "foo.c", directory: "/tmp/")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 1, file: !36, enums: !37, retainedTypes: !37, subprograms: !32, globals: !31, imports:  !37)
!3 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DILocalVariable(name: "x", line: 12, arg: 1, scope: !5, file: !1, type: !3)
!5 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 13, file: !36, scope: !1, type: !6, variables: !33)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !3}
!8 = !DILocalVariable(name: "myvar", line: 17, arg: 1, scope: !9, file: !1, type: !13)
!9 = distinct !DISubprogram(name: "bar", linkageName: "bar", line: 17, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 17, file: !36, scope: !1, type: !10, variables: !34)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !36, scope: !1, baseType: null)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !36, scope: !1, baseType: !14)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", line: 2, size: 128, align: 64, file: !36, scope: !1, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "c", line: 3, size: 32, align: 32, file: !36, scope: !14, baseType: !3)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "d", line: 4, size: 64, align: 64, offset: 64, file: !36, scope: !14, baseType: !13)
!18 = !DILocalVariable(name: "argc", line: 22, arg: 1, scope: !19, file: !1, type: !3)
!19 = distinct !DISubprogram(name: "main", linkageName: "main", line: 22, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 22, file: !36, scope: !1, type: !20, variables: !35)
!20 = !DISubroutineType(types: !21)
!21 = !{!3, !3, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !36, scope: !1, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !36, scope: !1, baseType: !24)
!24 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!25 = !DILocalVariable(name: "argv", line: 22, arg: 2, scope: !19, file: !1, type: !22)
!26 = !DILocalVariable(name: "e", line: 23, scope: !27, file: !1, type: !14)
!27 = distinct !DILexicalBlock(line: 22, column: 0, file: !36, scope: !19)
!28 = !DILocation(line: 18, scope: !29)
!29 = distinct !DILexicalBlock(line: 17, column: 0, file: !36, scope: !9)
!30 = !DILocation(line: 19, scope: !29)
!31 = !{!0}
!32 = !{!5, !9, !19}
!33 = !{!4}
!34 = !{!8}
!35 = !{!18, !25, !26}
!36 = !DIFile(filename: "foo.c", directory: "/tmp/")
!37 = !{}

; The variable bar:myvar changes registers after the first movq.
; It is cobbered by popq %rbx
; CHECK: movq
; CHECK-NEXT: [[LABEL:Ltmp[0-9]*]]
; CHECK: .loc	1 19 0
; CHECK: popq
; CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]


; CHECK: Ldebug_loc0:
; CHECK-NEXT: [[SET1:.*]] = Lfunc_begin0-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET1]]
; CHECK-NEXT: [[SET2:.*]] = [[LABEL]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET2]]
; CHECK-NEXT: .short  1     ## Loc expr size
; CHECK-NEXT: .byte   85
; CHECK-NEXT: [[SET3:.*]] = [[LABEL]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET3]]
; CHECK-NEXT: [[SET4:.*]] = [[CLOBBER]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET4]]
; CHECK-NEXT: .short  1     ## Loc expr size
; CHECK-NEXT: .byte   83
!38 = !{i32 1, !"Debug Info Version", i32 3}
