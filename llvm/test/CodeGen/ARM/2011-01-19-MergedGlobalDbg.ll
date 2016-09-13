; RUN: llc -arm-global-merge -global-merge-group-by-use=false -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@x1 = internal global i8 1, align 1, !dbg !13
@x2 = internal global i8 1, align 1, !dbg !14
@x3 = internal global i8 1, align 1, !dbg !15
@x4 = internal global i8 1, align 1, !dbg !16
@x5 = global i8 1, align 1, !dbg !17

; Check debug info output for merged global.
; DW_AT_location
; 0x03 DW_OP_addr
; 0x.. .long __MergedGlobals
; 0x10 DW_OP_constu
; 0x.. offset
; 0x22 DW_OP_plus

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}} "x1"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x5> 03 [[ADDR:.. .. .. ..]]   )
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}} "x2"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x7> 03 [[ADDR]] 23 01  )

define zeroext i8 @get1(i8 zeroext %a) nounwind optsize !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !10, metadata !DIExpression()), !dbg !30
  %0 = load i8, i8* @x1, align 4, !dbg !30
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !11, metadata !DIExpression()), !dbg !30
  store i8 %a, i8* @x1, align 4, !dbg !30
  ret i8 %0, !dbg !31
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define zeroext i8 @get2(i8 zeroext %a) nounwind optsize !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !18, metadata !DIExpression()), !dbg !32
  %0 = load i8, i8* @x2, align 4, !dbg !32
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !19, metadata !DIExpression()), !dbg !32
  store i8 %a, i8* @x2, align 4, !dbg !32
  ret i8 %0, !dbg !33
}

define zeroext i8 @get3(i8 zeroext %a) nounwind optsize !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !21, metadata !DIExpression()), !dbg !34
  %0 = load i8, i8* @x3, align 4, !dbg !34
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !22, metadata !DIExpression()), !dbg !34
  store i8 %a, i8* @x3, align 4, !dbg !34
  ret i8 %0, !dbg !35
}

define zeroext i8 @get4(i8 zeroext %a) nounwind optsize !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !24, metadata !DIExpression()), !dbg !36
  %0 = load i8, i8* @x4, align 4, !dbg !36
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !25, metadata !DIExpression()), !dbg !36
  store i8 %a, i8* @x4, align 4, !dbg !36
  ret i8 %0, !dbg !37
}

define zeroext i8 @get5(i8 zeroext %a) nounwind optsize !dbg !9 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !27, metadata !DIExpression()), !dbg !38
  %0 = load i8, i8* @x5, align 4, !dbg !38
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !28, metadata !DIExpression()), !dbg !38
  store i8 %a, i8* @x5, align 4, !dbg !38
  ret i8 %0, !dbg !39
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!49}

!0 = distinct !DISubprogram(name: "get1", linkageName: "get1", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 4, file: !47, scope: !1, type: !3, variables: !42)
!1 = !DIFile(filename: "foo.c", directory: "/tmp/")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2369.8)", isOptimized: true, emissionKind: FullDebug, file: !47, enums: !48, retainedTypes: !48, globals: !41, imports:  !48)
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "_Bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!6 = distinct !DISubprogram(name: "get2", linkageName: "get2", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 7, file: !47, scope: !1, type: !3, variables: !43)
!7 = distinct !DISubprogram(name: "get3", linkageName: "get3", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 10, file: !47, scope: !1, type: !3, variables: !44)
!8 = distinct !DISubprogram(name: "get4", linkageName: "get4", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 13, file: !47, scope: !1, type: !3, variables: !45)
!9 = distinct !DISubprogram(name: "get5", linkageName: "get5", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 16, file: !47, scope: !1, type: !3, variables: !46)
!10 = !DILocalVariable(name: "a", line: 4, arg: 1, scope: !0, file: !1, type: !5)
!11 = !DILocalVariable(name: "b", line: 4, scope: !12, file: !1, type: !5)
!12 = distinct !DILexicalBlock(line: 4, column: 0, file: !47, scope: !0)
!13 = !DIGlobalVariable(name: "x1", line: 3, isLocal: true, isDefinition: true, scope: !1, file: !1, type: !5)
!14 = !DIGlobalVariable(name: "x2", line: 6, isLocal: true, isDefinition: true, scope: !1, file: !1, type: !5)
!15 = !DIGlobalVariable(name: "x3", line: 9, isLocal: true, isDefinition: true, scope: !1, file: !1, type: !5)
!16 = !DIGlobalVariable(name: "x4", line: 12, isLocal: true, isDefinition: true, scope: !1, file: !1, type: !5)
!17 = !DIGlobalVariable(name: "x5", line: 15, isLocal: false, isDefinition: true, scope: !1, file: !1, type: !5)
!18 = !DILocalVariable(name: "a", line: 7, arg: 1, scope: !6, file: !1, type: !5)
!19 = !DILocalVariable(name: "b", line: 7, scope: !20, file: !1, type: !5)
!20 = distinct !DILexicalBlock(line: 7, column: 0, file: !47, scope: !6)
!21 = !DILocalVariable(name: "a", line: 10, arg: 1, scope: !7, file: !1, type: !5)
!22 = !DILocalVariable(name: "b", line: 10, scope: !23, file: !1, type: !5)
!23 = distinct !DILexicalBlock(line: 10, column: 0, file: !47, scope: !7)
!24 = !DILocalVariable(name: "a", line: 13, arg: 1, scope: !8, file: !1, type: !5)
!25 = !DILocalVariable(name: "b", line: 13, scope: !26, file: !1, type: !5)
!26 = distinct !DILexicalBlock(line: 13, column: 0, file: !47, scope: !8)
!27 = !DILocalVariable(name: "a", line: 16, arg: 1, scope: !9, file: !1, type: !5)
!28 = !DILocalVariable(name: "b", line: 16, scope: !29, file: !1, type: !5)
!29 = distinct !DILexicalBlock(line: 16, column: 0, file: !47, scope: !9)
!30 = !DILocation(line: 4, scope: !0)
!31 = !DILocation(line: 4, scope: !12)
!32 = !DILocation(line: 7, scope: !6)
!33 = !DILocation(line: 7, scope: !20)
!34 = !DILocation(line: 10, scope: !7)
!35 = !DILocation(line: 10, scope: !23)
!36 = !DILocation(line: 13, scope: !8)
!37 = !DILocation(line: 13, scope: !26)
!38 = !DILocation(line: 16, scope: !9)
!39 = !DILocation(line: 16, scope: !29)
!41 = !{!13, !14, !15, !16, !17}
!42 = !{!10, !11}
!43 = !{!18, !19}
!44 = !{!21, !22}
!45 = !{!24, !25}
!46 = !{!27, !28}
!47 = !DIFile(filename: "foo.c", directory: "/tmp/")
!48 = !{}
!49 = !{i32 1, !"Debug Info Version", i32 3}
