; RUN: llc -arm-global-merge -global-merge-group-by-use=false -filetype=obj < %s | llvm-dwarfdump -debug-info --name=x1 --name=x2 -v - | FileCheck %s

source_filename = "test/CodeGen/ARM/2011-01-19-MergedGlobalDbg.ll"
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@x1 = internal global i8 1, align 1, !dbg !0
@x2 = internal global i8 1, align 1, !dbg !4
@x3 = internal global i8 1, align 1, !dbg !6
@x4 = internal global i8 1, align 1, !dbg !8
@x5 = global i8 1, align 1, !dbg !10

; CHECK: DW_TAG_variable
; CHECK:    DW_AT_name {{.*}} "x1"
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (DW_OP_addr [[ADDR:0x[0-9a-fA-F]+]])

; CHECK: DW_TAG_variable
; CHECK:    DW_AT_name {{.*}} "x2"
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (DW_OP_addr [[ADDR]], DW_OP_plus_uconst 0x1)

; Function Attrs: nounwind optsize
define zeroext i8 @get1(i8 zeroext %a) #0 !dbg !16 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, metadata !20, metadata !23), !dbg !24
  %0 = load i8, ptr @x1, align 4, !dbg !24
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !21, metadata !23), !dbg !24
  store i8 %a, ptr @x1, align 4, !dbg !24
  ret i8 %0, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nounwind optsize
define zeroext i8 @get2(i8 zeroext %a) #0 !dbg !26 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, metadata !28, metadata !23), !dbg !31
  %0 = load i8, ptr @x2, align 4, !dbg !31
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !29, metadata !23), !dbg !31
  store i8 %a, ptr @x2, align 4, !dbg !31
  ret i8 %0, !dbg !32
}

; Function Attrs: nounwind optsize

define zeroext i8 @get3(i8 zeroext %a) #0 !dbg !33 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, metadata !35, metadata !23), !dbg !38
  %0 = load i8, ptr @x3, align 4, !dbg !38
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !36, metadata !23), !dbg !38
  store i8 %a, ptr @x3, align 4, !dbg !38
  ret i8 %0, !dbg !39
}

; Function Attrs: nounwind optsize

define zeroext i8 @get4(i8 zeroext %a) #0 !dbg !40 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, metadata !42, metadata !23), !dbg !45
  %0 = load i8, ptr @x4, align 4, !dbg !45
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !43, metadata !23), !dbg !45
  store i8 %a, ptr @x4, align 4, !dbg !45
  ret i8 %0, !dbg !46
}

; Function Attrs: nounwind optsize

define zeroext i8 @get5(i8 zeroext %a) #0 !dbg !47 {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, metadata !49, metadata !23), !dbg !52
  %0 = load i8, ptr @x5, align 4, !dbg !52
  tail call void @llvm.dbg.value(metadata i8 %0, metadata !50, metadata !23), !dbg !52
  store i8 %a, ptr @x5, align 4, !dbg !52
  ret i8 %0, !dbg !53
}

attributes #0 = { nounwind optsize }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!12}
!llvm.module.flags = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x1", scope: !2, file: !2, line: 3, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "foo.c", directory: "/tmp/")
!3 = !DIBasicType(name: "_Bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "x2", scope: !2, file: !2, line: 6, type: !3, isLocal: true, isDefinition: true)
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "x3", scope: !2, file: !2, line: 9, type: !3, isLocal: true, isDefinition: true)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = !DIGlobalVariable(name: "x4", scope: !2, file: !2, line: 12, type: !3, isLocal: true, isDefinition: true)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = !DIGlobalVariable(name: "x5", scope: !2, file: !2, line: 15, type: !3, isLocal: false, isDefinition: true)
!12 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2369.8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !13, retainedTypes: !13, globals: !14, imports: !13)
!13 = !{}
!14 = !{!0, !4, !6, !8, !10}
!15 = !{i32 1, !"Debug Info Version", i32 3}
!16 = distinct !DISubprogram(name: "get1", linkageName: "get1", scope: !2, file: !2, line: 4, type: !17, isLocal: false, isDefinition: true, scopeLine: 4, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!3, !3}
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "a", arg: 1, scope: !16, file: !2, line: 4, type: !3)
!21 = !DILocalVariable(name: "b", scope: !22, file: !2, line: 4, type: !3)
!22 = distinct !DILexicalBlock(scope: !16, file: !2, line: 4)
!23 = !DIExpression()
!24 = !DILocation(line: 4, scope: !16)
!25 = !DILocation(line: 4, scope: !22)
!26 = distinct !DISubprogram(name: "get2", linkageName: "get2", scope: !2, file: !2, line: 7, type: !17, isLocal: false, isDefinition: true, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !27)
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "a", arg: 1, scope: !26, file: !2, line: 7, type: !3)
!29 = !DILocalVariable(name: "b", scope: !30, file: !2, line: 7, type: !3)
!30 = distinct !DILexicalBlock(scope: !26, file: !2, line: 7)
!31 = !DILocation(line: 7, scope: !26)
!32 = !DILocation(line: 7, scope: !30)
!33 = distinct !DISubprogram(name: "get3", linkageName: "get3", scope: !2, file: !2, line: 10, type: !17, isLocal: false, isDefinition: true, scopeLine: 10, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !34)
!34 = !{!35, !36}
!35 = !DILocalVariable(name: "a", arg: 1, scope: !33, file: !2, line: 10, type: !3)
!36 = !DILocalVariable(name: "b", scope: !37, file: !2, line: 10, type: !3)
!37 = distinct !DILexicalBlock(scope: !33, file: !2, line: 10)
!38 = !DILocation(line: 10, scope: !33)
!39 = !DILocation(line: 10, scope: !37)
!40 = distinct !DISubprogram(name: "get4", linkageName: "get4", scope: !2, file: !2, line: 13, type: !17, isLocal: false, isDefinition: true, scopeLine: 13, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !41)
!41 = !{!42, !43}
!42 = !DILocalVariable(name: "a", arg: 1, scope: !40, file: !2, line: 13, type: !3)
!43 = !DILocalVariable(name: "b", scope: !44, file: !2, line: 13, type: !3)
!44 = distinct !DILexicalBlock(scope: !40, file: !2, line: 13)
!45 = !DILocation(line: 13, scope: !40)
!46 = !DILocation(line: 13, scope: !44)
!47 = distinct !DISubprogram(name: "get5", linkageName: "get5", scope: !2, file: !2, line: 16, type: !17, isLocal: false, isDefinition: true, scopeLine: 16, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !48)
!48 = !{!49, !50}
!49 = !DILocalVariable(name: "a", arg: 1, scope: !47, file: !2, line: 16, type: !3)
!50 = !DILocalVariable(name: "b", scope: !51, file: !2, line: 16, type: !3)
!51 = distinct !DILexicalBlock(scope: !47, file: !2, line: 16)
!52 = !DILocation(line: 16, scope: !47)
!53 = !DILocation(line: 16, scope: !51)

