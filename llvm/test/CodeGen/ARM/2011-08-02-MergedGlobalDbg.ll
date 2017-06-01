; RUN: llc -arm-global-merge -global-merge-group-by-use=false -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

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
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x7> 03 [[ADDR]] 23 04  )

source_filename = "test/CodeGen/ARM/2011-08-02-MergedGlobalDbg.ll"
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.7.0"

@x1 = internal unnamed_addr global i32 1, align 4, !dbg !0
@x2 = internal unnamed_addr global i32 2, align 4, !dbg !6
@x3 = internal unnamed_addr global i32 3, align 4
@x4 = internal unnamed_addr global i32 4, align 4
@x5 = global i32 0, align 4

; Function Attrs: nounwind optsize ssp
define i32 @get1(i32 %a) #0 !dbg !10 {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !14, metadata !17), !dbg !18
  %1 = load i32, i32* @x1, align 4, !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !15, metadata !17), !dbg !19
  store i32 %a, i32* @x1, align 4, !dbg !19
  ret i32 %1, !dbg !19
}

; Function Attrs: nounwind optsize ssp

define i32 @get2(i32 %a) #0 !dbg !20 {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !22, metadata !17), !dbg !25
  %1 = load i32, i32* @x2, align 4, !dbg !26
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !23, metadata !17), !dbg !26
  store i32 %a, i32* @x2, align 4, !dbg !26
  ret i32 %1, !dbg !26
}

; Function Attrs: nounwind optsize ssp

define i32 @get3(i32 %a) #0 !dbg !27 {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !29, metadata !17), !dbg !32
  %1 = load i32, i32* @x3, align 4, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !30, metadata !17), !dbg !33
  store i32 %a, i32* @x3, align 4, !dbg !33
  ret i32 %1, !dbg !33
}

; Function Attrs: nounwind optsize ssp

define i32 @get4(i32 %a) #0 !dbg !34 {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !36, metadata !17), !dbg !39
  %1 = load i32, i32* @x4, align 4, !dbg !40
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !37, metadata !17), !dbg !40
  store i32 %a, i32* @x4, align 4, !dbg !40
  ret i32 %1, !dbg !40
}

; Function Attrs: nounwind optsize ssp

define i32 @get5(i32 %a) #0 !dbg !41 {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !43, metadata !17), !dbg !46
  %1 = load i32, i32* @x5, align 4, !dbg !47
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !44, metadata !17), !dbg !47
  store i32 %a, i32* @x5, align 4, !dbg !47
  ret i32 %1, !dbg !47
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind optsize ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "x1", scope: !2, file: !3, line: 4, type: !8, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "ss3.c", directory: "/private/tmp")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "x2", scope: !2, file: !3, line: 7, type: !8, isLocal: true, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "get1", scope: !3, file: !3, line: 5, type: !11, isLocal: false, isDefinition: true, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{!8}
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !3, line: 5, type: !8)
!15 = !DILocalVariable(name: "b", scope: !16, file: !3, line: 5, type: !8)
!16 = distinct !DILexicalBlock(scope: !10, file: !3, line: 5, column: 19)
!17 = !DIExpression()
!18 = !DILocation(line: 5, column: 16, scope: !10)
!19 = !DILocation(line: 5, column: 32, scope: !16)
!20 = distinct !DISubprogram(name: "get2", scope: !3, file: !3, line: 8, type: !11, isLocal: false, isDefinition: true, scopeLine: 8, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !21)
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "a", arg: 1, scope: !20, file: !3, line: 8, type: !8)
!23 = !DILocalVariable(name: "b", scope: !24, file: !3, line: 8, type: !8)
!24 = distinct !DILexicalBlock(scope: !20, file: !3, line: 8, column: 17)
!25 = !DILocation(line: 8, column: 14, scope: !20)
!26 = !DILocation(line: 8, column: 29, scope: !24)
!27 = distinct !DISubprogram(name: "get3", scope: !3, file: !3, line: 11, type: !11, isLocal: false, isDefinition: true, scopeLine: 11, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !28)
!28 = !{!29, !30}
!29 = !DILocalVariable(name: "a", arg: 1, scope: !27, file: !3, line: 11, type: !8)
!30 = !DILocalVariable(name: "b", scope: !31, file: !3, line: 11, type: !8)
!31 = distinct !DILexicalBlock(scope: !27, file: !3, line: 11, column: 19)
!32 = !DILocation(line: 11, column: 16, scope: !27)
!33 = !DILocation(line: 11, column: 32, scope: !31)
!34 = distinct !DISubprogram(name: "get4", scope: !3, file: !3, line: 14, type: !11, isLocal: false, isDefinition: true, scopeLine: 14, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !35)
!35 = !{!36, !37}
!36 = !DILocalVariable(name: "a", arg: 1, scope: !34, file: !3, line: 14, type: !8)
!37 = !DILocalVariable(name: "b", scope: !38, file: !3, line: 14, type: !8)
!38 = distinct !DILexicalBlock(scope: !34, file: !3, line: 14, column: 19)
!39 = !DILocation(line: 14, column: 16, scope: !34)
!40 = !DILocation(line: 14, column: 32, scope: !38)
!41 = distinct !DISubprogram(name: "get5", scope: !3, file: !3, line: 17, type: !11, isLocal: false, isDefinition: true, scopeLine: 17, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !42)
!42 = !{!43, !44}
!43 = !DILocalVariable(name: "a", arg: 1, scope: !41, file: !3, line: 17, type: !8)
!44 = !DILocalVariable(name: "b", scope: !45, file: !3, line: 17, type: !8)
!45 = distinct !DILexicalBlock(scope: !41, file: !3, line: 17, column: 19)
!46 = !DILocation(line: 17, column: 16, scope: !41)
!47 = !DILocation(line: 17, column: 32, scope: !45)

