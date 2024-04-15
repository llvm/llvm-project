; RUN: opt -passes=instcombine -S -o - < %s | FileCheck %s
; RUN: opt -passes=instcombine -S -o - < %s --try-experimental-debuginfo-iterators | FileCheck %s

; When the 'int Four = Two;' is sunk into the 'case 0:' block,
; the debug value for 'Three' is set incorrectly to 'poison'.

;  1 int One = 0;
;  2 int Two = 0;
;  3 int test() {
;  4   int Three = 0;
;  5   int Four = Two;
;  6   switch (One) {
;  7   case 0:
;  8     Three = Four;
;  9     break;
; 10   case 2:
; 11     Three = 4;
; 12     break;
; 13   }
; 14   return Three;
; 15 }

; CHECK-LABEL: sw.bb:
; CHECK: %[[REG:[0-9]+]] = load i32, ptr @"?Two{{.*}}
; CHECK: call void @llvm.dbg.value(metadata i32 %[[REG]], metadata ![[DBG1:[0-9]+]], {{.*}}
; CHECK: call void @llvm.dbg.value(metadata i32 %[[REG]], metadata ![[DBG2:[0-9]+]], {{.*}}
; CHECK-DAG: ![[DBG1]] = !DILocalVariable(name: "Four"{{.*}})
; CHECK-DAG: ![[DBG2]] = !DILocalVariable(name: "Three"{{.*}})

@"?One@@3HA" = dso_local global i32 0, align 4, !dbg !0
@"?Two@@3HA" = dso_local global i32 0, align 4, !dbg !5

define dso_local noundef i32 @"?test@@YAHXZ"() !dbg !15 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !20
  %0 = load i32, ptr @"?Two@@3HA", align 4, !dbg !21
  call void @llvm.dbg.value(metadata i32 %0, metadata !22, metadata !DIExpression()), !dbg !20
  %1 = load i32, ptr @"?One@@3HA", align 4, !dbg !23
  switch i32 %1, label %sw.epilog [
    i32 0, label %sw.bb
    i32 2, label %sw.bb1
  ], !dbg !23

sw.bb:                                            ; preds = %entry
  call void @llvm.dbg.value(metadata i32 %0, metadata !19, metadata !DIExpression()), !dbg !20
  br label %sw.epilog, !dbg !24

sw.bb1:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 4, metadata !19, metadata !DIExpression()), !dbg !20
  br label %sw.epilog, !dbg !26

sw.epilog:                                        ; preds = %sw.bb1, %sw.bb, %entry
  %Three.0 = phi i32 [ 0, %entry ], [ 4, %sw.bb1 ], [ %0, %sw.bb ], !dbg !20
  call void @llvm.dbg.value(metadata i32 %Three.0, metadata !19, metadata !DIExpression()), !dbg !20
  ret i32 %Three.0, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "One", linkageName: "?One@@3HA", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 18.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "Two", linkageName: "?Two@@3HA", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"CodeView", i32 1}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 2}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 1, !"MaxTLSAlign", i32 65536}
!14 = !{!"clang version 18.0.0"}
!15 = distinct !DISubprogram(name: "test", linkageName: "?test@@YAHXZ", scope: !3, file: !3, line: 3, type: !16, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!7}
!18 = !{}
!19 = !DILocalVariable(name: "Three", scope: !15, file: !3, line: 4, type: !7)
!20 = !DILocation(line: 0, scope: !15)
!21 = !DILocation(line: 5, scope: !15)
!22 = !DILocalVariable(name: "Four", scope: !15, file: !3, line: 5, type: !7)
!23 = !DILocation(line: 6, scope: !15)
!24 = !DILocation(line: 9, scope: !25)
!25 = distinct !DILexicalBlock(scope: !15, file: !3, line: 6)
!26 = !DILocation(line: 12, scope: !25)
!27 = !DILocation(line: 14, scope: !15)
