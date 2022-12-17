; RUN: opt -S -passes=loop-idiom -mtriple=systemz-unknown -mcpu=z13 %s | FileCheck %s

; CHECK: @llvm.ctlz.i32

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0
define dso_local i32 @CeilLog2(i32 %arg) local_unnamed_addr #1 !dbg !38 {
bb:
  %tmp4 = add i32 %arg, -1, !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !45
  %tmp71 = icmp eq i32 %tmp4, 0, !dbg !45
  br i1 %tmp71, label %bb13, label %bb8.preheader, !dbg !48

bb8.preheader:                                    ; preds = %bb
  br label %bb8, !dbg !49

bb8:                                              ; preds = %bb8.preheader, %bb8
  %tmp2.03 = phi i32 [ %tmp12, %bb8 ], [ 0, %bb8.preheader ]
  %tmp1.02 = phi i32 [ %tmp10, %bb8 ], [ %tmp4, %bb8.preheader ]
  call void @llvm.dbg.value(metadata i32 %tmp2.03, metadata !44, metadata !DIExpression()), !dbg !45
  %tmp10 = lshr i32 %tmp1.02, 1, !dbg !49
  %tmp12 = add nuw nsw i32 %tmp2.03, 1, !dbg !51
  call void @llvm.dbg.value(metadata i32 %tmp12, metadata !44, metadata !DIExpression()), !dbg !45
  %tmp7 = icmp eq i32 %tmp10, 0, !dbg !45
  br i1 %tmp7, label %bb13.loopexit, label %bb8, !dbg !48, !llvm.loop !52

bb13.loopexit:                                    ; preds = %bb8
  %tmp12.lcssa = phi i32 [ %tmp12, %bb8 ], !dbg !51
  br label %bb13, !dbg !54

bb13:                                             ; preds = %bb13.loopexit, %bb
  %tmp2.0.lcssa = phi i32 [ 0, %bb ], [ %tmp12.lcssa, %bb13.loopexit ], !dbg !55
  call void @llvm.dbg.value(metadata i32 %tmp2.0.lcssa, metadata !44, metadata !DIExpression()), !dbg !45
  ret i32 %tmp2.0.lcssa, !dbg !54
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone speculatable "target-cpu"="z13" }
attributes #1 = { norecurse nounwind readnone "target-cpu"="z13" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (ijonpan@m35lp38.lnxne.boe:llvm/llvm-dev-2/tools/clang a87ff88c6466fbedd6281513b9480a2cad6c08c8) (llvm/llvm-dev-2 922a3b1b3254bf3310c467e880a5419c1e13c87f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "configfile.c", directory: "/home/ijonpan/minispec-2006/spec-llvm/464.h264ref/build")
!2 = !{}
!4 = !DIFile(filename: "./global.h", directory: "/home/ijonpan/minispec-2006/spec-llvm/464.h264ref/build")
!5 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{i32 1, !"wchar_size", i32 4}
!38 = distinct !DISubprogram(name: "CeilLog2", scope: !1, file: !1, line: 599, type: !39, scopeLine: 600, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !41)
!39 = !DISubroutineType(types: !40)
!40 = !{!5, !5}
!41 = !{!42, !43, !44}
!42 = !DILocalVariable(name: "uiVal", arg: 1, scope: !38, file: !1, line: 599, type: !5)
!43 = !DILocalVariable(name: "uiTmp", scope: !38, file: !1, line: 601, type: !5)
!44 = !DILocalVariable(name: "uiRet", scope: !38, file: !1, line: 602, type: !5)
!45 = !DILocation(line: 601, column: 25, scope: !38)
!48 = !DILocation(line: 604, column: 3, scope: !38)
!49 = !DILocation(line: 606, column: 11, scope: !50)
!50 = distinct !DILexicalBlock(scope: !38, file: !1, line: 605, column: 3)
!51 = !DILocation(line: 607, column: 10, scope: !50)
!52 = distinct !{!52, !48, !53}
!53 = !DILocation(line: 608, column: 3, scope: !38)
!54 = !DILocation(line: 609, column: 3, scope: !38)
!55 = !DILocation(line: 0, scope: !38)
