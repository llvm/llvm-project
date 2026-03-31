; RUN: opt -disable-output -passes="debug-record-count-pass,delete-debug-record-pass,debug-record-count-pass" %s 2>&1 | FileCheck %s

; CHECK:Function: max
; CHECK:        #dbg_values : 3
; CHECK:        #dbg_declare: 0
; CHECK:        #dbg_assign : 0

; CHECK:Function: max
; CHECK:        #dbg_values : 0
; CHECK:        #dbg_declare: 0
; CHECK:        #dbg_assign : 0

; CHECK:Function: sum
; CHECK:        #dbg_values : 5
; CHECK:        #dbg_declare: 0
; CHECK:        #dbg_assign : 0

; CHECK:Function: sum
; CHECK:        #dbg_values : 0
; CHECK:        #dbg_declare: 0
; CHECK:        #dbg_assign : 0

; CHECK:Function: swap
; CHECK:        #dbg_values : 3
; CHECK:        #dbg_declare: 0
; CHECK:        #dbg_assign : 0

; CHECK:Function: swap
; CHECK:        #dbg_values : 0
; CHECK:        #dbg_declare: 0
; CHECK:        #dbg_assign : 0

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @max(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 !dbg !13 {
entry:
    #dbg_value(i32 %a, !18, !DIExpression(), !21)
    #dbg_value(i32 %b, !19, !DIExpression(), !21)
  %a.b = tail call i32 @llvm.smax.i32(i32 %a, i32 %b)
    #dbg_value(i32 %a.b, !20, !DIExpression(), !21)
  ret i32 %a.b, !dbg !22
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @sum(i32 noundef %n) local_unnamed_addr #0 !dbg !23 {
entry:
    #dbg_value(i32 %n, !27, !DIExpression(), !31)
    #dbg_value(i32 0, !28, !DIExpression(), !31)
    #dbg_value(i32 0, !29, !DIExpression(), !32)
  %cmp4 = icmp sgt i32 %n, 0, !dbg !33
  br i1 %cmp4, label %for.cond.cleanup.loopexit, label %for.cond.cleanup, !dbg !35

for.cond.cleanup.loopexit:                        ; preds = %entry
    #dbg_value(i32 poison, !29, !DIExpression(), !32)
    #dbg_value(i32 poison, !28, !DIExpression(), !31)
  %0 = add nsw i32 %n, -1, !dbg !36
  %1 = zext nneg i32 %0 to i33, !dbg !36
  %2 = add nsw i32 %n, -2, !dbg !36
  %3 = zext i32 %2 to i33, !dbg !36
  %4 = mul i33 %1, %3, !dbg !36
  %5 = lshr i33 %4, 1, !dbg !36
  %6 = trunc nuw i33 %5 to i32, !dbg !36
  %7 = add i32 %n, %6, !dbg !36
  %8 = add i32 %7, -1, !dbg !36
  br label %for.cond.cleanup, !dbg !37

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %total.0.lcssa = phi i32 [ 0, %entry ], [ %8, %for.cond.cleanup.loopexit ], !dbg !31
  ret i32 %total.0.lcssa, !dbg !37
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @swap(ptr noundef captures(none) %a, ptr noundef captures(none) %b) local_unnamed_addr #1 !dbg !38 {
entry:
    #dbg_value(ptr %a, !43, !DIExpression(), !46)
    #dbg_value(ptr %b, !44, !DIExpression(), !46)
  %0 = load i32, ptr %a, align 4, !dbg !47, !tbaa !9
    #dbg_value(i32 %0, !45, !DIExpression(), !46)
  %1 = load i32, ptr %b, align 4, !dbg !48, !tbaa !9
  store i32 %1, ptr %a, align 4, !dbg !49, !tbaa !9
  store i32 %0, ptr %b, align 4, !dbg !50, !tbaa !9
  ret void, !dbg !51
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}
!llvm.errno.tbaa = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git (https://github.com/minazivic-htec/llvm-project.git 9232377d061824bb8a5893102ee71fd728220cd4)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/mina/llvm-test", checksumkind: CSK_MD5, checksum: "d7b6a9b88dc7b0be7ebc31254e9c7872")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!8 = !{!"clang version 23.0.0git (https://github.com/minazivic-htec/llvm-project.git 9232377d061824bb8a5893102ee71fd728220cd4)"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = distinct !DISubprogram(name: "max", scope: !1, file: !1, line: 1, type: !14, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17, keyInstructions: true)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !16, !16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18, !19, !20}
!18 = !DILocalVariable(name: "a", arg: 1, scope: !13, file: !1, line: 1, type: !16)
!19 = !DILocalVariable(name: "b", arg: 2, scope: !13, file: !1, line: 1, type: !16)
!20 = !DILocalVariable(name: "result", scope: !13, file: !1, line: 2, type: !16)
!21 = !DILocation(line: 0, scope: !13)
!22 = !DILocation(line: 7, column: 3, scope: !13, atomGroup: 5, atomRank: 1)
!23 = distinct !DISubprogram(name: "sum", scope: !1, file: !1, line: 10, type: !24, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !26, keyInstructions: true)
!24 = !DISubroutineType(types: !25)
!25 = !{!16, !16}
!26 = !{!27, !28, !29}
!27 = !DILocalVariable(name: "n", arg: 1, scope: !23, file: !1, line: 10, type: !16)
!28 = !DILocalVariable(name: "total", scope: !23, file: !1, line: 11, type: !16)
!29 = !DILocalVariable(name: "i", scope: !30, file: !1, line: 12, type: !16)
!30 = distinct !DILexicalBlock(scope: !23, file: !1, line: 12, column: 3)
!31 = !DILocation(line: 0, scope: !23)
!32 = !DILocation(line: 0, scope: !30)
!33 = !DILocation(line: 12, column: 21, scope: !34, atomGroup: 10, atomRank: 1)
!34 = distinct !DILexicalBlock(scope: !30, file: !1, line: 12, column: 3)
!35 = !DILocation(line: 12, column: 3, scope: !30, atomGroup: 11, atomRank: 1)
!36 = !DILocation(line: 12, column: 3, scope: !30)
!37 = !DILocation(line: 14, column: 3, scope: !23, atomGroup: 9, atomRank: 1)
!38 = distinct !DISubprogram(name: "swap", scope: !1, file: !1, line: 17, type: !39, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !42, keyInstructions: true)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !41, !41}
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!42 = !{!43, !44, !45}
!43 = !DILocalVariable(name: "a", arg: 1, scope: !38, file: !1, line: 17, type: !41)
!44 = !DILocalVariable(name: "b", arg: 2, scope: !38, file: !1, line: 17, type: !41)
!45 = !DILocalVariable(name: "tmp", scope: !38, file: !1, line: 18, type: !16)
!46 = !DILocation(line: 0, scope: !38)
!47 = !DILocation(line: 18, column: 13, scope: !38, atomGroup: 1, atomRank: 2)
!48 = !DILocation(line: 19, column: 8, scope: !38, atomGroup: 2, atomRank: 2)
!49 = !DILocation(line: 19, column: 6, scope: !38, atomGroup: 2, atomRank: 1)
!50 = !DILocation(line: 20, column: 6, scope: !38, atomGroup: 3, atomRank: 1)
!51 = !DILocation(line: 21, column: 1, scope: !38, atomGroup: 4, atomRank: 1)