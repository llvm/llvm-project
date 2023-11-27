; RUN: opt < %s --passes='print<source-expr>' -disable-output  2>&1 | FileCheck %s



; CHECK-LABEL: Load Store Expression foo
define dso_local i64 @foo(ptr nocapture noundef readonly %lp, i64 noundef %n1, i64 noundef %n2) local_unnamed_addr #0 !dbg !9 {
entry:
  call void @llvm.dbg.value(metadata ptr %lp, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i64 %n1, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i64 %n2, metadata !17, metadata !DIExpression()), !dbg !18
; CHECK: %mul = (n1 << 1)
  %mul = shl nsw i64 %n1, 1, !dbg !19
; CHECK: %add = ((n1 << 1) + n2)
  %add = add nsw i64 %mul, %n2, !dbg !20
; CHECK: %arrayidx = &lp[((n1 << 1) + n2)]
  %arrayidx = getelementptr inbounds i64, ptr %lp, i64 %add, !dbg !21
; CHECK: %0 = lp[((n1 << 1) + n2)]
  %0 = load i64, ptr %arrayidx, align 8, !dbg !21, !tbaa !22
  ret i64 %0, !dbg !26
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (https://github.com/phyBrackets/llvm-project-1.git 598f579cf15336fb818edb33659a839e3338e624)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "../cpp/ts.c", directory: "/home/shivam/llvm-project-1", checksumkind: CSK_MD5, checksum: "9e9f3a66ae451d81cff547dc10cd8006")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 17.0.0 (https://github.com/phyBrackets/llvm-project-1.git 598f579cf15336fb818edb33659a839e3338e624)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !10, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13, !12, !12}
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!14 = !{!15, !16, !17}
!15 = !DILocalVariable(name: "lp", arg: 1, scope: !9, file: !1, line: 1, type: !13)
!16 = !DILocalVariable(name: "n1", arg: 2, scope: !9, file: !1, line: 1, type: !12)
!17 = !DILocalVariable(name: "n2", arg: 3, scope: !9, file: !1, line: 1, type: !12)
!18 = !DILocation(line: 0, scope: !9)
!19 = !DILocation(line: 3, column: 15, scope: !9)
!20 = !DILocation(line: 3, column: 20, scope: !9)
!21 = !DILocation(line: 3, column: 10, scope: !9)
!22 = !{!23, !23, i64 0}
!23 = !{!"long", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 3, column: 3, scope: !9)
