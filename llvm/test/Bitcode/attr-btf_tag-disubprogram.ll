; REQUIRES: x86-registered-target
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local i32 @f(i32 %a) local_unnamed_addr #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !13, metadata !DIExpression()), !dbg !17
  ret i32 0, !dbg !18
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git a6dd9d402a04d53403664bbb47771f2573c7ade0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "func.c", directory: "/home/yhs/work/tests/llvm/btf_tag")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git a6dd9d402a04d53403664bbb47771f2573c7ade0)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12, annotations: !14)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!14 = !{!15, !16}
!15 = !{!"btf_decl_tag", !"a"}
!16 = !{!"btf_decl_tag", !"b"}

; CHECK:        distinct !DISubprogram(name: "f"
; CHECK-SAME:   annotations: ![[ANNOT:[0-9]+]]
; CHECK:        ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
; CHECK:        ![[TAG1]] = !{!"btf_decl_tag", !"a"}
; CHECK:        ![[TAG2]] = !{!"btf_decl_tag", !"b"}

!17 = !DILocation(line: 0, scope: !8)
!18 = !DILocation(line: 1, column: 77, scope: !8)
