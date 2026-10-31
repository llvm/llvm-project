; ModuleID = 'src/heavy.cpp'
source_filename = "src/heavy.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

$_Z5heavyIN3lib2v23TagEEii = comdat any

; Function Attrs: mustprogress nounwind uwtable
define weak_odr dso_local noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef %n) #0 comdat !dbg !11 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4, !tbaa !23
    #dbg_declare(ptr %n.addr, !16, !DIExpression(), !24)
  %0 = load i32, ptr %n.addr, align 4, !dbg !25, !tbaa !23
  %xor = xor i32 %0, 5, !dbg !26
  %mul = mul nsw i32 %xor, 33, !dbg !27
  ret i32 %mul, !dbg !28
}

attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}
!llvm.errno.tbaa = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (https://github.com/llvm/llvm-project.git 0d266b728984b5cb6c513ba3cc374ae158b5cbdd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "src/heavy.cpp", directory: "/home/h2h/test/repro_import_guid_nowrapper", checksumkind: CSK_MD5, checksum: "46f2b239d009d23359ee4b739415cd4e")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git 0d266b728984b5cb6c513ba3cc374ae158b5cbdd)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"__libc_errno", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !DISubprogram(name: "heavy<lib::v2::Tag>", linkageName: "_Z5heavyIN3lib2v23TagEEii", scope: !1, file: !1, line: 2, type: !12, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !17, retainedNodes: !15, keyInstructions: true)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DILocalVariable(name: "n", arg: 1, scope: !11, file: !1, line: 2, type: !14)
!17 = !{!18}
!18 = !DITemplateTypeParameter(name: "T", type: !19)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "Tag", scope: !21, file: !20, line: 8, size: 8, flags: DIFlagFwdDecl, identifier: "_ZTSN3lib2v23TagE")
!20 = !DIFile(filename: "src/lib.h", directory: "/home/h2h/test/repro_import_guid_nowrapper", checksumkind: CSK_MD5, checksum: "02981f27edc3d1ac7c8acdcde98d9826")
!21 = !DINamespace(name: "v2", scope: !22, exportSymbols: true)
!22 = !DINamespace(name: "lib", scope: null)
!23 = !{!8, !8, i64 0}
!24 = !DILocation(line: 2, column: 34, scope: !11)
!25 = !DILocation(line: 2, column: 47, scope: !11)
!26 = !DILocation(line: 2, column: 49, scope: !11)
!27 = !DILocation(line: 2, column: 54, scope: !11, atomGroup: 1, atomRank: 2)
!28 = !DILocation(line: 2, column: 39, scope: !11, atomGroup: 1, atomRank: 1)
