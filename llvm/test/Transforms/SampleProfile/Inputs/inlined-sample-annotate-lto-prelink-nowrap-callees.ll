; ModuleID = 'src/heavy.cpp'
source_filename = "src/heavy.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

$_Z5heavyIN3lib2v23TagEEii = comdat any

; Function Attrs: mustprogress nounwind uwtable
define weak_odr dso_local noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef %n) #0 comdat !dbg !10 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4, !tbaa !13
  %0 = load i32, ptr %n.addr, align 4, !dbg !14, !tbaa !13
  %xor = xor i32 %0, 5, !dbg !15
  %mul = mul nsw i32 %xor, 33, !dbg !16
  ret i32 %mul, !dbg !17
}

attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (https://github.com/llvm/llvm-project.git 6bcf1c81a057579159e38dc03adf0afdd51c9c29)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "src/heavy.cpp", directory: "/home/h2h/test/repro_import_guid_nowrapper")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git 6bcf1c81a057579159e38dc03adf0afdd51c9c29)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !DISubprogram(name: "heavy<lib::v2::Tag>", linkageName: "_Z5heavyIN3lib2v23TagEEii", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !{!7, !7, i64 0}
!14 = !DILocation(line: 2, column: 47, scope: !10)
!15 = !DILocation(line: 2, column: 49, scope: !10)
!16 = !DILocation(line: 2, column: 54, scope: !10)
!17 = !DILocation(line: 2, column: 39, scope: !10)
