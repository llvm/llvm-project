; ModuleID = '/tmp/t.c'
source_filename = "/tmp/t.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @f() #0 !dbg !9 {
entry:
  ret void, !dbg !14
}

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0git (git@github.com:llvm/llvm-project.git 29ee66f4a0967e43a035f147c960743c7b640f2f)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "/INLINE/inlined.c", directory: "/Volumes/Data/llvm-project", checksumkind: CSK_MD5, checksum: "3183154a5cb31debe9a8e27ca500bc3c")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{!"clang version 18.0.0git (git@github.com:llvm/llvm-project.git 29ee66f4a0967e43a035f147c960743c7b640f2f)"}
!9 = distinct !DISubprogram(name: "f", scope: !10, file: !10, line: 2, type: !11, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DIFile(filename: "/INLINE/inlined.c", directory: "", source: "void stop();
void f() {
  // This is inline source code.
}
")
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!14 = !DILocation(line: 4, column: 1, scope: !9)
