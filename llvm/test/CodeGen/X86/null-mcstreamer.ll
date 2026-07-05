; RUN: llc -mtriple=i386-pc-windows-msvc -filetype=null %s

define void @foo() !dbg !6 {
entry:
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 16.0.0 (git@github.com:llvm/llvm-project.git a8762195d56fb196d60d98045c75eb33af68df0c)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/Users/matt/src/llvm-project/clang/test/Misc/<stdin>", directory: "/Users/matt/src/llvm-project/build_debug", checksumkind: CSK_MD5, checksum: "de25aa8ed7057b63c6695dfd0822438b")
!2 = !{i32 1, !"NumRegisterParameters", i32 0}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = distinct !DISubprogram(name: "foo", scope: !7, file: !7, line: 5, type: !8, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!7 = !DIFile(filename: "clang/test/Misc/x86-emit-codegen-only.c", directory: "/Users/matt/src/llvm-project", checksumkind: CSK_MD5, checksum: "de25aa8ed7057b63c6695dfd0822438b")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = !DILocation(line: 5, column: 13, scope: !6)
