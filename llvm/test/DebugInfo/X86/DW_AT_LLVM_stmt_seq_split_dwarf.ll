; RUN: llc -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo -o %t.o -filetype=obj %s -emit-func-debug-line-table-offsets
; RUN: llvm-dwarfdump -v -all %t.dwo | FileCheck %s

; CHECK: DW_AT_LLVM_stmt_sequence

target triple = "x86_64-unknown-linux-gnu"

define void @splitmountain07030226() !dbg !4 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (https://github.com/llvm/llvm-project 005b3858f602ec5a193740cb166244d5234ce98f)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "/tmp/test.dwo", emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "newspicer073126", directory: "yosemite080326")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "splitmountain07030226", linkageName: "splitmountain07030226", scope: !5, file: !5, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DIFile(filename: "newspicer073126", directory: "yosemite080326", checksumkind: CSK_MD5, checksum: "20fb564d6e0e7db96628bc4acf1721b4")
!6 = !DISubroutineType(types: !2)
