; RUN: opt < %s -verify -S | FileCheck %s

; CHECK: DW_LANG_Mips_Assembler

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.declare(metadata i32* null, metadata !1, metadata !DIExpression()), !dbg !DILocation(scope: !6)
  ret void
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5}
!2 = distinct !DICompileUnit(language: DW_LANG_Mips_Assembler, producer: "clang version 3.3 ", isOptimized: false, emissionKind: FullDebug, file: !4, enums: !3, retainedTypes: !3, subprograms: !{!6}, globals: !3, imports:  !3)
!3 = !{}
!0 = !DILocation(line: 662302, column: 26, scope: !1)
!1 = !DILocalVariable(name: "foo", scope: !6)
!4 = !DIFile(filename: "scratch.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!6 = distinct !DISubprogram()

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
!5 = !{i32 1, !"Debug Info Version", i32 3}
