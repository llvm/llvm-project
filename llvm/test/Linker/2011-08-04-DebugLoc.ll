; RUN: llvm-link %s %p/2011-08-04-DebugLoc2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if DebugLoc is mapped properly or not.


;CHECK-NOT: metadata !{i32 589870, i32 0, metadata !{{[0-9]+}}, metadata !"bar", metadata !"bar", metadata !"", metadata !{{[0-9]+}}, i32 1, metadata !{{[0-9]+}}, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, null, null, null}


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo() nounwind ssp !dbg !1 {
  ret i32 42, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "Apple clang version 3.0 (tags/Apple/clang-209.11) (based on LLVM 3.0svn)", isOptimized: true, emissionKind: FullDebug, file: !8, enums: !9, retainedTypes: !9)
!1 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, file: !8, scope: !2, type: !3)
!2 = !DIFile(filename: "a.c", directory: "/private/tmp")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocation(line: 2, column: 13, scope: !7)
!7 = distinct !DILexicalBlock(line: 2, column: 11, file: !8, scope: !1)
!8 = !DIFile(filename: "a.c", directory: "/private/tmp")
!9 = !{}
!11 = !{i32 1, !"Debug Info Version", i32 3}
