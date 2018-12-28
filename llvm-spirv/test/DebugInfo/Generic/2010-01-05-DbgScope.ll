; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple < %t.ll -o /dev/null
; PR 5942
define i8* @foo() nounwind {
entry:
  %0 = load i32, i32* undef, align 4, !dbg !0          ; <i32> [#uses=1]
  %1 = inttoptr i32 %0 to i8*, !dbg !0            ; <i8*> [#uses=1]
  ret i8* %1, !dbg !10

}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = !DILocation(line: 571, column: 3, scope: !1)
!1 = distinct !DILexicalBlock(line: 1, column: 1, file: !11, scope: !2)
!2 = distinct !DISubprogram(name: "foo", linkageName: "foo", file: !11, line: 561, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scope: !3, type: !4)
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang 1.1", isOptimized: true, emissionKind: FullDebug, file: !11, enums: !12, retainedTypes: !12)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !DILocation(line: 588, column: 1, scope: !2)
!11 = !DIFile(filename: "hashtab.c", directory: "/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty")
!12 = !{}
!14 = !{i32 1, !"Debug Info Version", i32 3}
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
