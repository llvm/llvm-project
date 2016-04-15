; This file is for use with 2011-08-10-unique-class-type.ll
; RUN: true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%"class.N1::A" = type { i8 }

define void @_Z3barN2N11AE() nounwind uwtable ssp !dbg !5 {
entry:
  %youra = alloca %"class.N1::A", align 1
  call void @llvm.dbg.declare(metadata %"class.N1::A"* %youra, metadata !9, metadata !DIExpression()), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.0 (trunk 137954)", isOptimized: true, emissionKind: FullDebug, file: !16, enums: !2, retainedTypes: !2, globals: !2)
!1 = !{!2}
!2 = !{}
!5 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barN2N11AE", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scope: !6, type: !7)
!6 = !DIFile(filename: "n2.c", directory: "/private/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "youra", line: 4, arg: 1, scope: !5, file: !6, type: !10)
!10 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 3, size: 8, align: 8, file: !17, scope: !11, elements: !2)
!11 = !DINamespace(name: "N1", line: 2, file: !17, scope: null)
!12 = !DIFile(filename: "./n.h", directory: "/private/tmp")
!13 = !DILocation(line: 4, column: 12, scope: !5)
!14 = !DILocation(line: 4, column: 20, scope: !15)
!15 = distinct !DILexicalBlock(line: 4, column: 19, file: !16, scope: !5)
!16 = !DIFile(filename: "n2.c", directory: "/private/tmp")
!17 = !DIFile(filename: "./n.h", directory: "/private/tmp")
!18 = !{i32 1, !"Debug Info Version", i32 3}
