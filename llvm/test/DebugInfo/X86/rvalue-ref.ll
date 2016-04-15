; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj -O0
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_rvalue_reference_type

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

define void @_Z3fooOi(i32* %i) uwtable ssp !dbg !5 {
entry:
  %i.addr = alloca i32*, align 8
  store i32* %i, i32** %i.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %i.addr, metadata !11, metadata !DIExpression()), !dbg !12
  %0 = load i32*, i32** %i.addr, align 8, !dbg !13
  %1 = load i32, i32* %0, align 4, !dbg !13
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %1), !dbg !13
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 (trunk 157054) (llvm/trunk 157060)", isOptimized: false, emissionKind: FullDebug, file: !16, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooOi", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !16, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, size: 64, align: 64, baseType: !10)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "i", line: 4, arg: 1, scope: !5, file: !6, type: !9)
!12 = !DILocation(line: 4, column: 17, scope: !5)
!13 = !DILocation(line: 6, column: 3, scope: !14)
!14 = distinct !DILexicalBlock(line: 5, column: 1, file: !16, scope: !5)
!15 = !DILocation(line: 7, column: 1, scope: !14)
!16 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!17 = !{i32 1, !"Debug Info Version", i32 3}
