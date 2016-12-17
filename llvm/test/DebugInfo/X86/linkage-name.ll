; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_subprogram [9] *
; CHECK-NOT: DW_AT_{{(MIPS_)?}}linkage_name
; CHECK: DW_AT_specification

%class.A = type { i8 }

@a = global %class.A zeroinitializer, align 1, !dbg !20

define i32 @_ZN1A1aEi(%class.A* %this, i32 %b) nounwind uwtable ssp align 2 !dbg !5 {
entry:
  %this.addr = alloca %class.A*, align 8
  %b.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !21, metadata !DIExpression()), !dbg !23
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !24, metadata !DIExpression()), !dbg !25
  %this1 = load %class.A*, %class.A** %this.addr
  %0 = load i32, i32* %b.addr, align 4, !dbg !26
  ret i32 %0, !dbg !26
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.1 (trunk 152691) (llvm/trunk 152692)", isOptimized: false, emissionKind: FullDebug, file: !28, enums: !1, retainedTypes: !1, globals: !18, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEi", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !6, scope: null, type: !7, declaration: !13)
!6 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !11)
!11 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 1, size: 8, align: 8, file: !28, elements: !12)
!12 = !{!13}
!13 = !DISubprogram(name: "a", linkageName: "_ZN1A1aEi", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false, file: !6, scope: !11, type: !7)
!18 = !{!20}
!20 = !DIGlobalVariable(name: "a", line: 9, isLocal: false, isDefinition: true, scope: null, file: !6, type: !11)
!21 = !DILocalVariable(name: "this", line: 5, arg: 1, flags: DIFlagArtificial, scope: !5, file: !6, type: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!23 = !DILocation(line: 5, column: 8, scope: !5)
!24 = !DILocalVariable(name: "b", line: 5, arg: 2, scope: !5, file: !6, type: !9)
!25 = !DILocation(line: 5, column: 14, scope: !5)
!26 = !DILocation(line: 6, column: 4, scope: !27)
!27 = distinct !DILexicalBlock(line: 5, column: 17, file: !6, scope: !5)
!28 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo")
!29 = !{i32 1, !"Debug Info Version", i32 3}
