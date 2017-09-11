; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; Skip the definition of zed(foo*)
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name {{.*}} "_ZN3foo3barEv"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_specification {{.*}} "_ZN3foo3barEv"

%struct.foo = type { i8 }

define void @_Z3zedP3foo(%struct.foo* %x) uwtable !dbg !5 {
entry:
  %x.addr = alloca %struct.foo*, align 8
  store %struct.foo* %x, %struct.foo** %x.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %x.addr, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = load %struct.foo*, %struct.foo** %x.addr, align 8, !dbg !25
  call void @_ZN3foo3barEv(%struct.foo* %0), !dbg !25
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3foo3barEv(%struct.foo* %this) nounwind uwtable align 2 !dbg !20 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !28, metadata !DIExpression()), !dbg !29
  %this1 = load %struct.foo*, %struct.foo** %this.addr
  ret void, !dbg !30
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.0 ()", isOptimized: false, emissionKind: FullDebug, file: !32, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "zed", linkageName: "_Z3zedP3foo", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !6, scope: !6, type: !7)
!6 = !DIFile(filename: "/home/espindola/llvm/test.cc", directory: "/home/espindola/tmpfs/build")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_class_type, name: "foo", line: 1, size: 8, align: 8, file: !32, elements: !11)
!11 = !{!12}
!12 = !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !6, scope: !10, type: !13)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !10)
!20 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !6, scope: null, type: !13, declaration: !12)
!23 = !DILocalVariable(name: "x", line: 4, arg: 1, scope: !5, file: !6, type: !9)
!24 = !DILocation(line: 4, column: 15, scope: !5)
!25 = !DILocation(line: 4, column: 20, scope: !26)
!26 = distinct !DILexicalBlock(line: 4, column: 18, file: !6, scope: !5)
!27 = !DILocation(line: 4, column: 30, scope: !26)
!28 = !DILocalVariable(name: "this", line: 2, arg: 1, flags: DIFlagArtificial, scope: !20, file: !6, type: !15)
!29 = !DILocation(line: 2, column: 8, scope: !20)
!30 = !DILocation(line: 2, column: 15, scope: !31)
!31 = distinct !DILexicalBlock(line: 2, column: 14, file: !6, scope: !20)
!32 = !DIFile(filename: "/home/espindola/llvm/test.cc", directory: "/home/espindola/tmpfs/build")
!33 = !{i32 1, !"Debug Info Version", i32 3}
