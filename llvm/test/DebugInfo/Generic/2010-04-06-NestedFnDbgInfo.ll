; RUN: %llc_dwarf -O0 -filetype=obj -o - < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s
; Radar 7833483
; Do not emit a separate out-of-line definition DIE for the function-local 'foo'
; function (member of the function local 'A' type)
; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_class_type
; CHECK-NEXT: DW_AT_name {{.*}} "A"
; Check that the subprogram inside the class definition has low_pc, only
; attached to the definition.
; CHECK: [[FOO_INL:0x........]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_low_pc
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "foo"
; And just double check that there's no out of line definition that references
; this subprogram.
; CHECK-NOT: DW_AT_specification {{.*}} {[[FOO_INL]]}

%class.A = type { i8 }
%class.B = type { i8 }

define i32 @main() ssp !dbg !2 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=3]
  %b = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.A* %b, metadata !0, metadata !DIExpression()), !dbg !14
  %call = call i32 @_ZN1B2fnEv(%class.A* %b), !dbg !15 ; <i32> [#uses=1]
  store i32 %call, i32* %retval, !dbg !15
  %0 = load i32, i32* %retval, !dbg !16                ; <i32> [#uses=1]
  ret i32 %0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr i32 @_ZN1B2fnEv(%class.A* %this) ssp align 2 !dbg !10 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  %a = alloca %class.A, align 1                   ; <%class.A*> [#uses=1]
  %i = alloca i32, align 4                        ; <i32*> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !18
  %this1 = load %class.A*, %class.A** %this.addr             ; <%class.A*> [#uses=0]
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !19, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32* %i, metadata !28, metadata !DIExpression()), !dbg !29
  %call = call i32 @_ZZN1B2fnEvEN1A3fooEv(%class.A* %a), !dbg !30 ; <i32> [#uses=1]
  store i32 %call, i32* %i, !dbg !30
  %tmp = load i32, i32* %i, !dbg !31                   ; <i32> [#uses=1]
  store i32 %tmp, i32* %retval, !dbg !31
  %0 = load i32, i32* %retval, !dbg !32                ; <i32> [#uses=1]
  ret i32 %0, !dbg !32
}

define internal i32 @_ZZN1B2fnEvEN1A3fooEv(%class.A* %this) ssp align 2 !dbg !23 {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %this.addr = alloca %class.A*, align 8          ; <%class.A**> [#uses=2]
  store %class.A* %this, %class.A** %this.addr
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !33, metadata !DIExpression(DW_OP_deref)), !dbg !34
  %this1 = load %class.A*, %class.A** %this.addr             ; <%class.A*> [#uses=0]
  store i32 42, i32* %retval, !dbg !35
  %0 = load i32, i32* %retval, !dbg !35                ; <i32> [#uses=1]
  ret i32 %0, !dbg !35
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!40}
!37 = !{!2, !10, !23}

!0 = !DILocalVariable(name: "b", line: 16, scope: !1, file: !3, type: !8)
!1 = distinct !DILexicalBlock(line: 15, column: 12, file: !38, scope: !2)
!2 = distinct !DISubprogram(name: "main", linkageName: "main", line: 15, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, scopeLine: 15, file: !38, scope: !3, type: !5)
!3 = !DIFile(filename: "one.cc", directory: "/tmp")
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang 1.5", isOptimized: false, emissionKind: FullDebug, file: !38, enums: !39, retainedTypes: !39, imports:  null)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DICompositeType(tag: DW_TAG_class_type, name: "B", line: 2, size: 8, align: 8, file: !38, scope: !3, elements: !9)
!9 = !{!10}
!10 = distinct !DISubprogram(name: "fn", linkageName: "_ZN1B2fnEv", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, scopeLine: 4, file: !38, scope: !8, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{!7, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !38, scope: !3, baseType: !8)
!14 = !DILocation(line: 16, column: 5, scope: !1)
!15 = !DILocation(line: 17, column: 3, scope: !1)
!16 = !DILocation(line: 18, column: 1, scope: !2)
; Manually modified to avoid pointers (thus dependence on pointer size) in Generic test
!17 = !DILocalVariable(name: "this", line: 4, arg: 1, scope: !10, file: !3, type: !8)
!18 = !DILocation(line: 4, column: 7, scope: !10)
!19 = !DILocalVariable(name: "a", line: 9, scope: !20, file: !3, type: !21)
!20 = distinct !DILexicalBlock(line: 4, column: 12, file: !38, scope: !10)
!21 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 5, size: 8, align: 8, file: !38, scope: !10, elements: !22)
!22 = !{!23}
!23 = distinct !DISubprogram(name: "foo", linkageName: "_ZZN1B2fnEvEN1A3fooEv", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, scopeLine: 7, file: !38, scope: !21, type: !24)
!24 = !DISubroutineType(types: !25)
!25 = !{!7, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !38, scope: !3, baseType: !21)
!27 = !DILocation(line: 9, column: 7, scope: !20)
!28 = !DILocalVariable(name: "i", line: 10, scope: !20, file: !3, type: !7)
!29 = !DILocation(line: 10, column: 9, scope: !20)
!30 = !DILocation(line: 10, column: 5, scope: !20)
!31 = !DILocation(line: 11, column: 5, scope: !20)
!32 = !DILocation(line: 12, column: 3, scope: !10)
; Manually modified like !17 above
!33 = !DILocalVariable(name: "this", line: 7, arg: 1, scope: !23, file: !3, type: !21)
!34 = !DILocation(line: 7, column: 11, scope: !23)
!35 = !DILocation(line: 7, column: 19, scope: !36)
!36 = distinct !DILexicalBlock(line: 7, column: 17, file: !38, scope: !23)
!38 = !DIFile(filename: "one.cc", directory: "/tmp")
!39 = !{}
!40 = !{i32 1, !"Debug Info Version", i32 3}
