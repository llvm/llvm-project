;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -filetype=obj %s -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; Test case derived from compiling the following source with clang -g:
;
; namespace pr14763 {
; struct foo {
;   foo(const foo&);
; };
;
; foo func(foo f) {
;   return f; // reference 'f' for now because otherwise we hit another bug
; }
;
; void sink(void*);
;
; void func2(bool b, foo g) {
;   if (b)
;     sink(&g); // reference 'f' for now because otherwise we hit another bug
; }
; }

; CHECK: debug_info contents
; The parameter is accessed indirectly (with a zero offset) from the second
; register parameter. RDI is consumed by 'sret'.
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name{{.*}} = "func"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location {{.*}}
; CHECK-NEXT: DW_OP_breg4 RSI+0, DW_OP_deref
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}} = "f"

; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name{{.*}} = "func2"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location{{.*}}(DW_OP_fbreg +23)
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location{{.*}}(
; CHECK-NEXT: {{.*}}: DW_OP_breg7 RSP+8, DW_OP_deref, DW_OP_deref
; CHECK-NEXT: {{.*}}: DW_OP_breg4 RSI+0, DW_OP_deref
; CHECK-NEXT: {{.*}}: DW_OP_breg7 RSP+8, DW_OP_deref, DW_OP_deref)
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}} = "g"

%"struct.pr14763::foo" = type { i8 }

; Function Attrs: uwtable
define void @_ZN7pr147634funcENS_3fooE(%"struct.pr14763::foo"* noalias sret %agg.result, %"struct.pr14763::foo"* %f) #0 !dbg !4 {
entry:
  call void @llvm.dbg.declare(metadata %"struct.pr14763::foo"* %f, metadata !22, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @_ZN7pr147633fooC1ERKS0_(%"struct.pr14763::foo"* %agg.result, %"struct.pr14763::foo"* %f), !dbg !25
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_ZN7pr147633fooC1ERKS0_(%"struct.pr14763::foo"*, %"struct.pr14763::foo"*) #2

; Function Attrs: uwtable
define void @_ZN7pr147635func2EbNS_3fooE(i1 zeroext %b, %"struct.pr14763::foo"* %g) #0 !dbg !17 {
entry:
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %b.addr, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata %"struct.pr14763::foo"* %g, metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !27
  %0 = load i8, i8* %b.addr, align 1, !dbg !29
  %tobool = trunc i8 %0 to i1, !dbg !29
  br i1 %tobool, label %if.then, label %if.end, !dbg !29

if.then:                                          ; preds = %entry
  %1 = bitcast %"struct.pr14763::foo"* %g to i8*, !dbg !31
  call void @_ZN7pr147634sinkEPv(i8* %1), !dbg !31
  br label %if.end, !dbg !31

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !32
}

declare void @_ZN7pr147634sinkEPv(i8*)

attributes #0 = { uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !33}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "pass.cpp", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "func", linkageName: "_ZN7pr147634funcENS_3fooE", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 6, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DINamespace(name: "pr14763", scope: null)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 2, size: 8, align: 8, file: !1, scope: !5, elements: !9)
!9 = !{!10}
!10 = !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !8, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !8)
!14 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!17 = distinct !DISubprogram(name: "func2", linkageName: "_ZN7pr147635func2EbNS_3fooE", line: 12, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 12, file: !1, scope: !5, type: !18, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20, !8}
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!21 = !{i32 2, !"Dwarf Version", i32 3}
!22 = !DILocalVariable(name: "f", line: 6, arg: 1, scope: !4, file: !23, type: !8)
!23 = !DIFile(filename: "pass.cpp", directory: "/tmp")
!24 = !DILocation(line: 6, scope: !4)
!25 = !DILocation(line: 7, scope: !4)
!26 = !DILocalVariable(name: "b", line: 12, arg: 1, scope: !17, file: !23, type: !20)
!27 = !DILocation(line: 12, scope: !17)
!28 = !DILocalVariable(name: "g", line: 12, arg: 2, scope: !17, file: !23, type: !8)
!29 = !DILocation(line: 13, scope: !30)
!30 = distinct !DILexicalBlock(line: 13, column: 0, file: !1, scope: !17)
!31 = !DILocation(line: 14, scope: !30)
!32 = !DILocation(line: 15, scope: !17)
!33 = !{i32 1, !"Debug Info Version", i32 3}
