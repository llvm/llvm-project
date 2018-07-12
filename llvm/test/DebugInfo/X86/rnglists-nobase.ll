; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; We make sure that we generate DW_RLE_start_length range list entries when 
; we do not have a base, which is the case when functions go into different
; sections. We should have 3 individual range list entries because there are
; 3 functions.
; 
; From the following source:
;
; class A
; {
; public:
;   A();
; };
; 
; static A glob;
; void foo() {
; }
;
; Compile with clang -O1 -gdwarf-5 -S -emit-llvm
;
; CHECK:      .debug_rnglists contents:
; CHECK-NEXT: 0x00000000: Range List Header: length = 0x00000027, version = 0x0005,
; CHECK-SAME: addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
; CHECK-NEXT: Ranges:
; CHECK-NEXT: 0x0000000c: [DW_RLE_start_length]:
; CHECK-NEXT: 0x00000016: [DW_RLE_start_length]:
; CHECK-NEXT: 0x00000020: [DW_RLE_start_length]:
; CHECK-NEXT: 0x0000002a: [DW_RLE_end_of_list ]

%class.A = type { i8 }

@_ZL4glob = internal global %class.A zeroinitializer, align 1, !dbg !0
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_xx.cpp, i8* null }]

; Function Attrs: uwtable
define internal fastcc void @__cxx_global_var_init() unnamed_addr section ".text.startup" !dbg !16 {
entry:
  tail call void @_ZN1AC1Ev(%class.A* nonnull @_ZL4glob), !dbg !19
  ret void, !dbg !19
}

declare dso_local void @_ZN1AC1Ev(%class.A*) unnamed_addr

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local void @_Z3foov() local_unnamed_addr !dbg !20 {
entry:
  ret void, !dbg !21
}

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_xx.cpp() section ".text.startup" !dbg !22 {
entry:
  tail call fastcc void @__cxx_global_var_init(), !dbg !24
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glob", linkageName: "_ZL4glob", scope: !2, file: !3, line: 7, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 335191)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.cpp", directory: "/home/test", checksumkind: CSK_MD5, checksum: "535784cf49522e3a6d1166f6c4e482ba")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1A")
!7 = !{!8}
!8 = !DISubprogram(name: "A", scope: !6, file: !3, line: 4, type: !9, isLocal: false, isDefinition: false, scopeLine: 4, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = !{i32 2, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 7.0.0 (trunk 335191)"}
!16 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !3, file: !3, line: 7, type: !17, isLocal: true, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !DILocation(line: 7, column: 10, scope: !16)
!20 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 9, type: !17, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!21 = !DILocation(line: 10, column: 1, scope: !20)
!22 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_xx.cpp", scope: !3, file: !3, type: !23, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: true, unit: !2, retainedNodes: !4)
!23 = !DISubroutineType(types: !4)
!24 = !DILocation(line: 0, scope: !22)
