; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; CHECK:      0x00000033: DW_TAG_formal_parameter [3]
; CHECK-NEXT:               DW_AT_location [DW_FORM_sec_offset]   (0x0000000c
; CHECK-NEXT:                  [0x0000000000000000, 0x0000000000000004): DW_OP_breg5 RDI+0
; CHECK-NEXT:                  [0x0000000000000004, 0x0000000000000012): DW_OP_breg3 RBX+0)
; CHECK-NEXT:               DW_AT_name [DW_FORM_strx1]    (indexed (0000000e) string = "a")
; CHECK-NEXT:               DW_AT_decl_file [DW_FORM_data1]       ("/home/folder{{\\|\/}}test.cc")
; CHECK-NEXT:               DW_AT_decl_line [DW_FORM_data1]       (6)
; CHECK-NEXT:               DW_AT_type [DW_FORM_ref4]     (cu + 0x0040 => {0x00000040} "A")

; CHECK:      .debug_loclists contents:
; CHECK-NEXT: 0x00000000: locations list header: length = 0x00000017, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
; CHECK-NEXT: 0x00000000:
; CHECK-NEXT:  [0x0000000000000000, 0x0000000000000004): DW_OP_breg5 RDI+0
; CHECK-NEXT:  [0x0000000000000004, 0x0000000000000012): DW_OP_breg3 RBX+0

; There is no way to use llvm-dwarfdump atm (2018, october) to verify the DW_LLE_* codes emited,
; because dumper is not yet implements that. Use asm code to do this check instead.
;
; RUN: llc -mtriple=x86_64-pc-linux -filetype=asm < %s -o - | FileCheck %s --check-prefix=ASM
; ASM:      .section .debug_loclists,"",@progbits
; ASM-NEXT: .long .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
; ASM-NEXT: .Ldebug_loclist_table_start0:
; ASM-NEXT:  .short 5                              # Version
; ASM-NEXT:  .byte 8                               # Address size
; ASM-NEXT:  .byte 0                               # Segment selector size
; ASM-NEXT:  .long 0                               # Offset entry count
; ASM-NEXT: .Lloclists_table_base0:                
; ASM-NEXT: .Ldebug_loc0:
; ASM-NEXT:  .byte 4                               # DW_LLE_offset_pair
; ASM-NEXT:  .uleb128 .Lfunc_begin0-.Lfunc_begin0  # starting offset
; ASM-NEXT:  .uleb128 .Ltmp0-.Lfunc_begin0         # ending offset
; ASM-NEXT:  .short 2                              # Loc expr size
; ASM-NEXT:  .byte 117                             # DW_OP_breg5
; ASM-NEXT:  .byte 0                               # 0
; ASM-NEXT:  .byte 4                               # DW_LLE_offset_pair
; ASM-NEXT:  .uleb128 .Ltmp0-.Lfunc_begin0         # starting offset
; ASM-NEXT:  .uleb128 .Ltmp1-.Lfunc_begin0         # ending offset
; ASM-NEXT:  .short 2                              # Loc expr size
; ASM-NEXT:  .byte 115                             # DW_OP_breg3
; ASM-NEXT:  .byte 0                               # 0
; ASM-NEXT:  .byte 0                               # DW_LLE_end_of_list
; ASM-NEXT: .Ldebug_loclist_table_end0:

; ModuleID = 'test.cc'
source_filename = "test.cc"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 (...)** }

@_ZTV1A = dso_local unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A3fooEv to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A3barEv to i8*)] }, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS1A = dso_local constant [3 x i8] c"1A\00", align 1
@_ZTI1A = dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 8

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z3baz1A(%struct.A* %a) #0 !dbg !7 {
entry:
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !23, metadata !DIExpression()), !dbg !24
  call void @_ZN1A3fooEv(%struct.A* %a), !dbg !25
  call void @_ZN1A3barEv(%struct.A* %a), !dbg !26
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_ZN1A3fooEv(%struct.A* %this) unnamed_addr #2 align 2 !dbg !28 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !29, metadata !DIExpression()), !dbg !31
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  ret void, !dbg !32
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_ZN1A3barEv(%struct.A* %this) unnamed_addr #2 align 2 !dbg !33 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !34, metadata !DIExpression()), !dbg !35
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  ret void, !dbg !36
}

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #3 !dbg !37 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !38
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 344035)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "/home/folder", checksumkind: CSK_MD5, checksum: "e0f357ad6dcb791a774a0dae55baf5e7")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 344035)"}
!7 = distinct !DISubprogram(name: "baz", linkageName: "_Z3baz1A", scope: !1, file: !1, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 64, flags: DIFlagTypePassByReference, elements: !11, vtableHolder: !10, identifier: "_ZTS1A")
!11 = !{!12, !18, !22}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !1, file: !1, baseType: !13, size: 64, flags: DIFlagArtificial)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !15, size: 64)
!15 = !DISubroutineType(types: !16)
!16 = !{!17}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEv", scope: !10, file: !1, line: 2, type: !19, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !10, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DISubprogram(name: "bar", linkageName: "_ZN1A3barEv", scope: !10, file: !1, line: 3, type: !19, isLocal: false, isDefinition: false, scopeLine: 3, containingType: !10, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 1, flags: DIFlagPrototyped, isOptimized: false)
!23 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 6, type: !10)
!24 = !DILocation(line: 6, column: 19, scope: !7)
!25 = !DILocation(line: 7, column: 6, scope: !7)
!26 = !DILocation(line: 8, column: 6, scope: !7)
!27 = !DILocation(line: 9, column: 1, scope: !7)
!28 = distinct !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEv", scope: !10, file: !1, line: 12, type: !19, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !18, retainedNodes: !2)
!29 = !DILocalVariable(name: "this", arg: 1, scope: !28, type: !30, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!31 = !DILocation(line: 0, scope: !28)
!32 = !DILocation(line: 12, column: 16, scope: !28)
!33 = distinct !DISubprogram(name: "bar", linkageName: "_ZN1A3barEv", scope: !10, file: !1, line: 13, type: !19, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !22, retainedNodes: !2)
!34 = !DILocalVariable(name: "this", arg: 1, scope: !33, type: !30, flags: DIFlagArtificial | DIFlagObjectPointer)
!35 = !DILocation(line: 0, scope: !33)
!36 = !DILocation(line: 13, column: 16, scope: !33)
!37 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !15, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!38 = !DILocation(line: 16, column: 3, scope: !37)
