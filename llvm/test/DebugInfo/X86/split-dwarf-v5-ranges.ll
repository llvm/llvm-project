; RUN: llc -split-dwarf-file=foo.dwo -mtriple=x86_64-unknown-linux-gnu -filetype=obj < %s \
; RUN: 	    | llvm-dwarfdump -v -debug-info -debug-rnglists - | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: .debug_info.dwo contents:
; CHECK: DW_AT_ranges [DW_FORM_rnglistx] (indexed (0x0) rangelist = 0x00000010
; CHECK:          [0x0000000000000001, 0x000000000000000c) ".text"
; CHECK:          [0x000000000000000e, 0x0000000000000013) ".text")

; CHECK: .debug_rnglists.dwo contents:
; CHECK: 0x00000000: range list header: length = 0x00000015, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000001
; CHECK: offsets: [
; CHECK: 0x00000004 => 0x00000010
; CHECK: ]
; CHECK: ranges:
; CHECK: 0x00000010: [DW_RLE_base_addressx]:  0x0000000000000000
; CHECK: 0x00000012: [DW_RLE_offset_pair  ]:  0x0000000000000001, 0x000000000000000c => [0x0000000000000001, 0x000000000000000c)
; CHECK: 0x00000015: [DW_RLE_offset_pair  ]:  0x000000000000000e, 0x0000000000000013 => [0x000000000000000e, 0x0000000000000013)
; CHECK: 0x00000018: [DW_RLE_end_of_list  ]

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2f3v() !dbg !7 {
entry:
  %x = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %x, metadata !10, metadata !DIExpression()), !dbg !13
  %call = call i32 @_Z2f2v(), !dbg !14
  store i32 %call, i32* %x, align 4, !dbg !13
  %0 = load i32, i32* %x, align 4, !dbg !13
  %tobool = icmp ne i32 %0, 0, !dbg !13
  br i1 %tobool, label %if.then, label %if.end, !dbg !15

if.then:                                          ; preds = %entry
  call void @_Z2f1v(), !dbg !16
  br label %if.end, !dbg !18

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare dso_local i32 @_Z2f2v()

declare dso_local void @_Z2f1v()

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f4v() #3 section "x" !dbg !20 {
entry:
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 344806) (llvm/trunk 344835)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: GNU)
!1 = !DIFile(filename: "ranges.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "a1e825b91fba21d696f05eb06d440aa3")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 344806) (llvm/trunk 344835)"}
!7 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "x", scope: !11, file: !1, line: 4, type: !12)
!11 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 11)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 4, column: 11, scope: !11)
!14 = !DILocation(line: 4, column: 15, scope: !11)
!15 = !DILocation(line: 4, column: 11, scope: !7)
!16 = !DILocation(line: 5, column: 5, scope: !17)
!17 = distinct !DILexicalBlock(scope: !11, file: !1, line: 4, column: 21)
!18 = !DILocation(line: 6, column: 3, scope: !17)
!19 = !DILocation(line: 7, column: 1, scope: !7)
!20 = distinct !DISubprogram(name: "f4", linkageName: "_Z2f4v", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 8, column: 42, scope: !20)
