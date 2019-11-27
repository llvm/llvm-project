; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -function-sections -o %t < %s
; RUN: llvm-dwarfdump -v -debug-info -debug-loclists %t | FileCheck %s

; RUN: llc -dwarf-version=5 -split-dwarf-file=foo.dwo -mtriple=x86_64-pc-linux -filetype=obj -function-sections -o %t < %s
; RUN: llvm-dwarfdump -v -debug-info -debug-loclists %t | FileCheck %s --check-prefix=DWO

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_loclistx]   (indexed (0x0) loclist = 0x00000018:
; CHECK-NEXT:     [0x0000000000000000, 0x0000000000000003) ".text._Z2f1ii": DW_OP_consts +3, DW_OP_stack_value
; CHECK-NEXT:     [0x0000000000000003, 0x0000000000000004) ".text._Z2f1ii": DW_OP_consts +4, DW_OP_stack_value)
; CHECK-NEXT:   DW_AT_name {{.*}} "y"

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_loclistx]   (indexed (0x1) loclist = 0x00000029:
; CHECK-NEXT:     [0x0000000000000000, 0x0000000000000003) ".text._Z2f1ii": DW_OP_consts +5, DW_OP_stack_value)
; CHECK-NEXT:   DW_AT_name {{.*}} "x"

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_loclistx]   (indexed (0x2) loclist = 0x00000031:
; CHECK-NEXT:     [0x0000000000000003, 0x0000000000000004) ".text._Z2f1ii": DW_OP_reg0 RAX)
; CHECK-NEXT:   DW_AT_name {{.*}} "r"

; CHECK:      .debug_loclists contents:
; CHECK-NEXT: 0x00000000: locations list header: length = 0x00000035, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000003

; DWO:      .debug_loclists.dwo contents:
; DWO-NEXT: 0x00000000: locations list header: length = 0x00000035, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000003

; CHECK-NEXT: offsets: [
; CHECK-NEXT: 0x0000000c => 0x00000018
; CHECK-NEXT: 0x0000001d => 0x00000029
; CHECK-NEXT: 0x00000025 => 0x00000031
; CHECK-NEXT: ]

; Don't use startx_length if there's more than one entry, because the shared
; base address will be useful for both the range that does start at the start of
; the function, and the one that doesn't.

; CHECK-NEXT: 0x00000018:
; CHECK-NEXT:             DW_LLE_base_addressx (0x0000000000000000)
; CHECK-NEXT:             DW_LLE_offset_pair   (0x0000000000000000, 0x0000000000000003): DW_OP_consts +3, DW_OP_stack_value
; CHECK-NEXT:             DW_LLE_offset_pair   (0x0000000000000003, 0x0000000000000004): DW_OP_consts +4, DW_OP_stack_value
; CHECK-NEXT:             DW_LLE_end_of_list   ()

; Show that startx_length can be used when the address range starts at the start of the function.

; CHECK:      0x00000029:
; CHECK-NEXT:             DW_LLE_startx_length (0x0000000000000000, 0x0000000000000003): DW_OP_consts +5, DW_OP_stack_value
; CHECK-NEXT:             DW_LLE_end_of_list   ()

; And use a base address when the range doesn't start at an existing/useful
; address in the pool.

; CHECK:      0x00000031:
; CHECK-NEXT:             DW_LLE_base_addressx (0x0000000000000000)
; CHECK-NEXT:             DW_LLE_offset_pair   (0x0000000000000003, 0x0000000000000004): DW_OP_reg0 RAX
; CHECK-NEXT:             DW_LLE_end_of_list   ()

; Built with clang -O3 -ffunction-sections from source:
; 
; int f1(int i, int j) {
;   int x = 5;
;   int y = 3;
;   int r = i + j;
;   int undef;
;   x = undef;
;   y = 4;
;   return r;
; }
; void f2() {
; }

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @_Z2f1ii(i32 %i, i32 %j) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %i, metadata !12, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %j, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 5, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 3, metadata !15, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %j, %i, !dbg !19
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 undef, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 4, metadata !15, metadata !DIExpression()), !dbg !18
  ret i32 %add, !dbg !20
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local void @_Z2f2v() local_unnamed_addr !dbg !21 {
entry:
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (trunk 374581) (llvm/trunk 374579)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "loc2.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "91e0069c680e2a63f4f885ec93f5d07e")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (trunk 374581) (llvm/trunk 374579)"}
!7 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1ii", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !DILocalVariable(name: "i", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "j", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 2, type: !10)
!15 = !DILocalVariable(name: "y", scope: !7, file: !1, line: 3, type: !10)
!16 = !DILocalVariable(name: "r", scope: !7, file: !1, line: 4, type: !10)
!17 = !DILocalVariable(name: "undef", scope: !7, file: !1, line: 5, type: !10)
!18 = !DILocation(line: 0, scope: !7)
!19 = !DILocation(line: 4, column: 13, scope: !7)
!20 = !DILocation(line: 8, column: 3, scope: !7)
!21 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 10, type: !22, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !DILocation(line: 11, column: 1, scope: !21)
