; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - -use-dwarf-ranges-base-address-specifier | FileCheck %s
; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - | FileCheck %s
; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - -dwarf-version 5 | FileCheck --check-prefix=DWARF5 %s

; CHECK: {{^.Ldebug_ranges0}}
; CHECK-NEXT:   .quad   .Ltmp0-.Lfunc_begin0
; CHECK-NEXT:   .quad   .Ltmp1-.Lfunc_begin0
; CHECK-NEXT:   .quad   .Ltmp2-.Lfunc_begin0
; CHECK-NEXT:   .quad   .Ltmp3-.Lfunc_begin0
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   0

; DWARF5: {{^.Ldebug_ranges0}}
; DWARF5-NEXT:                               # DW_RLE_offset_pair
; DWARF5-NEXT: .uleb128 .Ltmp0-.Lfunc_begin0 #   starting offset
; DWARF5-NEXT: .uleb128 .Ltmp1-.Lfunc_begin0 #   ending offset
; DWARF5-NEXT:                               # DW_RLE_offset_pair
; DWARF5-NEXT: .uleb128 .Ltmp2-.Lfunc_begin0 #   starting offset
; DWARF5-NEXT: .uleb128 .Ltmp3-.Lfunc_begin0 #   ending offset
; DWARF5-NEXT:                               # DW_RLE_end_of_list

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2f2v() !dbg !7 {
entry:
  %b = alloca i8, align 1
  call void @llvm.dbg.declare(metadata i8* %b, metadata !11, metadata !DIExpression()), !dbg !14
  call void @_Z2f1v(), !dbg !15
  call void @_Z2f1v(), !dbg !10
  call void @_Z2f1v(), !dbg !16
  ret void, !dbg !17
}

declare dso_local void @_Z2f1v()

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk 337388) (llvm/trunk 337392)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "range_reloc_base.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 337388) (llvm/trunk 337392)"}
!7 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 3, column: 3, scope: !7)
!11 = !DILocalVariable(name: "b", scope: !12, file: !1, line: 5, type: !13)
!12 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 3)
!13 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!14 = !DILocation(line: 5, column: 10, scope: !12)
!15 = !DILocation(line: 6, column: 5, scope: !12)
!16 = !DILocation(line: 7, column: 5, scope: !12)
!17 = !DILocation(line: 9, column: 1, scope: !7)
