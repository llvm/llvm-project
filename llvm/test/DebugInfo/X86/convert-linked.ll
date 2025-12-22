; RUN: llc -mtriple=x86_64 -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_TAG_compile_unit
; CHECK: [[CU0BT0:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_ATE_signed_8
; CHECK: [[CU0BT1:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_ATE_signed_32
; CHECK: DW_TAG_variable
; CHECK: DW_OP_convert ([[CU0BT0]]) "DW_ATE_signed_8", DW_OP_convert ([[CU0BT1]]) "DW_ATE_signed_32"

; CHECK: DW_TAG_compile_unit
; CHECK: [[CU1BT0:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_ATE_signed_8
; CHECK: [[CU1BT1:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_ATE_signed_16
; CHECK: DW_TAG_variable
; CHECK: DW_OP_convert ([[CU1BT0]]) "DW_ATE_signed_8", DW_OP_convert ([[CU1BT1]]) "DW_ATE_signed_16"

define dso_local signext i8 @foo(i8 signext %x) !dbg !9 {
entry:
  call void @llvm.dbg.value(metadata i8 %x, metadata !13, metadata !DIExpression()), !dbg !14
;; This test depends on "convert" surviving all the way to the final object.
;; So, insert something before DW_OP_LLVM_convert that the expression folder
;; will not attempt to eliminate.  At the moment, only "convert" ops are folded.
  call void @llvm.dbg.value(metadata i8 32, metadata !15, metadata !DIExpression(DW_OP_lit0, DW_OP_plus, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value)), !dbg !17
  ret i8 %x, !dbg !18
}

define dso_local signext i8 @bar(i8 signext %x) !dbg !19 {
entry:
  call void @llvm.dbg.value(metadata i8 %x, metadata !20, metadata !DIExpression()), !dbg !21
;; This test depends on "convert" surviving all the way to the final object.
;; So, insert something before DW_OP_LLVM_convert that the expression folder
;; will not attempt to eliminate.  At the moment, only "convert" ops are folded.
  call void @llvm.dbg.value(metadata i8 32, metadata !22, metadata !DIExpression(DW_OP_lit0, DW_OP_plus, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 16, DW_ATE_signed, DW_OP_stack_value)), !dbg !24
  ret i8 %x, !dbg !25
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "dbg-foo.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "b35f80a032deb2a30bc187d564b5a775")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4, producer: "clang version 9.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!4 = !DIFile(filename: "dbg-bar.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "9836bb594260d883960455e7d8bc51ea")
!5 = !{!"clang version 9.0.0 "}
!6 = !{i32 2, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 7, type: !10, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!13 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 7, type: !12)
!14 = !DILocation(line: 7, column: 29, scope: !9)
!15 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 9, type: !16)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 9, column: 14, scope: !9)
!18 = !DILocation(line: 10, column: 3, scope: !9)
!19 = distinct !DISubprogram(name: "bar", scope: !4, file: !4, line: 1, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!20 = !DILocalVariable(name: "x", arg: 1, scope: !19, file: !4, line: 1, type: !12)
!21 = !DILocation(line: 1, column: 29, scope: !19)
!22 = !DILocalVariable(name: "z", scope: !19, file: !4, line: 3, type: !23)
!23 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!24 = !DILocation(line: 3, column: 16, scope: !19)
!25 = !DILocation(line: 4, column: 3, scope: !19)
