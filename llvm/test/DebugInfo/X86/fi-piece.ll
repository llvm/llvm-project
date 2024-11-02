; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s
; Test that multi-DW_OP_piece expressions are emitted for FI variables.
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin
; CHECK: DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_exprloc]	(DW_OP_fbreg -4, DW_OP_piece 0x2, DW_OP_fbreg -8, DW_OP_piece 0x2)
; CHECK-NEXT:   DW_AT_abstract_origin {{.*}}"a"
; Inlined variable, not to be merged.
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:   DW_TAG_variable
; CHECK-NEXT:     DW_AT_location
; CHECK-NEXT:   DW_AT_abstract_origin {{.*}}"a"

; ModuleID = '/tmp/t.c'
source_filename = "/tmp/t.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline nounwind optnone ssp uwtable
define void @f() #0 !dbg !8 {
entry:
  %a = alloca i16, align 4
  %b = alloca i16, align 4
  call void @llvm.dbg.declare(metadata ptr %a, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 16)), !dbg !14
  store i16 1, ptr %a, align 4, !dbg !14
  call void @llvm.dbg.declare(metadata ptr %b, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 16, 16)), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %a, metadata !11, metadata !13), !dbg !17
  store i16 2, ptr %b, align 4, !dbg !17
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone ssp uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "adrian", emissionKind: FullDebug)
!1 = !DIFile(filename: "/tmp/t.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIExpression()
!14 = !DILocation(line: 2, column: 7, scope: !8)
!15 = !DILocalVariable(name: "b", scope: !8, file: !1, line: 3, type: !12)
!16 = !DILocation(line: 3, column: 7, scope: !8)
!17 = !DILocation(line: 3, column: 7, scope: !8, inlinedAt: !16)
