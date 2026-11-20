; A copy from the zero register loads zero: describe it as such in call-site
; parameters, rather than by its source register. WZR and XZR are DWARF
; register 31, which is also WSP and SP, so naming the register described
; every zero argument as the stack pointer (DW_OP_breg31 WSP+0).
;
; RUN: llc -mtriple=aarch64-linux-gnu -O2 -emit-call-site-info -filetype=obj %s -o %t.o
; RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s --implicit-check-not=breg31

; CHECK:      DW_TAG_call_site
; CHECK:        DW_TAG_call_site_parameter
; CHECK-NEXT:     DW_AT_location (DW_OP_reg2 W2)
; CHECK-NEXT:     DW_AT_call_value (DW_OP_breg19 W19+0)
; CHECK:        DW_TAG_call_site_parameter
; CHECK-NEXT:     DW_AT_location (DW_OP_reg1 W1)
; CHECK-NEXT:     DW_AT_call_value (DW_OP_lit0)
; CHECK:        DW_TAG_call_site_parameter
; CHECK-NEXT:     DW_AT_location (DW_OP_reg0 W0)
; CHECK-NEXT:     DW_AT_call_value (DW_OP_lit0)
; CHECK:      DW_TAG_call_site
; CHECK:        DW_TAG_call_site_parameter
; CHECK-NEXT:     DW_AT_location (DW_OP_reg2 W2)
; CHECK-NEXT:     DW_AT_call_value (DW_OP_lit0)
; CHECK:        DW_TAG_call_site_parameter
; CHECK-NEXT:     DW_AT_location (DW_OP_reg1 W1)
; CHECK-NEXT:     DW_AT_call_value (DW_OP_lit0)
; CHECK:        DW_TAG_call_site_parameter
; CHECK-NEXT:     DW_AT_location (DW_OP_reg0 W0)
; CHECK-NEXT:     DW_AT_call_value (DW_OP_breg19 W19+0)

target triple = "aarch64-unknown-linux-gnu"

declare !dbg !20 void @callee(i32, i64, i32)

define void @caller(i32 %x) !dbg !10 {
entry:
  call void @callee(i32 0, i64 0, i32 %x), !dbg !13
  call void @callee(i32 %x, i64 0, i32 0), !dbg !14
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "zero.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!10 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DILocation(line: 2, column: 1, scope: !10)
!14 = !DILocation(line: 3, column: 1, scope: !10)
!20 = !DISubprogram(name: "callee", scope: !1, file: !1, line: 9, type: !4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
