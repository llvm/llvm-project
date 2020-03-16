; RUN: llc -O3 -emit-call-site-info -debug-entry-values -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Verify that we produce call site entries for the zero-valued parameters.
;
; Based on the following reproducer:
;
; #include <stdint.h>
; extern void callee(uint8_t, uint16_t, uint32_t, void *);
; int caller() {
;   callee(0, 0, 0, (void *)0);
;   return 1;
; }

; CHECK: DW_TAG_GNU_call_site_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_reg2 RCX)
; CHECK-NEXT: DW_AT_GNU_call_site_value	(DW_OP_lit0)

; CHECK: DW_TAG_GNU_call_site_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_reg1 RDX)
; CHECK-NEXT: DW_AT_GNU_call_site_value	(DW_OP_lit0)

; CHECK: DW_TAG_GNU_call_site_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_reg4 RSI)
; CHECK-NEXT: DW_AT_GNU_call_site_value	(DW_OP_lit0)

; CHECK: DW_TAG_GNU_call_site_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_reg5 RDI)
; CHECK-NEXT: DW_AT_GNU_call_site_value	(DW_OP_lit0)

; Function Attrs: nounwind uwtable
define i32 @caller() #0 !dbg !15 {
entry:
  tail call void @callee(i8 zeroext 0, i16 zeroext 0, i32 0, i8* null), !dbg !19
  ret i32 1, !dbg !20
}

declare !dbg !5 void @callee(i8 zeroext, i16 zeroext, i32, i8*)

attributes #0 = { nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "zero.c", directory: "/")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !DISubprogram(name: "callee", scope: !1, file: !1, line: 2, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !9, !10, !4}
!8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!10 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 10.0.0"}
!15 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 3, type: !16, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!18}
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DILocation(line: 4, scope: !15)
!20 = !DILocation(line: 5, scope: !15)
