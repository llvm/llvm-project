; RUN: llc -O3 -mtriple=i686-w64-mingw32 -o %t_no -filetype=obj %s
; RUN: llvm-dwarfdump -v -all %t_no | FileCheck %s -check-prefix=NO_STMT_SEQ

; RUN: llc -O3 -mtriple=i686-w64-mingw32 -o %t_yes -filetype=obj %s -emit-func-debug-line-table-offsets
; RUN: llvm-dwarfdump -v -all %t_yes | FileCheck %s -check-prefix=STMT_SEQ

; NO_STMT_SEQ-NOT:      DW_AT_LLVM_stmt_sequence

; STMT_SEQ:   [[[ABBREV_CODE1:[0-9]+]]] DW_TAG_subprogram
; STMT_SEQ:  	       DW_AT_LLVM_stmt_sequence    DW_FORM_sec_offset
; STMT_SEQ:   [[[ABBREV_CODE2:[0-9]+]]] DW_TAG_subprogram
; STMT_SEQ:  	       DW_AT_LLVM_stmt_sequence    DW_FORM_sec_offset
; STMT_SEQ:   DW_TAG_subprogram [[[ABBREV_CODE1]]]
; STMT_SEQ:       DW_AT_LLVM_stmt_sequence [DW_FORM_sec_offset]	(0x00000043)
; STMT_SEQ:   DW_AT_name {{.*}}func01
; STMT_SEQ:   DW_TAG_subprogram [[[ABBREV_CODE2]]]
; STMT_SEQ:       DW_AT_LLVM_stmt_sequence [DW_FORM_sec_offset]	(0x00000058)
; STMT_SEQ:   DW_AT_name {{.*}}main

;; Check the entire line sequence to see that it's correct
; STMT_SEQ:                   Address            Line   Column File   ISA Discriminator OpIndex Flags
; STMT_SEQ-NEXT:              ------------------ ------ ------ ------ --- ------------- ------- -------------
; STMT_SEQ-NEXT:  0x00000043: 04 DW_LNS_set_file (0)
; STMT_SEQ-NEXT:  0x00000045: 05 DW_LNS_set_column (9)
; STMT_SEQ-NEXT:  0x00000047: 0a DW_LNS_set_prologue_end
; STMT_SEQ-NEXT:  0x00000048: 00 DW_LNE_set_address (0x00000000)
; STMT_SEQ-NEXT:  0x0000004f: 16 address += 0,  line += 4,  op-index += 0
; STMT_SEQ-NEXT:              0x0000000000000000      5      9      0   0             0       0  is_stmt prologue_end
; STMT_SEQ-NEXT:  0x00000050: 05 DW_LNS_set_column (3)
; STMT_SEQ-NEXT:  0x00000052: 67 address += 6,  line += 1,  op-index += 0
; STMT_SEQ-NEXT:              0x0000000000000006      6      3      0   0             0       0  is_stmt
; STMT_SEQ-NEXT:  0x00000053: 02 DW_LNS_advance_pc (addr += 2, op-index += 0)
; STMT_SEQ-NEXT:  0x00000055: 00 DW_LNE_end_sequence
; STMT_SEQ-NEXT:              0x0000000000000008      6      3      0   0             0       0  is_stmt end_sequence
; STMT_SEQ-NEXT:  0x00000058: 04 DW_LNS_set_file (0)
; STMT_SEQ-NEXT:  0x0000005a: 00 DW_LNE_set_address (0x00000008)
; STMT_SEQ-NEXT:  0x00000061: 03 DW_LNS_advance_line (10)
; STMT_SEQ-NEXT:  0x00000063: 01 DW_LNS_copy
; STMT_SEQ-NEXT:              0x0000000000000008     10      0      0   0             0       0  is_stmt
; STMT_SEQ-NEXT:  0x00000064: 05 DW_LNS_set_column (10)
; STMT_SEQ-NEXT:  0x00000066: 0a DW_LNS_set_prologue_end
; STMT_SEQ-NEXT:  0x00000067: 83 address += 8,  line += 1,  op-index += 0
; STMT_SEQ-NEXT:              0x0000000000000010     11     10      0   0             0       0  is_stmt prologue_end
; STMT_SEQ-NEXT:  0x00000068: 05 DW_LNS_set_column (3)
; STMT_SEQ-NEXT:  0x0000006a: 9f address += 10,  line += 1,  op-index += 0
; STMT_SEQ-NEXT:              0x000000000000001a     12      3      0   0             0       0  is_stmt
; STMT_SEQ-NEXT:  0x0000006b: 02 DW_LNS_advance_pc (addr += 5, op-index += 0)
; STMT_SEQ-NEXT:  0x0000006d: 00 DW_LNE_end_sequence
; STMT_SEQ-NEXT:              0x000000000000001f     12      3      0   0             0       0  is_stmt end_sequence

; generated from:
; clang -Oz -g -S -emit-llvm test.c -o test.ll
; ======= test.c ======
; volatile int g_var1 = 1;
; #define ATTR __attribute__((noinline))
; ATTR int func01()  {
;   g_var1++;
;   func01();
;   return 1;
; }
; ATTR int main() {
;   g_var1 = 100;
;   func01();
;   g_var1--;
;   return g_var1;
; }
; =====================


; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@g_var1 = dso_local global i32 1, align 4, !dbg !0
; Function Attrs: minsize nofree noinline norecurse noreturn nounwind optsize memory(readwrite, argmem: none) uwtable
define dso_local noundef i32 @func01() local_unnamed_addr #0 !dbg !14 {
entry:
  br label %tailrecurse
tailrecurse:                                      ; preds = %tailrecurse, %entry
  %0 = load volatile i32, ptr @g_var1, align 4, !dbg !17, !tbaa !18
  %inc = add nsw i32 %0, 1, !dbg !17
  store volatile i32 %inc, ptr @g_var1, align 4, !dbg !17, !tbaa !18
  br label %tailrecurse, !dbg !22
}
; Function Attrs: minsize nofree noinline norecurse noreturn nounwind optsize uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 !dbg !23 {
entry:
  store volatile i32 100, ptr @g_var1, align 4, !dbg !24, !tbaa !18
  %call = tail call i32 @func01() #2, !dbg !25
  unreachable, !dbg !26
}
attributes #0 = { minsize nofree noinline norecurse noreturn nounwind optsize memory(readwrite, argmem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { minsize nofree noinline norecurse noreturn nounwind optsize uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { minsize optsize }
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_var1", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/tst", checksumkind: CSK_MD5, checksum: "eee003eb3c4fd0a1ff078d3148679e06")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{!"clang version 20.0.0"}
!14 = distinct !DISubprogram(name: "func01", scope: !3, file: !3, line: 4, type: !15, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!6}
!17 = !DILocation(line: 5, column: 9, scope: !14)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 6, column: 3, scope: !14)
!23 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 10, type: !15, scopeLine: 10, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!24 = !DILocation(line: 11, column: 10, scope: !23)
!25 = !DILocation(line: 12, column: 3, scope: !23)
!26 = !DILocation(line: 13, column: 9, scope: !23)
