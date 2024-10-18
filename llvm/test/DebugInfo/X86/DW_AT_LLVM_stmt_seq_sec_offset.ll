; RUN: llc -mtriple=i686-w64-mingw32 -o %t -filetype=obj %s
; RUN: llvm-dwarfdump -v -all %t | FileCheck %s -check-prefix=NO_STMT_SEQ

; RUN: llc -mtriple=i686-w64-mingw32 -o %t -filetype=obj %s -emit-func-debug-line-table-offsets
; RUN: llvm-dwarfdump -v -all %t | FileCheck %s -check-prefix=STMT_SEQ

; NO_STMT_SEQ-NOT:      DW_AT_LLVM_stmt_sequence

; STMT_SEQ:   [[[ABBREV_CODE:[0-9]+]]] DW_TAG_subprogram
; STMT_SEQ:  	       DW_AT_LLVM_stmt_sequence    DW_FORM_sec_offset
; STMT_SEQ:   DW_TAG_subprogram [[[ABBREV_CODE]]]
; STMT_SEQ:       DW_AT_LLVM_stmt_sequence [DW_FORM_sec_offset]	(0x00000028)
; STMT_SEQ:   DW_AT_name {{.*}}func01
; STMT_SEQ:   DW_TAG_subprogram [[[ABBREV_CODE]]]
; STMT_SEQ:       DW_AT_LLVM_stmt_sequence [DW_FORM_sec_offset]	(0x00000033)
; STMT_SEQ:   DW_AT_name {{.*}}main

;; Check that the line table starts at 0x00000028 (first function)
; STMT_SEQ:            Address            Line   Column File   ISA Discriminator OpIndex Flags
; STMT_SEQ-NEXT:       ------------------ ------ ------ ------ --- ------------- ------- -------------
; STMT_SEQ-NEXT:  0x00000028: 00 DW_LNE_set_address (0x00000006)

;; Check that we have an 'end_sequence' just before the next function (0x00000033)
; STMT_SEQ:            0x0000000000000006      1      0      1   0             0       0  is_stmt end_sequence
; STMT_SEQ-NEXT: 0x00000033: 00 DW_LNE_set_address (0x00000027)

;; Check that the end of the line table still has an 'end_sequence'
; STMT_SEQ       0x00000049: 00 DW_LNE_end_sequence
; STMT_SEQ-NEXT        0x0000000000000027      6      3      1   0             0       0  end_sequence


; generated from:
; clang -g -S -emit-llvm test.c -o test.ll
; ======= test.c ======
; int func01() {
;   return 1;
; }
; int main() {
;   return 0;
; }
; =====================


; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @func01() #0 !dbg !9 {
  ret i32 1, !dbg !13
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @main() #0 !dbg !14 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 0, !dbg !15
}

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "Homebrew clang version 17.0.6", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk", sdk: "MacOSX14.sdk")
!1 = !DIFile(filename: "test.c", directory: "/tmp/clang_test")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{!"Homebrew clang version 17.0.6"}
!9 = distinct !DISubprogram(name: "func01", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 2, column: 3, scope: !9)
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !10, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 6, column: 3, scope: !14)
