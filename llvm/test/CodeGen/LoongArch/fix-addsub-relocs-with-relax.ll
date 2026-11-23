; RUN: llc --filetype=obj --mtriple=loongarch64 %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s
; RUN: llc --filetype=obj --mtriple=loongarch64 --mattr=+relax %s -o %t.r
; RUN: llvm-readobj -r %t.r | FileCheck %s

; CHECK:      Relocations [
; CHECK-NEXT:   Section ({{.*}}) .rela.text {
; CHECK-NEXT:     0x14 R_LARCH_PCALA_HI20 sym 0x0
; CHECK-NEXT:     0x14 R_LARCH_RELAX - 0x0
; CHECK-NEXT:     0x18 R_LARCH_PCALA_LO12 sym 0x0
; CHECK-NEXT:     0x18 R_LARCH_RELAX - 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Section ({{.*}}) .rela.debug_info {
; CHECK-NEXT:     0x8 R_LARCH_32 .debug_abbrev 0x0
; CHECK-NEXT:     0x11 R_LARCH_32 .L0 0x0
; CHECK-NEXT:     0x15 R_LARCH_32 .Lline_table_start0 0x0
; CHECK-NEXT:     0x1B R_LARCH_ADD32 .L0 0x0
; CHECK-NEXT:     0x1B R_LARCH_SUB32 .L0 0x0
; CHECK-NEXT:     0x1F R_LARCH_32 .L0 0x0
; CHECK-NEXT:     0x25 R_LARCH_ADD32 .L0 0x0
; CHECK-NEXT:     0x25 R_LARCH_SUB32 .L0 0x0
; CHECK-NEXT:   }
; CHECK:        Section ({{.*}}) .rela.debug_frame {
; CHECK-NEXT:     0x1C R_LARCH_32 .L0 0x0
; CHECK-NEXT:     0x20 R_LARCH_64 .L0 0x0
; CHECK-NEXT:     0x28 R_LARCH_ADD64 .L0 0x0
; CHECK-NEXT:     0x28 R_LARCH_SUB64 .L0 0x0
; CHECK-NEXT:     0x3F R_LARCH_ADD6 .L0 0x0
; CHECK-NEXT:     0x3F R_LARCH_SUB6 .L0 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Section ({{.*}}) .rela.debug_line {
; CHECK-NEXT:     0x22 R_LARCH_32 .debug_line_str 0x0
; CHECK-NEXT:     0x31 R_LARCH_32 .debug_line_str 0x2
; CHECK-NEXT:     0x46 R_LARCH_32 .debug_line_str 0x9
; CHECK-NEXT:     0x4F R_LARCH_64 .L0 0x0
; CHECK-NEXT:     0x5F R_LARCH_ADD16 .L0 0x0
; CHECK-NEXT:     0x5F R_LARCH_SUB16 .L0 0x0
; CHECK-NEXT:   }
; CHECK-NEXT: ]

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "loongarch64"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @foo() #0 !dbg !8 {
  call void asm sideeffect ".cfi_remember_state\0A\09.cfi_adjust_cfa_offset 16\0A\09nop\0A\09la.pcrel $$t0, sym\0A\09nop\0A\09.cfi_restore_state\0A\09", ""() #1, !dbg !12, !srcloc !13
  ret i32 0, !dbg !14
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="loongarch64" "target-features"="+64bit,+d,+f,+ual,+relax" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: ".", checksumkind: CSK_MD5, checksum: "f44d6d71bc4da58b4abe338ca507c007", source: "int foo()\0A{\0A  asm volatile(\0A    \22.cfi_remember_state\\n\\t\22\0A    \22.cfi_adjust_cfa_offset 16\\n\\t\22\0A    \22nop\\n\\t\22\0A    \22la.pcrel $t0, sym\\n\\t\22\0A    \22nop\\n\\t\22\0A    \22.cfi_restore_state\\n\\t\22);\0A  return 0;\0A}\0A")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"direct-access-external-data", i32 0}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !8)
!13 = !{i64 34, i64 56, i64 92, i64 106, i64 134, i64 148, i64 177}
!14 = !DILocation(line: 10, column: 3, scope: !8)
