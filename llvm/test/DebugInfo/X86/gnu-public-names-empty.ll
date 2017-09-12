; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections -filetype=obj < %s | llvm-dwarfdump -v - | FileCheck %s

; Generated from:

; static int a __attribute__((section("a")));

; Check that the attributes in the compile unit both point to a correct
; location, even when nothing is exported.
; CHECK: DW_AT_GNU_pubnames [DW_FORM_flag_present]   (true)
; CHECK-NOT: DW_AT_GNU_pubtypes [

; CHECK: .debug_gnu_pubnames contents:
; CHECK-NEXT: length = 0x0000000e
; CHECK-NEXT: Offset
; CHECK-NEXT: {{^$}}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 191846) (llvm/trunk 191866)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !5, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
