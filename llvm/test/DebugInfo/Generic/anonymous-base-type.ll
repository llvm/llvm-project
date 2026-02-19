;; Check that a base type can be anonymous.

; Use llvm-dis here to check round-tripping.
; RUN: llvm-as < %s | llvm-dis | llc -O0 -filetype=obj -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; CHECK: DW_TAG_base_type
; CHECK-NOT: DW_AT_name
; CHECK: DW_AT_encoding
; CHECK-NEXT: DW_AT_byte_size

; ModuleID = 'subrange_type.ll'
source_filename = "/dir/anonymous-base-type.adb"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "subrange_type.adb", directory: "/dir")
!4 = !{}
!5 = !{!6}
!6 = !DIBasicType(size: 32, encoding: DW_ATE_signed)
