; RUN: llc -mtriple=bpfel -filetype=obj -o %t %s
; RUN: llvm-objdump --no-print-imm-hex -dr --no-show-raw-insn %t | FileCheck %s

; Only field-byte-offset relocations may be folded into memory offsets. Enum
; values also require full-width patch sites, including when used as shifts.

; CHECK-LABEL: <read_at_positive_offset>:
; CHECK: r[[POS_REG:[0-9]+]] = 4 ll
; CHECK-NEXT: CO-RE <enumval_value> {{\[}}[[TYPE_ID:[0-9]+]]{{\]}} enum E::POSITIVE = 4
; CHECK: r1 += r[[POS_REG]]
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 0)
; CHECK-LABEL: <read_at_negative_offset>:
; CHECK: r[[NEG_REG:[0-9]+]] = -4 ll
; CHECK-NEXT: CO-RE <enumval_value> {{\[}}[[TYPE_ID]]{{\]}} enum E::NEGATIVE = -4
; CHECK: r1 += r[[NEG_REG]]
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 0)
; CHECK-LABEL: <shift_by_enum>:
; CHECK: r[[SHIFT_REG:[0-9]+]] = 4 ll
; CHECK-NEXT: CO-RE <enumval_value> {{\[}}[[TYPE_ID]]{{\]}} enum E::POSITIVE = 4
; CHECK: r{{[0-9]+}} <<= r[[SHIFT_REG]]
; CHECK-LABEL: <read_at_type_size>:
; CHECK: r[[SIZE_REG:[0-9]+]] = 4{{$}}
; CHECK-NEXT: CO-RE <type_size> {{\[}}[[TYPE_ID]]{{\]}} enum E
; CHECK: r1 += r[[SIZE_REG]]
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 0)

@"llvm.E:11:4$0" = external global i64, !llvm.preserve.access.index !3 #0
@"llvm.E:11:-4$1" = external global i64, !llvm.preserve.access.index !3 #0
@"llvm.E:9:4$0" = external global i32, !llvm.preserve.access.index !3 #0

define i32 @read_at_positive_offset(ptr %base) !dbg !7 {
  %offset = load i64, ptr @"llvm.E:11:4$0", align 8
  %address = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %address, align 4
  ret i32 %value
}

define i32 @read_at_negative_offset(ptr %base) !dbg !10 {
  %offset = load i64, ptr @"llvm.E:11:-4$1", align 8
  %address = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %address, align 4
  ret i32 %value
}

define i64 @shift_by_enum(i64 %value) !dbg !15 {
  %shift = load i64, ptr @"llvm.E:11:4$0", align 8
  %shifted = shl i64 %value, %shift
  ret i64 %shifted
}

define i32 @read_at_type_size(ptr %base) !dbg !17 {
  %size = load i32, ptr @"llvm.E:9:4$0", align 4
  %offset = zext i32 %size to i64
  %address = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %address, align 4
  ret i32 %value
}

attributes #0 = { "btf_ama" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !{!3})
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !1, baseType: !2, size: 32, elements: !4)
!4 = !{!5, !6}
!5 = !DIEnumerator(name: "POSITIVE", value: 4)
!6 = !DIEnumerator(name: "NEGATIVE", value: -4)
!7 = distinct !DISubprogram(name: "read_at_positive_offset", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DISubroutineType(types: !{!2, !9})
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2, size: 64)
!10 = distinct !DISubprogram(name: "read_at_negative_offset", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!14 = !DISubroutineType(types: !{!13, !13})
!15 = distinct !DISubprogram(name: "shift_by_enum", scope: !1, file: !1, type: !14, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DISubprogram(name: "read_at_type_size", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
