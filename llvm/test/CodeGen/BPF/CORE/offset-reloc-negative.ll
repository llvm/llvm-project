; RUN: llc -mtriple=bpfel -filetype=obj %s -o - | \
; RUN:   llvm-objdump -dr --no-show-raw-insn --no-print-imm-hex - | FileCheck %s

@"llvm.delta:11:-4$0" = external global i64, !llvm.preserve.access.index !3 #0
@"llvm.delta:11:4$1" = external global i64, !llvm.preserve.access.index !3 #0

define i32 @read_before(ptr %p) !dbg !7 {
  %offset = load i64, ptr @"llvm.delta:11:-4$0"
  %addr = getelementptr i8, ptr %p, i64 %offset
  %value = load i32, ptr %addr
  ret i32 %value
}

; CHECK-LABEL: <read_before>:
; CHECK: r{{[0-9]+}} = -4 ll
; CHECK-NEXT: {{.*}}CO-RE <enumval_value> {{.*}} enum delta::BEFORE = -4

define i32 @read_after(ptr %p) !dbg !10 {
  %offset = load i64, ptr @"llvm.delta:11:4$1"
  %addr = getelementptr i8, ptr %p, i64 %offset
  %value = load i32, ptr %addr
  ret i32 %value
}

; CHECK-LABEL: <read_after>:
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 4)
; CHECK-NEXT: {{.*}}CO-RE <enumval_value> {{.*}} enum delta::AFTER = 4

attributes #0 = { "btf_ama" }
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug, enums: !{!3})
!1 = !DIFile(filename: "delta.c", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "delta", file: !1, baseType: !2, size: 32, elements: !{!4, !5})
!4 = !DIEnumerator(name: "BEFORE", value: -4)
!5 = !DIEnumerator(name: "AFTER", value: 4)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2, size: 64)
!7 = distinct !DISubprogram(name: "read_before", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !{!2, !6})
!10 = distinct !DISubprogram(name: "read_after", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!11 = !{i32 2, !"Debug Info Version", i32 3}
