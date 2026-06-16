; Verify that DW_TAG_LLVM_annotation DIEs are not emitted into .debug_names,
; and that the verifier does not require them to be.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -accel-tables=Dwarf -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -verify %t | FileCheck %s
; RUN: llvm-dwarfdump --debug-names %t | FileCheck %s --check-prefix=NAMES

; CHECK: Verifying .debug_names...
; CHECK-NEXT: No errors.

; The annotation DIE is present in .debug_info (see DW_AT_name "btf_decl_tag"
; with DW_AT_const_value "tag1"), but neither the tag nor its strings should
; appear anywhere in the .debug_names index.
; NAMES:     .debug_names contents:
; NAMES-NOT: DW_TAG_LLVM_annotation
; NAMES-NOT: btf_decl_tag
; NAMES-NOT: tag1

; Source:
;   #define __tag1 __attribute__((btf_decl_tag("tag1")))
;   int g1 __tag1;

@g1 = dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g1", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true, annotations: !7)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "t.c", directory: "/tmp")
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8}
!8 = !{!"btf_decl_tag", !"tag1"}
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
