; Verify that DW_TAG_LLVM_annotation DIEs are not required to appear in
; .debug_names. 

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -accel-tables=Dwarf -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -verify %t | FileCheck %s

; CHECK: Verifying .debug_names...
; CHECK-NEXT: No errors.

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
