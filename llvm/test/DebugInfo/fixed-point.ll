;; This fixes https://github.com/llvm/llvm-project/issues/81555
; REQUIRES: object-emission
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - -verify | FileCheck %s --check-prefix=VERIFY

; VERIFY-NOT: error:

; CHECK: {{.*}}:   DW_TAG_base_type
; CHECK-NEXT:          DW_AT_name  ("var")
; CHECK-NEXT:          DW_AT_encoding  (DW_ATE_signed_fixed)
define void @func() !dbg !26 {
entry:
  %classifier = alloca i32, align 4
  tail call void @llvm.dbg.value(metadata i32 32768, metadata !37, metadata !DIExpression()), !dbg !39
  store i32 32768, ptr %classifier, align 4, !dbg !39
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a", directory: "")
!6 = !DIBasicType(name: "var", size: 32, encoding: DW_ATE_signed_fixed)
!19 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: null)
!26 = distinct !DISubprogram(unit: !0, type: !3)
!37 = !DILocalVariable(name: "intercept", arg: 2, scope: !26, file: !1, line: 7, type: !6)
!39 = !DILocation(line: 0, scope: !26)

