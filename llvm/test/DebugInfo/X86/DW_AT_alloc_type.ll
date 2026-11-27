; RUN: llc -O3 -o %t -filetype=obj %s
; RUN: llvm-dwarfdump %t | FileCheck %s

; based on clang++ output for `int *alloc_int() { return new int; }`


target triple = "x86_64-unknown-linux-gnu"

define dso_local ptr @alloc_int() !dbg !3 {
; CHECK: DW_TAG_subprogram
entry:
  %call = call ptr @alloc(i64 noundef 4), !heapallocsite !7
; CHECK: DW_TAG_call_site
; CHECK: DW_AT_LLVM_alloc_type ([[ALLOCSITE:.*]])
  ret ptr %call
}

; CHECK: {{.*}}[[ALLOCSITE]]: DW_TAG_base_type
; CHECK: DW_AT_name ("int")

declare dso_local ptr @alloc(i64 noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2,!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "alloc_int", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 5}
