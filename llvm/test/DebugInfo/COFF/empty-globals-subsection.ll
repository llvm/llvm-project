; RUN: llc < %s | FileCheck %s

; The debug info contains a struct with a non-const static member.
; Check that we don't emit a symbol subsection for globals.

; CHECK-NOT:  Symbol subsection for globals

; ModuleID = 'a.cpp'
source_filename = "a.cpp"
target triple = "x86_64-pc-windows-msvc19.25.28614"

define dso_local i32 @main() {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "C:/path/to/test", checksumkind: CSK_MD5, checksum: "a1dbf3aabea9e8f9d1be48f60287942f")
!2 = !{}
!3 = !{!4}
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 5, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !5, identifier: ".?AU?$A@J@@")
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !4, file: !1, line: 6, baseType: !7, size: 32, flags: DIFlagStaticMember)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"CodeView", i32 1}
!9 = !{i32 2, !"Debug Info Version", i32 3}
