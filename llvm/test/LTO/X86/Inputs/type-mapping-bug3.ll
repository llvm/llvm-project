target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; T2 is the non-opaque struct required to trigger the uniqued T2.0 and T3.0 to
; respectively T2 and T3 in the destination module.
%"T2" = type { ptr }
%"T3" = type opaque

; Use/refer to T2 so it gets added as an IdentifiedStructType.  The debug
; reference to !6 is required to transitively load the metadata !5.
define void @a(%"T2") !dbg !6 {
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, retainedTypes: !4)
!3 = !DIFile(filename: "f2", directory: "")
!4 = !{!5}

; This DICompositeType refers to !5 in type-mapping-bug3.ll
!5 = !DICompositeType(tag: DW_TAG_structure_type, flags: DIFlagFwdDecl, identifier: "SHARED")

!6 = distinct !DISubprogram(unit: !2)
