target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.CCSM = type opaque
%class.CWBD = type { float }

%"class.std::_Unique_ptr_base" = type { %class.CWBD* }

%class.CB = type { %"class.std::unique_ptr_base.1" }
; (stage1.1)
;   %class.std::unique_ptr_base.1(t1.o) is mapped to %class.std::unique_ptr_base(t0.o)
;   %class.CCSM(t1.o) is mapped to %class.CWBD(t0.o)
%"class.std::unique_ptr_base.1" = type { %class.CCSM* }

; (stage1.2)
;   %class.CCSM(t1.o) -> %class.CWBD(t0.o) mapping of stage1.1 maps this to
;   "declare void @h(%class.CWBD*)"
declare void @h(%class.CCSM*)
define void @j() {
  call void @h(%class.CCSM* undef)
  ret void
}

define void @a() {
  ; Without the fix in D87001 to delay materialization of @d until its module is linked
  ; (stage1.3)
  ;   mapping `%class.CB* undef` creates the first instance of %class.CB (%class.CB).
  ; (stage2)
  ;   mapping `!6` starts the stage2, during which second instance of %class.CB (%class.CB.1)
  ;   is created for the mapped @d declaration.
  ;       define void @d(%class.CB.1*)
  ;   After this, %class.CB (t2.o) (aka %class.CB.1) and
  ;   %"class.std::unique_ptr_base.2" (t2.o) are added to DstStructTypesSet.
  call void @llvm.dbg.value(metadata %class.CB* undef, metadata !6, metadata !DIExpression()), !dbg !4
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3)
!3 = !DIFile(filename: "f2", directory: "")

!4 = !DILocation(line: 117, column: 34, scope: !7)

; This DICompositeType refers to !5 in type-mapping-bug4.ll
!5 = !DICompositeType(tag: DW_TAG_structure_type, flags: DIFlagFwdDecl, identifier: "SHARED")

!6 = !DILocalVariable(name: "this", arg: 1, scope: !7, flags: DIFlagArtificial | DIFlagObjectPointer)
!7 = distinct !DISubprogram(name: "a", type: !8, unit: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !5}
