; This test checks that strip-dead-debug-info pass doesn't delete debug compile
; units if they are used by @llvm.dbg.* intrinsics

; RUN: opt -passes='strip-dead-debug-info,verify' %s -S | FileCheck %s

; CHECK: !llvm.dbg.cu = !{!{{[0-9]+}}, !{{[0-9]+}}}
; CHECK-COUNT-2: !DICompileUnit

declare void @llvm.dbg.value(metadata, metadata, metadata)

define void @func() {
  %a = alloca i64
  call void @llvm.dbg.value(metadata ptr %a, metadata !7, metadata !DIExpression()), !dbg !9
  ret void
}

!llvm.dbg.cu = !{!0, !1}
!llvm.module.flags = !{!10}


; We have two different compile units to able to check different paths of
; finding compile units (intrinsic argument and attached location to instruction)
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !11)
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !11)

!2 = distinct !DISubprogram(name: "func", unit: !0)
!3 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !2)
!4 = !DIDerivedType(tag: DW_TAG_member, scope: !3, baseType: !5)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6)
!6 = !DIBasicType(tag: DW_TAG_base_type)
!7 = !DILocalVariable(name: "a", type: !5, scope: !8)
!8 = distinct !DISubprogram(name: "func", unit: !1)
!9 = !DILocation(scope: !8)

!10 = !{i32 2, !"Debug Info Version", i32 3}

!11 = !DIFile(filename: "a", directory: "")
