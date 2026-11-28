; RUN: llc %s -mtriple=arm64e-apple-darwin -filetype=obj -o - \
; RUN:   | llvm-dwarfdump -v --debug-info - | FileCheck %s

; Check that constants for variables of ptrauth type take their signedness
; from the type wrapped by the ptrauth type.

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_const_value [DW_FORM_udata] (18446744073709551615)
; CHECK-NEXT:   DW_AT_name {{.*}} "p"
; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_const_value [DW_FORM_sdata] (-1)
; CHECK-NEXT:   DW_AT_name {{.*}} "i"

define void @f() !dbg !3 {
    #dbg_value(i64 -1, !4, !DIExpression(), !6)
    #dbg_value(i32 -1, !7, !DIExpression(), !6)
  ret void, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "f", unit: !0, type: !DISubroutineType(types: !{}), spFlags: DISPFlagDefinition)
!4 = !DILocalVariable(name: "p", scope: !3, type: !5)
!5 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64), ptrAuthKey: 1)
!6 = !DILocation(line: 1, scope: !3)
!7 = !DILocalVariable(name: "i", scope: !3, type: !8)
!8 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed), ptrAuthKey: 1)
