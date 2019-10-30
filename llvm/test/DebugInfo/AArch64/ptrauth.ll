; RUN: llc %s -filetype=obj -mtriple arm64e-apple-darwin -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_AT_type	(0x{{0+}}[[TY:.*]] "*__ptrauth(4, 1, 0x04d2)")
; CHECK: 0x{{0+}}[[TY]]: DW_TAG_APPLE_ptrauth_type
; CHECK-NEXT: DW_AT_type {{.*}}"*"
; CHECK-NEXT: DW_AT_APPLE_ptrauth_key (0x04)
; CHECK-NEXT: DW_AT_APPLE_ptrauth_address_discriminated (true)
; CHECK-NEXT: DW_AT_APPLE_ptrauth_extra_discriminator (0x04d2)

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@p = common global i8* null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "/tmp/p.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_APPLE_ptrauth_type, baseType: !9, ptrAuthKey: 4, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1234)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
