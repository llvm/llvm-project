; RUN: llc %s -filetype=obj -mtriple arm64e-apple-darwin -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_AT_type	(0x{{0+}}[[TY:.*]] "void *__ptrauth(4, 0, 0x04d2)")
; CHECK: 0x{{0+}}[[TY]]: DW_TAG_LLVM_ptrauth_type
; CHECK-NEXT: DW_AT_type {{.*}}"void *"
; CHECK-NEXT: DW_AT_LLVM_ptrauth_key (0x04)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_extra_discriminator (0x04d2)

; CHECK: DW_AT_type	(0x{{0+}}[[TY:.*]] "void *__ptrauth(4, 1, 0x04d3)")
; CHECK: 0x{{0+}}[[TY]]: DW_TAG_LLVM_ptrauth_type
; CHECK-NEXT: DW_AT_type {{.*}}"void *"
; CHECK-NEXT: DW_AT_LLVM_ptrauth_key (0x04)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_address_discriminated (true)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_extra_discriminator (0x04d3)

; CHECK: DW_AT_type	(0x{{0+}}[[TY:.*]] "void *__ptrauth(4, 1, 0x04d4, "isa-pointer")")
; CHECK: 0x{{0+}}[[TY]]: DW_TAG_LLVM_ptrauth_type
; CHECK-NEXT: DW_AT_type {{.*}}"void *"
; CHECK-NEXT: DW_AT_LLVM_ptrauth_key (0x04)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_address_discriminated (true)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_extra_discriminator (0x04d4)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_isa_pointer	(true)

; CHECK: DW_AT_type	(0x{{0+}}[[TY:.*]] "void *__ptrauth(4, 1, 0x04d5, "authenticates-null-values")")
; CHECK: 0x{{0+}}[[TY]]: DW_TAG_LLVM_ptrauth_type
; CHECK-NEXT: DW_AT_type {{.*}}"void *"
; CHECK-NEXT: DW_AT_LLVM_ptrauth_key (0x04)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_address_discriminated (true)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_extra_discriminator (0x04d5)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_authenticates_null_values	(true)

; CHECK: DW_AT_type	(0x{{0+}}[[TY:.*]] "void *__ptrauth(4, 1, 0x04d6, "isa-pointer,authenticates-null-values")")
; CHECK: 0x{{0+}}[[TY]]: DW_TAG_LLVM_ptrauth_type
; CHECK-NEXT: DW_AT_type {{.*}}"void *"
; CHECK-NEXT: DW_AT_LLVM_ptrauth_key (0x04)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_address_discriminated (true)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_extra_discriminator (0x04d6)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_isa_pointer	(true)
; CHECK-NEXT: DW_AT_LLVM_ptrauth_authenticates_null_values	(true)

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@p = global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!19, !20}

!0 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!1 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!2 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!3 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!4 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "p1", scope: !10, file: !11, line: 1, type: !14, isLocal: false, isDefinition: true)
!6 = distinct !DIGlobalVariable(name: "p2", scope: !10, file: !11, line: 1, type: !15, isLocal: false, isDefinition: true)
!7 = distinct !DIGlobalVariable(name: "p3", scope: !10, file: !11, line: 1, type: !16, isLocal: false, isDefinition: true)
!8 = distinct !DIGlobalVariable(name: "p4", scope: !10, file: !11, line: 1, type: !17, isLocal: false, isDefinition: true)
!9 = distinct !DIGlobalVariable(name: "p5", scope: !10, file: !11, line: 1, type: !18, isLocal: false, isDefinition: true)
!10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !11, emissionKind: FullDebug, globals: !13)
!11 = !DIFile(filename: "/tmp/p.c", directory: "/")
!12 = !{}
!13 = !{!0,!1,!2,!3,!4}
!14 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !21, ptrAuthKey: 4, ptrAuthIsAddressDiscriminated: false, ptrAuthExtraDiscriminator: 1234)
!15 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !21, ptrAuthKey: 4, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1235)
!16 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !21, ptrAuthKey: 4, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1236, ptrAuthIsaPointer: true)
!17 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !21, ptrAuthKey: 4, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1237, ptrAuthAuthenticatesNullValues: true)
!18 = !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type, baseType: !21, ptrAuthKey: 4, ptrAuthIsAddressDiscriminated: true, ptrAuthExtraDiscriminator: 1238, ptrAuthIsaPointer: true, ptrAuthAuthenticatesNullValues: true)
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
