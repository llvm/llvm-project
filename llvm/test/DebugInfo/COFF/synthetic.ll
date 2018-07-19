; RUN: llc -mtriple x86_64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

define dllexport void ()* @f() !dbg !6 {
entry:
  ret void ()* null, !dbg !28
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/Users/compnerd/Source/llvm", checksumkind: CSK_MD5, checksum: "2851eea4f12e754f1a68c47a7045406a")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DICompositeType(tag: DW_TAG_structure_type, scope: !1, size: 256, flags: DIFlagAppleBlock, elements: !11)
!11 = !{!12, !14, !16, !17, !21}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "__isa", scope: !1, file: !1, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "__flags", scope: !1, file: !1, baseType: !15, size: 32, offset: 64)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "__reserved", scope: !1, file: !1, baseType: !15, size: 32, offset: 96)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr", scope: !1, file: !1, baseType: !18, size: 64, offset: 128)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "__descriptor", scope: !1, baseType: !22, size: 64, align: 64, offset: 192)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor", scope: !1, size: 64, flags: DIFlagAppleBlock, elements: !24)
!24 = !{!25, !27}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", scope: !1, file: !1, baseType: !26, size: 32)
!26 = !DIBasicType(name: "long unsigned int", size: 32, encoding: DW_ATE_unsigned)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "Size", scope: !1, file: !1, baseType: !26, size: 32, offset: 32)
!28 = !DILocation(line: 1, scope: !6)

; CHECK: # Struct
; CHECK: #   TypeLeafKind: LF_STRUCTURE
; CHECK: #   MemberCount: 0
; CHECK: #   Properties [
; CHECK: #     ForwardReference
; CHECK: #   ]
; CHECK: #   FieldList: 0x0
; CHECK: #   DerivedFrom: 0x0
; CHECK: #   VShape: 0x0
; CHECK: #   SizeOf: 0
; CHECK: #   Name: __block_descriptor
; CHECK: # }

