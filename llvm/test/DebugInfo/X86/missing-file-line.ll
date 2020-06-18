
; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o - | llvm-dwarfdump -all - | FileCheck %s

; Test that we accept and generate DWARF entities for DW_TAG_structure_type,
; DW_TAG_member and DW_TAG_typedef with no source location. These can come up
; in some languages with predefined types.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { %struct.S* }

define void @f() !dbg !4 {
  %x = alloca %struct.S, align 8
  ; CHECK: DW_TAG_typedef
  ; CHECK-NOT: DW_AT_decl_file
  ; CHECK-NOT: DW_AT_decl_line

  ; CHECK: DW_TAG_structure_type
  ; CHECK-NOT: DW_AT_decl_file
  ; CHECK-NOT: DW_AT_decl_line

  ; CHECK: DW_TAG_member
  ; CHECK-NOT: DW_AT_decl_file
  ; CHECK-NOT: DW_AT_decl_line

  ; CHECK: {{DW_TAG|NULL}}
  call void @llvm.dbg.declare(metadata %struct.S* %x, metadata !10, metadata !16), !dbg !17
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "file.c", directory: "/dir")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang"}
!10 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 8, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "SS", baseType: !12)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", size: 64, align: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !12, baseType: !15, size: 64, align: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!16 = !DIExpression()
!17 = !DILocation(line: 8, column: 6, scope: !4)
!18 = !DILocation(line: 9, column: 1, scope: !4)
