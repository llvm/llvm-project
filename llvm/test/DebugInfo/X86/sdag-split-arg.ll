; RUN: llc -O0 -filetype=asm %s -o - | FileCheck %s
; Test large integral function arguments passed in multiple registers.
; CHECK: DEBUG_VALUE: foo:bar <- [DW_OP_LLVM_fragment 64 16] %ax
; CHECK: DEBUG_VALUE: foo:bar <- [DW_OP_LLVM_fragment 48 16] %r9w
; CHECK: DEBUG_VALUE: foo:bar <- [DW_OP_LLVM_fragment 32 16] %r10w
; CHECK: DEBUG_VALUE: foo:bar <- [DW_OP_LLVM_fragment 16 16] %r11w
; CHECK: DEBUG_VALUE: foo:bar <- [DW_OP_LLVM_fragment 0 16] %bx

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

%rec789 = type { [5 x i16] }

define void @foo(%rec789 %bar) !dbg !6 {
  %bar.2 = alloca %rec789, align 1
  call void @llvm.dbg.value(metadata %rec789 %bar, metadata !17, metadata !DIExpression()), !dbg !18
  %1 = extractvalue %rec789 %bar, 0
  %.repack = getelementptr inbounds %rec789, %rec789* %bar.2, i16 0, i32 0, i16 0
  %.elt = extractvalue [5 x i16] %1, 0
  store i16 %.elt, i16* %.repack, align 1
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #6

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "a.c", directory: "b")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 13, type: !7, isLocal: false, isDefinition: true, scopeLine: 14, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "MyStruct", file: !1, line: 11, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 9, size: 80, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "Array", scope: !10, file: !1, line: 10, baseType: !13, size: 80)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 80, elements: !15)
!14 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DISubrange(count: 5)
!17 = !DILocalVariable(name: "bar", arg: 1, scope: !6, line: 13, type: !9)
!18 = !DILocation(line: 13, column: 23, scope: !6)
!19 = !DILocation(line: 15, column: 5, scope: !6)
!20 = !DILocation(line: 16, column: 1, scope: !6)
