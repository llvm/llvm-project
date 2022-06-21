; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Check that any type can have a vtable holder.
; CHECK: [[SP:.*]]: DW_TAG_structure_type
; CHECK-NOT: TAG
; CHECK: DW_AT_containing_type [DW_FORM_ref4] ({{.*}} "int")
; CHECK: DW_AT_name {{.*}}"vtable"

; The code doesn't actually matter.
define i32 @main() #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

attributes #0 = { nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, producer: "clang version 3.4 (trunk 185475)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !15, imports: !2)
!1 = !DIFile(filename: "CodeGen/dwarf-version.c", directory: "test")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", line: 6, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, isOptimized: false, unit: !0, scopeLine: 6, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "CodeGen/dwarf-version.c", directory: "test")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !DILocation(line: 7, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 3}
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "vtable", size: 8, align: 8, elements: !2, identifier: "vtable", vtableHolder: !8)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = !DIGlobalVariable(name: "vtable", linkageName: "vtable", scope: null, file: !1, line: 1, type: !12, isLocal: true, isDefinition: true)
!15 = !{!13}
