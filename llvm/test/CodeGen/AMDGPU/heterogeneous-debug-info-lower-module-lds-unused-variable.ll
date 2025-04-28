; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -verify-machineinstrs -filetype=obj %s -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_name [DW_FORM_strx1] (indexed ({{[0-9]+}}) string = "unused")
; CHECK:       DW_AT_LLVM_memory_space [DW_FORM_data4] (DW_MSPACE_LLVM_group)
; CHECK:       DW_AT_location [DW_FORM_exprloc] (<empty>)
; CHECK-NOT: DW_TAG_subprogram
; CHECK-NOT: DW_TAG_variable

@unused = external addrspace(3) global [4 x float], !dbg.def !0

define amdgpu_kernel void @foo() !dbg !1 {
  ret void
}

!llvm.module.flags = !{!11, !12}
!llvm.dbg.cu = !{!2}
!llvm.dbg.retainedNodes = !{!5}

!0 = distinct !DIFragment()
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !2, file: !3, line: 1, type: !13, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, imports: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "file.cpp", directory: "/dir")
!4 = !{}
!5 = distinct !DILifetime(object: !6, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref([4 x float])), argObjects: {!0})
!6 = distinct !DIGlobalVariable(name: "unused", scope: !1, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 128, elements: !9)
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!9 = !{!10}
!10 = !DISubrange(count: 4)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 4}
!13 = distinct !DISubroutineType(types: !14)
!14 = !{null}
