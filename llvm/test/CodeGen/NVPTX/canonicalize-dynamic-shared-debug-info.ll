; RUN: opt -S -passes=nvptx-canonicalize-dynamic-shared < %s | FileCheck %s

; The common declaration carries the debug info of every source variable, so
; each keeps its own name and type at the shared address.

target triple = "nvptx64-nvidia-cuda"

; CHECK: @a = external addrspace(3) global [0 x i32], align 16, !dbg [[A:![0-9]+]], !dbg [[B:![0-9]+]]
; CHECK-NOT: @b =
; CHECK: [[A]] = !DIGlobalVariableExpression(var: [[AV:![0-9]+]], expr: !DIExpression())
; CHECK: [[AV]] = distinct !DIGlobalVariable(name: "integers"
; CHECK: [[B]] = !DIGlobalVariableExpression(var: [[BV:![0-9]+]], expr: !DIExpression())
; CHECK: [[BV]] = distinct !DIGlobalVariable(name: "floats"
@a = external addrspace(3) global [0 x i32], align 4, !dbg !0
@b = external addrspace(3) global [0 x float], align 16, !dbg !5

define ptx_kernel void @kernel(ptr addrspace(1) %out) !dbg !12 {
  %x = load volatile i32, ptr addrspace(3) @a, !dbg !15
  %y = load volatile float, ptr addrspace(3) @b, !dbg !15
  store i32 %x, ptr addrspace(1) %out, !dbg !15
  %p = getelementptr i32, ptr addrspace(1) %out, i32 1
  store float %y, ptr addrspace(1) %p, !dbg !15
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "integers", linkageName: "a", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: false)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !9)
!3 = !DIFile(filename: "shared.cu", directory: "/")
!4 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !16)
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "floats", linkageName: "b", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: false)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, elements: !16)
!9 = !{!0, !5}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = distinct !DISubprogram(name: "kernel", scope: !3, file: !3, line: 3, type: !13, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocation(line: 4, column: 1, scope: !12)
!16 = !{!18}
!17 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!18 = !DISubrange(count: -1)
