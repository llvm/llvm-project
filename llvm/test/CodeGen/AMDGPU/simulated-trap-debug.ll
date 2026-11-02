; RUN: llc -global-isel=0 -mtriple=amdgpu11.00-amd-amdhsa < %s | FileCheck %s
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu11.00-amd-amdhsa < %s | FileCheck %s

declare void @llvm.trap()

define amdgpu_kernel void @simulated_trap_debug(i1 %continue) !dbg !4 {
; CHECK-LABEL: simulated_trap_debug:
; CHECK-NOT:   s_cbranch_execnz
; CHECK:       s_trap 2
; CHECK:       s_sethalt 5
entry:
  br i1 %continue, label %exit, label %trap

trap:
  call void @llvm.trap(), !dbg !8
    #dbg_value(i1 false, !7, !DIExpression(), !8)
  unreachable

exit:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "simulated-trap-debug.ll", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "simulated_trap_debug", linkageName: "simulated_trap_debug", scope: null, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !2)
!6 = !{!7}
!7 = !DILocalVariable(name: "value", scope: !4, file: !1, line: 1, type: !9)
!8 = !DILocation(line: 1, column: 1, scope: !4)
!9 = !DIBasicType(name: "bool", size: 1, encoding: DW_ATE_boolean)
