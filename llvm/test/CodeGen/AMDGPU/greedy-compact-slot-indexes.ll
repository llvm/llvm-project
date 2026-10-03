; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -O3 -stress-regalloc=8 \
; RUN:     -greedy-compact-slot-indexes -verify-machineinstrs -filetype=null < %s

; AMDGPU runs greedy three times with one LiveDebugVariables, so its indexes can
; point at instructions later passes erased. Emitting the DBG_VALUEs used to
; crash.

define amdgpu_kernel void @sdiv_i32_pow2_shl_denom(ptr addrspace(1) %out, i32 %x, i32 %y) !dbg !5 {
  %shl.y = shl i32 4096, %y, !dbg !12
    #dbg_value(i32 %shl.y, !9, !DIExpression(), !12)
  %r = sdiv i32 %x, %shl.y, !dbg !13
    #dbg_value(i32 %r, !11, !DIExpression(), !13)
  store i32 %r, ptr addrspace(1) %out, align 4, !dbg !14
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.ll", directory: "/")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "sdiv_i32_pow2_shl_denom", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !10)
!12 = !DILocation(line: 1, column: 1, scope: !5)
!13 = !DILocation(line: 2, column: 1, scope: !5)
!14 = !DILocation(line: 3, column: 1, scope: !5)
!15 = !DILocation(line: 4, column: 1, scope: !5)
