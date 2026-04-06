; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O0 < %s | FileCheck %s

; Test that constant materialization instructions get debug locations.

define amdgpu_kernel void @test_constant_debug_loc(ptr addrspace(1) %out, i32 %x) !dbg !7 {
entry:
  ; Compare against constant 15 - this will generate S_MOV_B32 to materialize the constant
  %cmp = icmp ugt i32 %x, 15, !dbg !11
  %sel = select i1 %cmp, i32 100, i32 200, !dbg !12
  store i32 %sel, ptr addrspace(1) %out, align 4, !dbg !13
  ret void, !dbg !14
}

; Verify constant 15 is materialized under the .loc scope from line 5
; CHECK: .loc	1 5 14
; CHECK: s_mov_b32 s{{[0-9]+}}, 15
; CHECK: .loc	1 6 10

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cl", directory: "/tmp")
!2 = !{}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "test_constant_debug_loc", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 5, column: 14, scope: !7)
!12 = !DILocation(line: 6, column: 10, scope: !7)
!13 = !DILocation(line: 7, column: 3, scope: !7)
!14 = !DILocation(line: 8, column: 1, scope: !7)

