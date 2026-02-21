; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 < %s | FileCheck %s

; Test that compiler-generated WQM/WWM exec mask manipulation instructions
; have debug locations attached.

define amdgpu_cs void @test_wqm_wwm_debug_loc(ptr addrspace(8) inreg %rsrc) !dbg !7 {
entry:
  %val = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0), !dbg !11
  %val.i = bitcast float %val to i32

  %inactive = call i32 @llvm.amdgcn.set.inactive.i32(i32 %val.i, i32 0), !dbg !12
  %dpp = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %inactive, i32 273, i32 15, i32 15, i1 false), !dbg !13
  %sum = add i32 %inactive, %dpp, !dbg !13
  %wwm = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %sum), !dbg !14

  %wqm = call i32 @llvm.amdgcn.strict.wqm.i32(i32 %wwm), !dbg !15

  %result = bitcast i32 %wqm to float
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %result, ptr addrspace(8) %rsrc, i32 4, i32 0, i32 0), !dbg !16
  ret void, !dbg !17
}

; Verify .loc directives appear before exec mask operations:
; - s_or_saveexec (ENTER_STRICT_WWM) should have line 5 debug loc
; - s_mov_b32 exec_lo (EXIT_STRICT_WWM/WQM) should have debug locs
; CHECK: .loc {{[0-9]+}} 5 10
; CHECK-NEXT: s_or_saveexec_b32
; CHECK: .loc {{[0-9]+}} 7 10
; CHECK-NEXT: s_mov_b32 exec_lo
; CHECK: .loc {{[0-9]+}} 10 10
; CHECK: s_mov_b32 exec_lo

declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32)
declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)
declare i32 @llvm.amdgcn.strict.wwm.i32(i32)
declare i32 @llvm.amdgcn.strict.wqm.i32(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "wqm_wwm_test.cl", directory: "/tmp")
!2 = !{}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "test_wqm_wwm_debug_loc", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!11 = !DILocation(line: 3, column: 10, scope: !7)
!12 = !DILocation(line: 5, column: 10, scope: !7)
!13 = !DILocation(line: 6, column: 10, scope: !7)
!14 = !DILocation(line: 7, column: 10, scope: !7)
!15 = !DILocation(line: 10, column: 10, scope: !7)
!16 = !DILocation(line: 13, column: 3, scope: !7)
!17 = !DILocation(line: 14, column: 1, scope: !7)
