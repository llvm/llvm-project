; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck %s

; CHECK-LABEL:  test:
; CHECK:        .loc    1 8 16 prologue_end             ; test.py:8:16
; CHECK-NEXT:   s_load_dword

define void @test(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg writeonly captures(none) %1) local_unnamed_addr !dbg !4 {
  %3 = load <1 x float>, ptr addrspace(1) %0, align 4, !dbg !8, !amdgpu.noclobber !6
  store <1 x float> %3, ptr addrspace(1) %1, align 4, !dbg !7

  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test.py", directory: "/path")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!4 = distinct !DISubprogram(name: "test", linkageName: "test", scope: !1, file: !1, line: 7, type: !5, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 9, column: 20, scope: !4)
!8 = !DILocation(line: 8, column: 16, scope: !4)
!9 = !DILocation(line: 9, column: 4, scope: !4)
