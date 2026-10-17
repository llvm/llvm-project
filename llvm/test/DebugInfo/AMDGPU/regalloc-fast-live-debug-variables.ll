; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -vgpr-regalloc=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -wwm-regalloc=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -wwm-regalloc=fast -vgpr-regalloc=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -sgpr-regalloc=fast -vgpr-regalloc=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -sgpr-regalloc=fast -wwm-regalloc=fast -vgpr-regalloc=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -enable-new-pm -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -vgpr-regalloc-npm=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -enable-new-pm -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -wwm-regalloc-npm=fast -stop-before=prolog-epilog < %s | FileCheck %s
; RUN: llc -enable-new-pm -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs -wwm-regalloc-npm=fast -vgpr-regalloc-npm=fast -stop-before=prolog-epilog < %s | FileCheck %s

; LiveDebugVariables must emit the values stashed by an earlier allocation
; stage before fast allocation invalidates its slot indexes. Unallocated
; virtual register locations must survive emission for the fast allocator.

; CHECK-DAG: ![[C:[0-9]+]] = !DILocalVariable(name: "constant"
; CHECK-DAG: ![[S:[0-9]+]] = !DILocalVariable(name: "scalar"
; CHECK-DAG: ![[V:[0-9]+]] = !DILocalVariable(name: "vector"
; CHECK-LABEL: name: k
; CHECK: DBG_VALUE float 0.000000e+00, $noreg, ![[C]], !DIExpression()
; CHECK: DBG_VALUE ${{(sgpr[0-9]+|noreg)}}, $noreg, ![[S]], !DIExpression()
; CHECK: DBG_VALUE $vgpr{{[0-9]+}}, $noreg, ![[V]], !DIExpression()

define amdgpu_kernel void @k() !dbg !3 {
entry:
    #dbg_value(float 0.000000e+00, !6, !DIExpression(), !8)
  %s = call i32 asm sideeffect "; scalar $0", "=s"(), !dbg !8
    #dbg_value(i32 %s, !9, !DIExpression(), !8)
  %v = call i32 asm sideeffect "; vector $0", "=v"(), !dbg !8
    #dbg_value(i32 %v, !10, !DIExpression(), !8)
  call void asm sideeffect "; use $0, $1", "s,v"(i32 %s, i32 %v), !dbg !8
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "regalloc-fast.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocalVariable(name: "constant", scope: !3, file: !1, line: 3, type: !7)
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!8 = !DILocation(line: 3, column: 3, scope: !3)
!9 = !DILocalVariable(name: "scalar", scope: !3, file: !1, line: 3, type: !11)
!10 = !DILocalVariable(name: "vector", scope: !3, file: !1, line: 3, type: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

