; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -filetype=obj < %s | llvm-dwarfdump --name=m --name=acc - | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -enable-new-pm -filetype=obj < %s | llvm-dwarfdump --name=m --name=acc - | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -experimental-debug-variable-locations=true -filetype=obj < %s | llvm-dwarfdump --name=m --name=acc - | FileCheck %s

; SGPRs are allocated, and rewritten by a VirtRegRewriter, before VGPRs.
; Check that variables whose values live in SGPRs still get SGPR locations
; when debug values are emitted after VGPR allocation.

; CHECK: DW_AT_location
; CHECK-NEXT: DW_OP_regx SGPR{{[0-9]+}}
; CHECK: DW_AT_name ("m")

; CHECK: DW_AT_location
; CHECK-NEXT: DW_OP_regx SGPR{{[0-9]+}}
; CHECK-NEXT: DW_AT_name ("acc")

define amdgpu_kernel void @straight_line(ptr addrspace(1) %out, i32 %n) !dbg !5 {
entry:
  %m = mul i32 %n, 13, !dbg !8
    #dbg_value(i32 %m, !9, !DIExpression(), !8)
  %tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %tid, !dbg !8
  store i32 %m, ptr addrspace(1) %p, !dbg !8
  ret void, !dbg !8
}

define amdgpu_kernel void @uniform_loop(ptr addrspace(1) %out, i32 %n) !dbg !12 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 1, %entry ], [ %acc.next, %loop ]
    #dbg_value(i32 %acc, !13, !DIExpression(), !14)
  %acc.next = mul i32 %acc, %n, !dbg !14
  %i.next = add i32 %i, 1, !dbg !14
  %c = icmp ult i32 %i.next, %n, !dbg !14
  br i1 %c, label %loop, label %exit, !dbg !14

exit:
  store i32 %acc.next, ptr addrspace(1) %out, !dbg !14
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "straight_line", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 2, column: 1, scope: !5)
!9 = !DILocalVariable(name: "m", scope: !5, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = distinct !DISubprogram(name: "uniform_loop", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DILocalVariable(name: "acc", scope: !12, file: !1, line: 11, type: !11)
!14 = !DILocation(line: 11, column: 1, scope: !12)
