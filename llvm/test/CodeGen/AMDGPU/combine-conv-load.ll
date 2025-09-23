; RUN: llc -mtriple=amdgcn -mcpu=gfx942  < %s | FileCheck %s

; CHECK-LABEL:  test:
; CHECK:        .loc    1 8 16                          ; test.py:8:16
; CHECK-NEXT:   s_load_dword

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @test(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg writeonly captures(none) %1, ptr addrspace(1) inreg readnone captures(none) %2, ptr addrspace(1) inreg readnone captures(none) %3) local_unnamed_addr #0 !dbg !4 {
  %5 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %6 = and i32 %5, 255, !dbg !7
  %7 = icmp eq i32 %6, 0, !dbg !7
  br i1 %7, label %8, label %10, !dbg !7

8:                                                ; preds = %4
  %9 = load <1 x float>, ptr addrspace(1) %0, align 4, !dbg !8, !amdgpu.noclobber !6
  store <1 x float> %9, ptr addrspace(1) %1, align 4, !dbg !7
  br label %10, !dbg !7

10:                                               ; preds = %8, %4
  ret void, !dbg !9
}

; Function Attrs: alwaysinline nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="1,1" "denormal-fp-math-f32"="ieee" "uniform-work-group-size"="false" }
attributes #1 = { alwaysinline nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
