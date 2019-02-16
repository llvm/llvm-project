; RUN: not llc -march=amdgcn -mcpu=tonga < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: foo.cl:1:42: in function rsq_legacy_f32 void (float addrspace(1)*, float): intrinsic not supported on subtarget

declare float @llvm.amdgcn.rsq.legacy(float) #0

define amdgpu_kernel void @rsq_legacy_f32(float addrspace(1)* %out, float %src) #1 {
  %rsq = call float @llvm.amdgcn.rsq.legacy(float %src), !dbg !4
  store float %rsq, float addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "foo.cl", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocation(line: 1, column: 42, scope: !5)
!5 = distinct !DISubprogram(name: "rsq_legacy_f32", scope: null, file: !1, line: 1, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0)
