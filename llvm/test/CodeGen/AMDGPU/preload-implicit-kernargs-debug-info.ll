; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -passes='amdgpu-attributor,function(amdgpu-lower-kernel-arguments)' -amdgpu-kernarg-preload-count=16 -S < %s 2>&1 | FileCheck %s

; CHECK: define amdgpu_kernel void @preload_block_count_x{{.*}} !dbg ![[#]]
; CHECK-NOT: declare void @0{{.*}} !dbg ![[#]]

define amdgpu_kernel void @preload_block_count_x(ptr addrspace(1) %out) !dbg !4 !max_work_group_size !7 {
  %imp_arg_ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load = load i32, ptr addrspace(4) %imp_arg_ptr, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}

declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !6)
!6 = !{null}
!7 = !{i32 1024, i32 1, i32 1}
