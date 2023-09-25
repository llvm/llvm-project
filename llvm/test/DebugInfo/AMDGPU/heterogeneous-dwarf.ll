; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -verify-machineinstrs -filetype=obj -emit-heterogeneous-dwarf-as-user-ops < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefixes=COMMON,FLAT-SCR-DIS %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -verify-machineinstrs -filetype=obj -emit-heterogeneous-dwarf-as-user-ops < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefixes=COMMON,FLAT-SCR-ENA %s

source_filename = "heterogeneous-dwarf.cl"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; CHECK: {{.*}}DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]      (DW_OP_regx SGPR33_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit8, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK-NEXT: DW_AT_name {{.*}}"A"

; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]      (DW_OP_regx SGPR33_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK-NEXT: DW_AT_name {{.*}}"B"

; COMMON: {{.*}}DW_TAG_variable
; FLAT-SCR-DIS: DW_AT_location [DW_FORM_exprloc]      (DW_OP_regx SGPR33_LO16, DW_OP_lit6, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_deref_size 0x4, DW_OP_swap, DW_OP_shr, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit20, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; FLAT-SCR-ENA: DW_AT_location [DW_FORM_exprloc]      (DW_OP_regx SGPR33_LO16, DW_OP_deref_size 0x4, DW_OP_constu 0x5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_lit16, DW_OP_stack_value, DW_OP_deref_size 0x4, DW_OP_LLVM_user DW_OP_LLVM_offset)
; COMMON: DW_AT_name {{.*}}"C"

define protected amdgpu_kernel void @testKernel(i32 addrspace(1)* %A) #0 !dbg !11 !kernel_arg_addr_space !17 !kernel_arg_access_qual !18 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !20 {
entry:
  %A.addr = alloca i32 addrspace(1)*, align 8, addrspace(5)
  %B = alloca i32, align 4, addrspace(5)
  %C = alloca i32, align 4, addrspace(5)
  store i32 addrspace(1)* %A, i32 addrspace(1)* addrspace(5)* %A.addr, align 8
  call void @llvm.dbg.def(metadata !21, metadata i32 addrspace(1)* addrspace(5)* %A.addr), !dbg !23
  call void @llvm.dbg.def(metadata !24, metadata i32 addrspace(5)* %B), !dbg !26
  call void @llvm.dbg.def(metadata !31, metadata ptr addrspace(5) %C), !dbg !34
  store i32 777, i32 addrspace(5)* %B, align 4, !dbg !26
  %0 = load i32, i32 addrspace(5)* %B, align 4, !dbg !27
  %1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %A.addr, align 8, !dbg !28
  store i32 %0, i32 addrspace(1)* %1, align 4, !dbg !29
  ret void, !dbg !30
}

declare void @llvm.dbg.def(metadata, metadata) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,256" "amdgpu-implicitarg-num-bytes"="56" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!opencl.ocl.version = !{!7, !8}
!llvm.ident = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "heterogeneous-dwarf.cl", directory: "/some/random/directory", checksumkind: CSK_MD5, checksum: "1bf703ec4e028cb8cf8a51769ce79495")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 4}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{i32 1, i32 2}
!8 = !{i32 2, i32 0}
!9 = !{!"clang version 14.0.0"}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "testKernel", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!12 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{}
!17 = !{i32 1}
!18 = !{!"none"}
!19 = !{!"int*"}
!20 = !{!""}
!21 = distinct !DILifetime(object: !22, location: !DIExpr(DIOpReferrer(i32 addrspace(1)* addrspace(5)*), DIOpDeref(i32 addrspace(1)*)))
!22 = !DILocalVariable(name: "A", arg: 1, scope: !11, file: !1, line: 1, type: !14)
!23 = !DILocation(line: 1, column: 36, scope: !11)
!24 = distinct !DILifetime(object: !25, location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)))
!25 = !DILocalVariable(name: "B", scope: !11, file: !1, line: 2, type: !15)
!26 = !DILocation(line: 2, column: 7, scope: !11)
!27 = !DILocation(line: 3, column: 8, scope: !11)
!28 = !DILocation(line: 3, column: 4, scope: !11)
!29 = !DILocation(line: 3, column: 6, scope: !11)
!30 = !DILocation(line: 4, column: 1, scope: !11)
!31 = distinct !DILifetime(object: !32, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!32 = !DILocalVariable(name: "C", scope: !11, file: !1, line: 1, type: !33)
!33 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!34 = !DILocation(line: 5, column: 1, scope: !11)
