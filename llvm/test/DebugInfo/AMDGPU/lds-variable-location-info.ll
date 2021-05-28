; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

@fun.variable_name = internal addrspace(3) global i32 undef, align 4, !dbg !0


; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}}"variable_name"
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; Function Attrs: convergent noinline nounwind optnone
define protected amdgpu_kernel void @fun(i32 %in) #0 !dbg !2 !kernel_arg_addr_space !16 !kernel_arg_access_qual !17 !kernel_arg_type !18 !kernel_arg_base_type !18 !kernel_arg_type_qual !19 {
entry:
  %in.addr = alloca i32, align 4, addrspace(5)
  store i32 %in, i32 addrspace(5)* %in.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(5)* %in.addr, metadata !20, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !21
  %0 = load i32, i32 addrspace(5)* %in.addr, align 4, !dbg !22
  store i32 %0, i32 addrspace(3)* @fun.variable_name, align 4, !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!10, !11, !12, !13}
!opencl.ocl.version = !{!14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef))
!1 = distinct !DIGlobalVariable(name: "variable_name", scope: !2, file: !3, line: 2, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "fun", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !8)
!3 = !DIFile(filename: "file", directory: "dir")
!4 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !5)
!5 = !{null, !6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !9, nameTableKind: None)
!8 = !{}
!9 = !{!0}
!10 = !{i32 2, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 1}
!14 = !{i32 2, i32 0}
!15 = !{!"clang"}
!16 = !{i32 0}
!17 = !{!"none"}
!18 = !{!"int"}
!19 = !{!""}
!20 = !DILocalVariable(name: "in", arg: 1, scope: !2, file: !3, line: 1, type: !6)
!21 = !DILocation(line: 1, column: 21, scope: !2)
!22 = !DILocation(line: 3, column: 19, scope: !2)
!23 = !DILocation(line: 3, column: 17, scope: !2)
!24 = !DILocation(line: 4, column: 1, scope: !2)
