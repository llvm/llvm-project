; RUN: llc -O0 -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs -filetype=obj < %s | llvm-dwarfdump -v - 2>&1 | FileCheck %s

; LLVM IR generated with the following command and OpenCL source:
;
; $clang -cl-std=CL2.0 -g -O0 -target amdgcn-amd-amdhsa -S -emit-llvm <path-to-file>
;
; kernel void kernel1(global int *A) {
;   *A = 11;
; }
;
; kernel void kernel2(global int *B) {
;   *B = 12;
; }

; CHECK-NOT: failed to compute relocation
; CHECK: file_names[  1] 0 0x00000000 0x00000000 dwarfdump-relocs.cl

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define amdgpu_kernel void @kernel1(i32 addrspace(1)* %A) !dbg !7 {
entry:
  %A.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %A, i32 addrspace(1)** %A.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %A.addr, metadata !16, metadata !17), !dbg !18
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4, !dbg !19
  store i32 11, i32 addrspace(1)* %0, align 4, !dbg !20
  ret void, !dbg !21
}

define amdgpu_kernel void @kernel2(i32 addrspace(1)* %B) !dbg !22 {
entry:
  %B.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %B, i32 addrspace(1)** %B.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %B.addr, metadata !23, metadata !17), !dbg !24
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %B.addr, align 4, !dbg !25
  store i32 12, i32 addrspace(1)* %0, align 4, !dbg !26
  ret void, !dbg !27
}

!llvm.dbg.cu = !{!0}
!opencl.ocl.version = !{!3, !3}
!llvm.module.flags = !{!4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dwarfdump-relocs.cl", directory: "/some/random/directory")
!2 = !{}
!3 = !{i32 2, i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "kernel1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 1}
!13 = !{!"none"}
!14 = !{!"int*"}
!15 = !{!""}
!16 = !DILocalVariable(name: "A", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!17 = !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)
!18 = !DILocation(line: 1, column: 33, scope: !7)
!19 = !DILocation(line: 2, column: 4, scope: !7)
!20 = !DILocation(line: 2, column: 6, scope: !7)
!21 = !DILocation(line: 3, column: 1, scope: !7)
!22 = distinct !DISubprogram(name: "kernel2", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!23 = !DILocalVariable(name: "B", arg: 1, scope: !22, file: !1, line: 5, type: !10)
!24 = !DILocation(line: 5, column: 33, scope: !22)
!25 = !DILocation(line: 6, column: 4, scope: !22)
!26 = !DILocation(line: 6, column: 6, scope: !22)
!27 = !DILocation(line: 7, column: 1, scope: !22)
