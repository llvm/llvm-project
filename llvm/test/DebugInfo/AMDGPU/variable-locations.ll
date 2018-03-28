; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

; LLVM IR generated with the following command and OpenCL source:
;
; $clang -cl-std=CL2.0 -g -O0 -target amdgcn-amd-amdhsa -S -emit-llvm <path-to-file>
;
; global int GlobA;
; global int GlobB;
;
; kernel void kernel1(unsigned int ArgN, global int  addrspace(5)*ArgA, global int  addrspace(5)*ArgB) {
;   ArgA[ArgN] += ArgB[ArgN];
; }

declare void @llvm.dbg.declare(metadata, metadata, metadata)

; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}}"GlobA"
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_external
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (DW_OP_addr 0x0)
@GlobA = common addrspace(1) global i32 0, align 4, !dbg !0

; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}}"GlobB"
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_external
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (DW_OP_addr 0x0)
@GlobB = common addrspace(1) global i32 0, align 4, !dbg !6

; CHECK: {{.*}}DW_TAG_subprogram
; CHECK: DW_AT_frame_base [DW_FORM_block1]	(DW_OP_reg9 SGPR9)

define amdgpu_kernel void @kernel1(
; CHECK: {{.*}}DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (DW_OP_fbreg +4, DW_OP_constu 0x1, DW_OP_swap, DW_OP_xderef)
; CHECK-NEXT: DW_AT_name {{.*}}"ArgN"
    i32 %ArgN,
; CHECK: {{.*}}DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (DW_OP_fbreg +8, DW_OP_constu 0x1, DW_OP_swap, DW_OP_xderef)
; CHECK-NEXT: DW_AT_name {{.*}}"ArgA"
    i32 addrspace(1)* %ArgA,
; CHECK: {{.*}}DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (DW_OP_fbreg +16, DW_OP_constu 0x1, DW_OP_swap, DW_OP_xderef)
; CHECK-NEXT: DW_AT_name {{.*}}"ArgB"
    i32 addrspace(1)* %ArgB) !dbg !13 {
entry:
  %ArgN.addr = alloca i32, align 4, addrspace(5)
  %ArgA.addr = alloca i32 addrspace(1)*, align 4, addrspace(5)
  %ArgB.addr = alloca i32 addrspace(1)*, align 4, addrspace(5)
  store i32 %ArgN, i32 addrspace(5)* %ArgN.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(5)* %ArgN.addr, metadata !22, metadata !23), !dbg !24
  store i32 addrspace(1)* %ArgA, i32 addrspace(1)* addrspace(5)* %ArgA.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)* addrspace(5)* %ArgA.addr, metadata !25, metadata !23), !dbg !26
  store i32 addrspace(1)* %ArgB, i32 addrspace(1)* addrspace(5)* %ArgB.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)* addrspace(5)* %ArgB.addr, metadata !27, metadata !23), !dbg !28
  %0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %ArgB.addr, align 4, !dbg !29
  %1 = load i32, i32 addrspace(5)* %ArgN.addr, align 4, !dbg !30
  %idxprom = zext i32 %1 to i64, !dbg !29
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %idxprom, !dbg !29
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4, !dbg !29
  %3 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %ArgA.addr, align 4, !dbg !31
  %4 = load i32, i32 addrspace(5)* %ArgN.addr, align 4, !dbg !32
  %idxprom1 = zext i32 %4 to i64, !dbg !31
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %3, i64 %idxprom1, !dbg !31
  %5 = load i32, i32 addrspace(1)* %arrayidx2, align 4, !dbg !33
  %add = add nsw i32 %5, %2, !dbg !33
  store i32 %add, i32 addrspace(1)* %arrayidx2, align 4, !dbg !33
  ret void, !dbg !34
}

!llvm.dbg.cu = !{!2}
!opencl.ocl.version = !{!9}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "GlobA", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "variable-locations.cl", directory: "/some/random/directory")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "GlobB", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, i32 0}
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 5.0.0"}
!13 = distinct !DISubprogram(name: "kernel1", scope: !3, file: !3, line: 4, type: !14, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !17, !17}
!16 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!18 = !{i32 0, i32 1, i32 1}
!19 = !{!"none", !"none", !"none"}
!20 = !{!"uint", !"int addrspace(5)*", !"int addrspace(5)*"}
!21 = !{!"", !"", !""}
!22 = !DILocalVariable(name: "ArgN", arg: 1, scope: !13, file: !3, line: 4, type: !16)
!23 = !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)
!24 = !DILocation(line: 4, column: 34, scope: !13)
!25 = !DILocalVariable(name: "ArgA", arg: 2, scope: !13, file: !3, line: 4, type: !17)
!26 = !DILocation(line: 4, column: 52, scope: !13)
!27 = !DILocalVariable(name: "ArgB", arg: 3, scope: !13, file: !3, line: 4, type: !17)
!28 = !DILocation(line: 4, column: 70, scope: !13)
!29 = !DILocation(line: 5, column: 17, scope: !13)
!30 = !DILocation(line: 5, column: 22, scope: !13)
!31 = !DILocation(line: 5, column: 3, scope: !13)
!32 = !DILocation(line: 5, column: 8, scope: !13)
!33 = !DILocation(line: 5, column: 14, scope: !13)
!34 = !DILocation(line: 6, column: 1, scope: !13)
