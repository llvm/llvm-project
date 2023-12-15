; RUN: llc -O0 -mtriple=amdgcn-unknown-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NOOPT %s
; RUN: llc -mtriple=amdgcn-unknown-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,OPT %s

; GCN-LABEL: {{^}}test_debug_value:
; NOOPT: .loc	1 1 42 prologue_end     ; /tmp/test_debug_value.cl:1:42
; NOOPT-NEXT: s_load_dwordx2 s[4:5], s[6:7], 0x0
; NOOPT-NEXT: .Ltmp
; NOOPT-NEXT: ;DEBUG_VALUE: test_debug_value:globalptr_arg <- $sgpr4_sgpr5

; GCN: flat_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @test_debug_value(ptr addrspace(1) nocapture %globalptr_arg) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata ptr addrspace(1) %globalptr_arg, metadata !10, metadata !13), !dbg !14
  store i32 123, ptr addrspace(1) %globalptr_arg, align 4, !dbg !14
  ret void, !dbg !14
}

; Check for infinite loop in some cases with dbg_value in
; SIOptimizeExecMaskingPreRA (somehow related to undef argument).

; GCN-LABEL: {{^}}only_undef_dbg_value:
; NOOPT: ;DEBUG_VALUE: test_debug_value:globalptr_arg <- undef
; NOOPT-NEXT: .cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02
; NOOPT-NEXT: .cfi_undefined 16
; NOOPT-NEXT: s_endpgm

; OPT: s_endpgm
define amdgpu_kernel void @only_undef_dbg_value() #1 {
bb:
  call void @llvm.dbg.value(metadata <4 x float> undef, metadata !10, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)) #2, !dbg !14
  ret void, !dbg !14
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind  }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 244715) (llvm/trunk 244718)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/test_debug_value.cl", directory: "/Users/matt/src/llvm/build_debug")
!2 = !{}
!4 = distinct !DISubprogram(name: "test_debug_value", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "globalptr_arg", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 42, scope: !4)
!15 = !{i32 1, !"amdgpu_code_object_version", i32 500}
