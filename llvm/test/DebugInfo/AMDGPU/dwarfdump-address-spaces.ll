; RUN: llc -mtriple=x86_64--gnu -filetype=obj --verify-machineinstrs < %s | llvm-dwarfdump - 2>&1 | FileCheck %s --check-prefixes=COMMON,X86
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -filetype=obj --verify-machineinstrs < %s | llvm-dwarfdump - 2>&1 | FileCheck %s --check-prefixes=COMMON,AMDGPU

; Check that the address spaces are correctly printed for AMDGPU.
; The interpretation of the address space is dependent on the target.

;COMMON: DW_TAG_compile_unit
;COMMON:   DW_TAG_subprogram
;COMMON:     DW_TAG_variable
;COMMON:       DW_AT_name ("A_none")
;COMMON:       DW_AT_type ([[PTR_NONE:0x[0-9a-f]+]]
;COMMON:     DW_TAG_variable
;COMMON:       DW_AT_name ("A_generic")
;COMMON:       DW_AT_type ([[PTR_FLAT:0x[0-9a-f]+]]
;COMMON:     DW_TAG_variable
;COMMON:       DW_AT_name ("A_region")
;COMMON:       DW_AT_type ([[PTR_REGION:0x[0-9a-f]+]]
;COMMON:     DW_TAG_variable
;COMMON:       DW_AT_name ("A_local")
;COMMON:       DW_AT_type ([[PTR_LOCAL:0x[0-9a-f]+]]
;COMMON:     DW_TAG_variable
;COMMON:       DW_AT_name ("A_private_lane")
;COMMON:       DW_AT_type ([[PTR_PRIVATE_LANE:0x[0-9a-f]+]]
;COMMON:     DW_TAG_variable
;COMMON:       DW_AT_name ("A_private_wave")
;COMMON:       DW_AT_type ([[PTR_PRIVATE_WAVE:0x[0-9a-f]+]]

;COMMON: [[PTR_NONE]]: DW_TAG_pointer_type
;COMMON:   DW_AT_type ([[INT:0x[0-9a-f]+]] "int")
;AMDGPU:   DW_AT_LLVM_address_space (0x00000000 "DW_ASPACE_LLVM_none")
;X86:      DW_AT_LLVM_address_space (0x00000000 "DW_ASPACE_LLVM_none")

;COMMON: [[INT]]: DW_TAG_base_type
;COMMON:   DW_AT_name ("int")

;COMMON: [[PTR_FLAT]]: DW_TAG_pointer_type
;COMMON:   DW_AT_type ([[INT]] "int")
;AMDGPU:   DW_AT_LLVM_address_space (0x00000001 "DW_ASPACE_LLVM_AMDGPU_generic")
;X86:      DW_AT_LLVM_address_space (0x00000001)

;COMMON: [[PTR_REGION]]: DW_TAG_pointer_type
;COMMON:   DW_AT_type ([[INT]] "int")
;AMDGPU:   DW_AT_LLVM_address_space (0x00000002 "DW_ASPACE_LLVM_AMDGPU_region")
;X86:      DW_AT_LLVM_address_space (0x00000002)

;COMMON: [[PTR_LOCAL]]: DW_TAG_pointer_type
;COMMON:   DW_AT_type ([[INT]] "int")
;AMDGPU:   DW_AT_LLVM_address_space (0x00000003 "DW_ASPACE_LLVM_AMDGPU_local")
;X86:      DW_AT_LLVM_address_space (0x00000003)

;COMMON: [[PTR_PRIVATE_LANE]]: DW_TAG_pointer_type
;COMMON:   DW_AT_type ([[INT]] "int")
;AMDGPU:   DW_AT_LLVM_address_space (0x00000005 "DW_ASPACE_LLVM_AMDGPU_private_lane")
;X86:      DW_AT_LLVM_address_space (0x00000005)

;COMMON: [[PTR_PRIVATE_WAVE]]: DW_TAG_pointer_type
;COMMON:   DW_AT_type ([[INT]] "int")
;AMDGPU:   DW_AT_LLVM_address_space (0x00000006 "DW_ASPACE_LLVM_AMDGPU_private_wave")
;X86:      DW_AT_LLVM_address_space (0x00000006)

define void @kernel() !dbg !7 {
entry:
  ret void, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dummy.cl", directory: "/some/random/directory")
!2 = !{}
!3 = !{!20, !21, !22, !23, !24, !25}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !DILocation(line: 3, column: 1, scope: !7)
!7 = distinct !DISubprogram(name: "kernel", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !3)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DILocalVariable(name: "A_none", scope: !7, file: !1, line: 1, type: !30)
!21 = !DILocalVariable(name: "A_generic", scope: !7, file: !1, line: 1, type: !31)
!22 = !DILocalVariable(name: "A_region", scope: !7, file: !1, line: 1, type: !32)
!23 = !DILocalVariable(name: "A_local", scope: !7, file: !1, line: 1, type: !33)
!24 = !DILocalVariable(name: "A_private_lane", scope: !7, file: !1, line: 1, type: !34)
!25 = !DILocalVariable(name: "A_private_wave", scope: !7, file: !1, line: 1, type: !35)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, addressSpace: 0)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, addressSpace: 1)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, addressSpace: 2)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, addressSpace: 3)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, addressSpace: 5)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, addressSpace: 6)
