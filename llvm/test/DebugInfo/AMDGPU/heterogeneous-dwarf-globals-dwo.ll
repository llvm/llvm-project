; RUN: llc -mcpu=gfx900 -O0 -dwarf-version=4 -filetype=obj -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo -emit-heterogeneous-dwarf-as-user-ops < %s -o %t.o
; RUN: llvm-dwarfdump -debug-info -show-form %t.o | FileCheck --check-prefixes=COMMON,V4,DWARF,DWARF-V4 %s
; RUN: llvm-dwarfdump -debug-info -show-form %t.dwo | FileCheck --check-prefixes=COMMON,V4,DWARF-DWO,DWARF-DWO-V4 %s
; RUN: llvm-objdump -r %t.o | FileCheck --check-prefixes=RELOCS,RELOCS-V4 %s

; RUN: llc -mcpu=gfx900 -O0 -dwarf-version=5 -filetype=obj -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo -emit-heterogeneous-dwarf-as-user-ops < %s -o %t.o
; RUN: llvm-dwarfdump -debug-info -show-form %t.o | FileCheck --check-prefixes=COMMON,V5,DWARF,DWARF-V5 %s
; RUN: llvm-dwarfdump -debug-info -show-form %t.dwo | FileCheck --check-prefixes=COMMON,V5,DWARF-DWO,DWARF-DWO-V5 %s
; RUN: llvm-objdump -r %t.o | FileCheck --check-prefixes=RELOCS,RELOCS-V5 %s

; DWARF: .debug_info contents:
; DWARF-DWO: .debug_info.dwo contents:
; COMMON: format = DWARF32

; In the .o, there is only an entry pointing to the dwo file. The later contains the debug info.
; DWARF-V4: DW_TAG_compile_unit
; DWARF-V5: DW_TAG_skeleton_unit
; DWARF-DWO: DW_TAG_compile_unit
; V4: DW_AT_GNU_dwo_name [{{DW_FORM_strp|DW_FORM_GNU_str_index}}] ("foo.dwo")
; V5: DW_AT_dwo_name [{{DW_FORM_strp|DW_FORM_strx1}}] ("foo.dwo")
; DWARF-NOT: DW_TAG


; For variables that are not in LDS, dwo uses DW_OP_GNU_addr_index (before DWARF 5) or DW_OP_addrx (after DWARF 5). The operand for these instructions is an index in the .debug_addr section of the .o .
; This is used to avoid relocations in the .dwo file (since it doesn't go through the linker).

; DWARF-DWO: DW_TAG_variable
; DWARF-DWO-NEXT: DW_AT_name [{{DW_FORM_GNU_str_index|DW_FORM_strx1}}] ("FileVarDevice")
; DWARF-DWO-NEXT: DW_AT_type [DW_FORM_ref4]
; DWARF-DWO-NEXT: DW_AT_external [DW_FORM_flag_present]
; DWARF-DWO-NEXT: DW_AT_decl_file [DW_FORM_data1]
; DWARF-DWO-NEXT: DW_AT_decl_line [DW_FORM_data1]
; DWARF-DWO-V4-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_GNU_addr_index 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; DWARF-DWO-V5-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

; DWARF-DWO: DW_TAG_variable
; DWARF-DWO-NEXT: DW_AT_name [{{DW_FORM_GNU_str_index|DW_FORM_strx1}}] ("FileVarDeviceShared")
; DWARF-DWO-NEXT: DW_AT_type [DW_FORM_ref4]
; DWARF-DWO-NEXT: DW_AT_external [DW_FORM_flag_present]
; DWARF-DWO-NEXT: DW_AT_decl_file [DW_FORM_data1]
; DWARF-DWO-NEXT: DW_AT_decl_line [DW_FORM_data1]
; DWARF-DWO-NEXT: DW_AT_location [DW_FORM_exprloc] (<empty>)

; DWARF-DWO: DW_TAG_variable
; DWARF-DWO-NEXT: DW_AT_name [{{DW_FORM_GNU_str_index|DW_FORM_strx1}}]       ("FileVarDeviceConstant")
; DWARF-DWO-NEXT: DW_AT_type [DW_FORM_ref4]
; DWARF-DWO-NEXT: DW_AT_external [DW_FORM_flag_present]
; DWARF-DWO-NEXT: DW_AT_decl_file [DW_FORM_data1]
; DWARF-DWO-NEXT: DW_AT_decl_line [DW_FORM_data1]
; DWARF-DWO-V4-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_GNU_addr_index 0x1, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; DWARF-DWO-V5-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addrx 0x1, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

; RELOCS: RELOCATION RECORDS FOR [.debug_addr]:
; RELOCS-V4: 0000000000000000 R_AMDGPU_ABS64           FileVarDevice
; RELOCS-V4: 0000000000000008 R_AMDGPU_ABS64           FileVarDeviceConstant
; RELOCS-V5: 0000000000000008 R_AMDGPU_ABS64           FileVarDevice
; RELOCS-V5: 0000000000000010 R_AMDGPU_ABS64           FileVarDeviceConstant

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

@FileVarDevice = addrspace(1) externally_initialized global i32 0, align 4, !dbg.def !0
@FileVarDeviceShared = addrspace(3) externally_initialized global i32 undef, align 4, !dbg.def !1
@FileVarDeviceConstant = addrspace(4) externally_initialized global i32 0, align 4, !dbg.def !2

!llvm.dbg.cu = !{!3}
!llvm.dbg.retainedNodes = !{!5, !9, !11}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DIFragment()
!1 = distinct !DIFragment()
!2 = distinct !DIFragment()
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git a035b1ccc2e4fdeed0f7122a8398fda4b3c0633b)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "clang/test/CodeGenHIP/<stdin>", directory: "/home/slinder1/llvm-project/amd-stg-open")
!5 = distinct !DILifetime(object: !6, location: !DIExpr(DIOpArg(0, i32 addrspace(1)*), DIOpDeref(i32)), argObjects: {!0})
!6 = distinct !DIGlobalVariable(name: "FileVarDevice", scope: !3, file: !7, line: 8, type: !8, isLocal: false, isDefinition: true)
!7 = !DIFile(filename: "clang/test/CodeGenHIP/debug-info-address-class-heterogeneous-dwarf.hip", directory: "/home/slinder1/llvm-project/amd-stg-open")
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DILifetime(object: !10, location: !DIExpr(DIOpArg(0, i32 addrspace(3)*), DIOpDeref(i32)), argObjects: {!1})
!10 = distinct !DIGlobalVariable(name: "FileVarDeviceShared", scope: !3, file: !7, line: 9, type: !8, isLocal: false, isDefinition: true)
!11 = distinct !DILifetime(object: !12, location: !DIExpr(DIOpArg(0, i32 addrspace(4)*), DIOpDeref(i32)), argObjects: {!2})
!12 = distinct !DIGlobalVariable(name: "FileVarDeviceConstant", scope: !3, file: !7, line: 10, type: !8, isLocal: false, isDefinition: true)
!13 = !{i32 2, !"Debug Info Version", i32 4}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git a035b1ccc2e4fdeed0f7122a8398fda4b3c0633b)"}
