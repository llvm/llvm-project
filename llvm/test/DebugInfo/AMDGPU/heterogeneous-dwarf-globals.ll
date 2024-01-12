; RUN: llc -mcpu=gfx900 -O0 -filetype=obj -emit-heterogeneous-dwarf-as-user-ops=false < %s -o %t.o
; RUN: llvm-dwarfdump -debug-info -show-form %t.o | FileCheck --check-prefixes=DWARF,DWARF-ORIG-OPS %s
; RUN: llvm-objdump -r %t.o | FileCheck --check-prefixes=RELOCS,RELOCS-OFF-USER-OPS %s

; RUN: llc -mcpu=gfx900 -O0 -filetype=obj < %s -o %t.o
; RUN: llvm-dwarfdump -debug-info -show-form %t.o | FileCheck --check-prefixes=DWARF,DWARF-USER-OPS %s
; RUN: llvm-objdump -r %t.o | FileCheck --check-prefixes=RELOCS,RELOCS-YES-USER-OPS %s

; DWARF: .debug_info contents:
; DWARF: format = DWARF32

; In DWARF32 the size of the attributes preceding the DW_AT_location of each of
; the following variables is 10 bytes:
;   DW_FORM_strp:         4 bytes
;   DW_FORM_ref4:         4 bytes
;   DW_FORM_flag_present: 0 bytes
;   DW_FORM_data1:        1 byte
;   DW_FORM_data1:        1 byte
;   Total encoded size:  10 bytes

; Each relocation is offset an additional 3 bytes, due to the DW_FORM_exprloc
; being encoded as a ULEB128 byte size, which in all of these cases fits in 2
; bytes, followed by the DW_OP_addr operation encoding which is 1 byte.

; This manual accounting of offsets is required as llvm-dwarfdump doesn't seem
; to have an option for printing the offsets of each individual field (where
; they are available), and we cannot depend generally on having lld available
; to resolve these relocations.

; DWARF-ORIG-OPS: 0x0000001e: DW_TAG_variable
; DWARF-USER-OPS: 0x0000001e: DW_TAG_variable
; DWARF-NEXT: DW_AT_name [DW_FORM_strp] ("FileVarDevice")
; DWARF-NEXT: DW_AT_type [DW_FORM_ref4]
; DWARF-NEXT: DW_AT_external [DW_FORM_flag_present]
; DWARF-NEXT: DW_AT_decl_file [DW_FORM_data1]
; DWARF-NEXT: DW_AT_decl_line [DW_FORM_data1]
; DWARF-ORIG-OPS-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addr 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_form_aspace_address)
; DWARF-USER-OPS-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addr 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

; DWARF-ORIG-OPS: 0x00000040: DW_TAG_variable
; DWARF-USER-OPS: 0x00000041: DW_TAG_variable
; DWARF-NEXT: DW_AT_name [DW_FORM_strp] ("FileVarDeviceShared")
; DWARF-NEXT: DW_AT_type [DW_FORM_ref4]
; DWARF-NEXT: DW_AT_external [DW_FORM_flag_present]
; DWARF-NEXT: DW_AT_decl_file [DW_FORM_data1]
; DWARF-NEXT: DW_AT_decl_line [DW_FORM_data1]
; DWARF-ORIG-OPS-NEXT: DW_AT_location [DW_FORM_exprloc] (<empty>)
; DWARF-USER-OPS-NEXT: DW_AT_location [DW_FORM_exprloc] (<empty>)

; DWARF-ORIG-OPS: 0x0000004c: DW_TAG_variable
; DWARF-USER-OPS: 0x0000004d: DW_TAG_variable
; DWARF-NEXT: DW_AT_name [DW_FORM_strp]       ("FileVarDeviceConstant")
; DWARF-NEXT: DW_AT_type [DW_FORM_ref4]
; DWARF-NEXT: DW_AT_external [DW_FORM_flag_present]
; DWARF-NEXT: DW_AT_decl_file [DW_FORM_data1]
; DWARF-NEXT: DW_AT_decl_line [DW_FORM_data1]
; DWARF-ORIG-OPS-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addr 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_form_aspace_address)
; DWARF-USER-OPS-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addr 0x0, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

; RELOCS: RELOCATION RECORDS FOR [.debug_info]:
; RELOCS-OFF: RELOCATION RECORDS FOR
;       0x1e + 0x0d = 0x2b
; RELOCS: 000000000000002b R_AMDGPU_ABS64           FileVarDevice
;       0x4c + 0x0d = 0x59
; RELOCS-OFF-USER-OPS: 0000000000000059 R_AMDGPU_ABS64           FileVarDeviceConstant
;       0x4d + 0x0d = 0x5a
; RELOCS-YES-USER-OPS: 000000000000005a R_AMDGPU_ABS64           FileVarDeviceConstant

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
