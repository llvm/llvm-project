; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm -o - %s | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -filetype=asm -o - %s | FileCheck --check-prefixes=CHECK,WAVE32 %s

; CHECK-LABEL: kern1:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; DW_CFA_def_cfa_expression [0x0f]
;   BLOCK_LENGTH ULEB128(3)=[0x03]
;     DW_OP_lit0 [0x30]
;     DW_OP_lit6 [0x36]
;     DW_OP_LLVM_form_aspace_address [0xe1]
; CHECK-NEXT: .cfi_escape 0x0f, 0x03, 0x30, 0x36, 0xe1
; PC_64 = 16
; CHECK-NEXT: .cfi_undefined 16

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define protected amdgpu_kernel void @kern1() #0 {
entry:
  ret void
}

; CHECK-LABEL: func1:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; SGPR32 = 64
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; DW_CFA_expression [0x10]
;   PC_64 ULEB128(17)=[0x10]
;   BLOCK_LENGTH ULEB128(8)=[0x08]
;     DW_OP_regx [0x90]
;       SGPR30 ULEB128(62)=[0x3e]
;     DW_OP_piece [0x93]
;       PIECE_SIZE [0x04]
;     DW_OP_regx [0x90]
;       SGPR31 ULEB128(63)=[0x3f]
;     DW_OP_piece [0x93]
;       PIECE_SIZE [0x04]
; CHECK-NEXT: .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define hidden void @func1() #0 {
entry:
  ret void
}

declare hidden void @ex() #0

; CHECK-LABEL: func2:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04
; VGPR0_wave64 = 2560
; WAVE64-NEXT: .cfi_undefined 2560
; WAVE64-NEXT: .cfi_undefined 2561
; WAVE64-NEXT: .cfi_undefined 2562
; WAVE64-NEXT: .cfi_undefined 2563
; WAVE64-NEXT: .cfi_undefined 2564
; WAVE64-NEXT: .cfi_undefined 2565
; WAVE64-NEXT: .cfi_undefined 2566
; WAVE64-NEXT: .cfi_undefined 2567
; WAVE64-NEXT: .cfi_undefined 2568
; WAVE64-NEXT: .cfi_undefined 2569
; WAVE64-NEXT: .cfi_undefined 2570
; WAVE64-NEXT: .cfi_undefined 2571
; WAVE64-NEXT: .cfi_undefined 2572
; WAVE64-NEXT: .cfi_undefined 2573
; WAVE64-NEXT: .cfi_undefined 2574
; WAVE64-NEXT: .cfi_undefined 2575
; WAVE64-NEXT: .cfi_undefined 2576
; WAVE64-NEXT: .cfi_undefined 2577
; WAVE64-NEXT: .cfi_undefined 2578
; WAVE64-NEXT: .cfi_undefined 2579
; WAVE64-NEXT: .cfi_undefined 2580
; WAVE64-NEXT: .cfi_undefined 2581
; WAVE64-NEXT: .cfi_undefined 2582
; WAVE64-NEXT: .cfi_undefined 2583
; WAVE64-NEXT: .cfi_undefined 2584
; WAVE64-NEXT: .cfi_undefined 2585
; WAVE64-NEXT: .cfi_undefined 2586
; WAVE64-NEXT: .cfi_undefined 2587
; WAVE64-NEXT: .cfi_undefined 2588
; WAVE64-NEXT: .cfi_undefined 2589
; WAVE64-NEXT: .cfi_undefined 2590
; WAVE64-NEXT: .cfi_undefined 2591
; VGPR0_wave32 = 1536
; WAVE32-NEXT: .cfi_undefined 1536
; WAVE32-NEXT: .cfi_undefined 1537
; WAVE32-NEXT: .cfi_undefined 1538
; WAVE32-NEXT: .cfi_undefined 1539
; WAVE32-NEXT: .cfi_undefined 1540
; WAVE32-NEXT: .cfi_undefined 1541
; WAVE32-NEXT: .cfi_undefined 1542
; WAVE32-NEXT: .cfi_undefined 1543
; WAVE32-NEXT: .cfi_undefined 1544
; WAVE32-NEXT: .cfi_undefined 1545
; WAVE32-NEXT: .cfi_undefined 1546
; WAVE32-NEXT: .cfi_undefined 1547
; WAVE32-NEXT: .cfi_undefined 1548
; WAVE32-NEXT: .cfi_undefined 1549
; WAVE32-NEXT: .cfi_undefined 1550
; WAVE32-NEXT: .cfi_undefined 1551
; WAVE32-NEXT: .cfi_undefined 1552
; WAVE32-NEXT: .cfi_undefined 1553
; WAVE32-NEXT: .cfi_undefined 1554
; WAVE32-NEXT: .cfi_undefined 1555
; WAVE32-NEXT: .cfi_undefined 1556
; WAVE32-NEXT: .cfi_undefined 1557
; WAVE32-NEXT: .cfi_undefined 1558
; WAVE32-NEXT: .cfi_undefined 1559
; WAVE32-NEXT: .cfi_undefined 1560
; WAVE32-NEXT: .cfi_undefined 1561
; WAVE32-NEXT: .cfi_undefined 1562
; WAVE32-NEXT: .cfi_undefined 1563
; WAVE32-NEXT: .cfi_undefined 1564
; WAVE32-NEXT: .cfi_undefined 1565
; WAVE32-NEXT: .cfi_undefined 1566
; WAVE32-NEXT: .cfi_undefined 1567
; SGPR0 = 32
; CHECK-NEXT: .cfi_undefined 32
; CHECK-NEXT: .cfi_undefined 33
; CHECK-NEXT: .cfi_undefined 34
; CHECK-NEXT: .cfi_undefined 35
; CHECK-NEXT: .cfi_undefined 36
; CHECK-NEXT: .cfi_undefined 37
; CHECK-NEXT: .cfi_undefined 38
; CHECK-NEXT: .cfi_undefined 39
; CHECK-NEXT: .cfi_undefined 40
; CHECK-NEXT: .cfi_undefined 41
; CHECK-NEXT: .cfi_undefined 42
; CHECK-NEXT: .cfi_undefined 43
; CHECK-NEXT: .cfi_undefined 44
; CHECK-NEXT: .cfi_undefined 45
; CHECK-NEXT: .cfi_undefined 46
; CHECK-NEXT: .cfi_undefined 47
; CHECK-NEXT: .cfi_undefined 48
; CHECK-NEXT: .cfi_undefined 49
; CHECK-NEXT: .cfi_undefined 50
; CHECK-NEXT: .cfi_undefined 51
; CHECK-NEXT: .cfi_undefined 52
; CHECK-NEXT: .cfi_undefined 53
; CHECK-NEXT: .cfi_undefined 54
; CHECK-NEXT: .cfi_undefined 55
; CHECK-NEXT: .cfi_undefined 56
; CHECK-NEXT: .cfi_undefined 57
; CHECK-NEXT: .cfi_undefined 58
; CHECK-NEXT: .cfi_undefined 59
; CHECK-NEXT: .cfi_undefined 60
; CHECK-NEXT: .cfi_undefined 61
; CHECK-NEXT: .cfi_undefined 62
; CHECK-NEXT: .cfi_undefined 63

; CHECK-NOT: .cfi_{{.*}}

; WAVE64: s_or_saveexec_b64 s[4:5], -1
; WAVE32: s_or_saveexec_b32 s4, -1
; CHECK-NEXT: buffer_store_dword v32, off, s[0:3], s32 ; 4-byte Folded Spill
; VGPR32_wave64 = 2592
; WAVE64-NEXT: .cfi_offset 2592, 0
; VGPR32_wave32 = 1568
; WAVE32-NEXT: .cfi_offset 1568, 0
; CHECK-NOT: .cfi_{{.*}}
; WAVE64: s_mov_b64 exec, s[4:5]
; WAVE32: s_mov_b32 exec_lo, s4

; CHECK-NOT: .cfi_{{.*}}

; CHECK: v_writelane_b32 v32, s33, 2

; DW_CFA_expression [0x10] SGPR33 ULEB128(65)=[0x41]
;   BLOCK_LENGTH ULEB128(5)=[0x05]
;     DW_OP_regx [0x90]
;       VGPR32_wave64 ULEB128(2592)=[0xa0, 0x14]
;     DW_OP_LLVM_offset_uconst [0xe4]
;       OFFSET ULEB128(0x08) [0x08]
; WAVE64-NEXT: .cfi_escape 0x10, 0x41, 0x05, 0x90, 0xa0, 0x14, 0xe4, 0x08

; DW_CFA_expression [0x10] SGPR33 ULEB128(65)=[0x41]
;   BLOCK_LENGTH ULEB128(5)=[0x05]
;     DW_OP_regx [0x90]
;       VGPR32_wave32 ULEB128(1568)=[0xa0, 0x0c]
;     DW_OP_LLVM_offset_uconst [0xe4]
;       OFFSET ULEB128(0x08) [0x08]
; WAVE32-NEXT: .cfi_escape 0x10, 0x41, 0x05, 0x90, 0xa0, 0x0c, 0xe4, 0x08

; CHECK-NOT: .cfi_{{.*}}

; CHECK: s_mov_b32 s33, s32
; SGPR33 = 65
; CHECK-NEXT: .cfi_def_cfa_register 65

; CHECK-NOT: .cfi_{{.*}}

; CHECK: s_sub_u32 s32, s32,
; CHECK-NEXT: v_readlane_b32 s33, v32, 2
; SGPR32 = 64
; CHECK-NEXT: .cfi_def_cfa_register 64

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define hidden void @func2() #0 {
entry:
  call void @ex() #0
  ret void
}

; CHECK-LABEL: func3:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; SGPR32 = 64
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04

; CHECK-NOT: .cfi_{{.*}}

; CHECK: buffer_store_dword v33, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill

; DW_CFA_expression [0x10]
;   VGPR33_wave64 ULEB128(1569)=[0xa1, 0x14]
;   BLOCK_LENGTH ULEB128(14)=[0x0e]
;     DW_OP_regx [0x90]
;       VGPR33_wave64 ULEB128(1569)=[0xa1, 0x14]
;     DW_OP_swap [0x16]
;     DW_OP_LLVM_offset_uconst [0xe4]
;       OFFSET ULEB128(256)=[0x80, 0x02]
;     DW_OP_LLVM_call_frame_entry_reg [0xe6]
;       EXEC_MASK_wave64 ULEB128(17)=[0x11]
;     DW_OP_deref_size [0x94]
;       SIZE [0x08]
;     DW_OP_LLVM_select_bit_piece [0xec]
;       ELEMENT_SIZE [0x20]
;       ELEMENT_COUNT [0x40]
; WAVE64-NEXT: .cfi_escape 0x10, 0xa1, 0x14, 0x0e, 0x90, 0xa1, 0x14, 0x16, 0xe4, 0x80, 0x02, 0xe6, 0x11, 0x94, 0x08, 0xec, 0x20, 0x40

; DW_CFA_expression [0x10]
;   VGPR33_wave32 ULEB128(1569)=[0xa1, 0x0c]
;   BLOCK_LENGTH ULEB128(14)=[0x0e]
;     DW_OP_regx [0x90]
;       VGPR33_wave32 ULEB128(1569)=[0xa1, 0x0c]
;     DW_OP_swap [0x16]
;     DW_OP_LLVM_offset_uconst [0xe4]
;       OFFSET ULEB128(128)=[0x80, 0x01]
;     DW_OP_LLVM_call_frame_entry_reg [0xe6]
;       EXEC_MASK_wave32 ULEB128(1)=[0x01]
;     DW_OP_deref_size [0x94]
;       SIZE [0x04]
;     DW_OP_LLVM_select_bit_piece [0xec]
;       ELEMENT_SIZE [0x20]
;       ELEMENT_COUNT [0x20]
; WAVE32-NEXT: .cfi_escape 0x10, 0xa1, 0x0c, 0x0e, 0x90, 0xa1, 0x0c, 0x16, 0xe4, 0x80, 0x01, 0xe6, 0x01, 0x94, 0x04, 0xec, 0x20, 0x20

; CHECK-NOT: .cfi_{{.*}}

; CHECK: buffer_store_dword v34, off, s[0:3], s32 ; 4-byte Folded Spill

; DW_CFA_expression [0x10]
;   VGPR34_wave64 ULEB128(2594)=[0xa2, 0x14]
;   BLOCK_LENGTH ULEB128(13)=[0x0d]
;     DW_OP_regx [0x90]
;       VGPR34_wave64 ULEB128(2593)=[0xa2, 0x14]
;     DW_OP_swap [0x16]
;     DW_OP_LLVM_offset_uconst [0xe4]
;       OFFSET ULEB128(0)=[0x00]
;     DW_OP_LLVM_call_frame_entry_reg [0xe6]
;       EXEC_MASK_wave64 ULEB128(17)=[0x11]
;     DW_OP_deref_size [0x94]
;       SIZE [0x08]
;     DW_OP_LLVM_select_bit_piece [0xec]
;       ELEMENT_SIZE [0x20]
;       ELEMENT_COUNT [0x40]
; WAVE64-NEXT: .cfi_escape 0x10, 0xa2, 0x14, 0x0d, 0x90, 0xa2, 0x14, 0x16, 0xe4, 0x00, 0xe6, 0x11, 0x94, 0x08, 0xec, 0x20, 0x40

; DW_CFA_expression [0x10]
;   VGPR34_wave32 ULEB128(1570)=[0xa2, 0x0c]
;   BLOCK_LENGTH ULEB128(13)=[0x0d]
;     DW_OP_regx [0x90]
;       VGPR34_wave32 ULEB128(1569)=[0xa2, 0x0c]
;     DW_OP_swap [0x16]
;     DW_OP_LLVM_offset_uconst [0xe4]
;       OFFSET ULEB128(0)=[0x00]
;     DW_OP_LLVM_call_frame_entry_reg [0xe6]
;       EXEC_MASK_wave32 ULEB128(1)=[0x01]
;     DW_OP_deref_size [0x94]
;       SIZE [0x04]
;     DW_OP_LLVM_select_bit_piece [0xec]
;       ELEMENT_SIZE [0x20]
;       ELEMENT_COUNT [0x20]
; WAVE32-NEXT: .cfi_escape 0x10, 0xa2, 0x0c, 0x0d, 0x90, 0xa2, 0x0c, 0x16, 0xe4, 0x00, 0xe6, 0x01, 0x94, 0x04, 0xec, 0x20, 0x20

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define hidden void @func3() #0 {
entry:
  call void asm sideeffect "; clobber", "~{v33}"() #0
  call void asm sideeffect "; clobber", "~{v34}"() #0
  ret void
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "filename", directory: "directory")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
