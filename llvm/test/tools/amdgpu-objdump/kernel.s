// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -filetype=obj %s | amdgpu-objdump - | FileCheck %s --check-prefix=DIS

.hsa_code_object_version 2,0

.hsa_code_object_isa 8,0,1,"AMD","AMDGPU"

.amdgpu_hsa_kernel test

test:
.amd_kernel_code_t
    amd_code_version_major = 1
    amd_code_version_minor = 0
    amd_machine_kind = 1
    amd_machine_version_major = 8
    amd_machine_version_minor = 0
    amd_machine_version_stepping = 1
    kernel_code_entry_byte_offset = 256
    max_scratch_backing_memory_byte_size = 0
    granulated_workitem_vgpr_count = 3
    granulated_wavefront_sgpr_count = 3
    priority = 0
    enable_dx10_clamp = 1
    enable_sgpr_private_segment_wave_byte_offset = 1
    user_sgpr_count = 10
    enable_sgpr_workgroup_id_x = 1
    enable_sgpr_workgroup_id_y = 1
    enable_sgpr_workgroup_id_z = 1
    enable_vgpr_workitem_id = 2
    granulated_lds_size = 0
    enable_sgpr_private_segment_buffer = 1
    enable_sgpr_dispatch_ptr = 1
    enable_sgpr_kernarg_segment_ptr = 1
    enable_sgpr_flat_scratch_init = 1
    private_element_size = 1
    is_ptr64 = 1
    is_debug_enabled = 1
    workitem_private_segment_byte_size = 88
    kernarg_segment_byte_size = 28
    wavefront_sgpr_count = 25
    workitem_vgpr_count = 15
    reserved_vgpr_first = 11
    reserved_vgpr_count = 4
    kernarg_segment_alignment = 4
    group_segment_alignment = 4
    private_segment_alignment = 4
    wavefront_size = 6
.end_amd_kernel_code_t

    s_mov_b32     flat_scratch_lo, s9
    s_add_u32     s8, s8, s13
    s_lshr_b32    flat_scratch_hi, s8, 8
    v_mov_b32     v3, s10
    s_nop         0x0000
    v_mov_b32     v4, s11
    s_nop         0x0000
    v_mov_b32     v5, s12
    s_nop         0x0000
    s_waitcnt     vmcnt(2) & expcnt(0)
    v_mov_b32     v1, 32
    s_load_dwordx2  s[8:9], s[6:7], 0x00
    s_load_dwordx2  s[14:15], s[6:7], 0x08
    s_load_dwordx2  s[16:17], s[6:7], 0x10
    s_load_dword  s11, s[6:7], 0x18
    s_waitcnt     lgkmcnt(0)
    s_mov_b32     s12, s11
    s_mov_b32     s18, s9
    s_waitcnt     vmcnt(0)
    v_mov_b32     v2, s18
    v_mov_b32     v6, 32
    buffer_store_dword  v2, v6, s[0:3], s13 offen offset:4
    s_mov_b32     s18, s8
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, s18
    buffer_store_dword  v2, v1, s[0:3], s13 offen
    s_mov_b32     s18, s15
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, s18
    buffer_store_dword  v2, v1, s[0:3], s13 offen offset:12
    s_mov_b32     s18, s14
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, s18
    buffer_store_dword  v2, v1, s[0:3], s13 offen offset:8
    s_mov_b32     s18, s17
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, s18
    buffer_store_dword  v2, v1, s[0:3], s13 offen offset:20
    s_mov_b32     s18, s16
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, s18
    buffer_store_dword  v2, v1, s[0:3], s13 offen offset:16
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, s11
    buffer_store_dword  v2, v1, s[0:3], s13 offen offset:24
    s_nop         0x0000
    s_load_dword  s11, s[4:5], 0x04
    s_mov_b32     s18, 0x0000ffff
    s_waitcnt     lgkmcnt(0)
    s_and_b32     s11, s11, s18
    s_mul_i32     s10, s11, s10
    v_add_u32     v0, vcc, s10, v0
    s_waitcnt     vmcnt(0) & expcnt(0)
    v_mov_b32     v2, v0
    buffer_store_dword  v2, v1, s[0:3], s13 offen offset:28
    s_waitcnt     vmcnt(0) & expcnt(0)
    s_nop         0x0000
    buffer_load_dword  v2, v1, s[0:3], s13 offen offset:24
    s_waitcnt     vmcnt(0)
    v_cmp_lt_i32  s[4:5], v0, v2
    v_writelane_b32  v10, s12, 17
    v_writelane_b32  v10, s4, 18
    v_writelane_b32  v10, s5, 19
    s_waitcnt     vmcnt(0) & expcnt(0)
    s_and_saveexec_b64  s[4:5], s[4:5]
    s_xor_b64     s[4:5], exec, s[4:5]
    s_cbranch_execz  label_00EC
    s_nop         0x0000
    s_waitcnt     vmcnt(0)
    buffer_load_dword  v1, v0, s[0:3], s13 offen
    s_nop         0x0000
    buffer_load_dword  v2, v0, s[0:3], s13 offen offset:4
    s_waitcnt     vmcnt(1)
    v_mov_b32     v3, v1
    s_waitcnt     vmcnt(0)
    v_mov_b32     v4, v2
    buffer_load_dword  v1, v0, s[0:3], s13 offen offset:28
    s_waitcnt     vmcnt(0)
    v_mov_b32     v2, v1
    v_ashrrev_i32  v2, 31, v2
    v_mov_b32     v5, v1
    v_mov_b32     v6, v2
    s_mov_b32     s4, 2
    v_mov_b32     v1, v5
    v_mov_b32     v2, v6
    v_mov_b32     v7, v3
    v_mov_b32     v8, v4
    v_add_u32     v7, vcc, v1, v7
    v_mov_b32     v9, v2
    v_addc_u32    v8, vcc, v8, v9, vcc
    v_mov_b32     v3, v7
    v_mov_b32     v4, v8
    s_nop         0x0000
    buffer_load_dword  v8, v0, s[0:3], s13 offen offset:8
    s_nop         0x0000
    buffer_load_dword  v9, v0, s[0:3], s13 offen offset:12
    s_waitcnt     vmcnt(1)
    v_mov_b32     v3, v8
    s_waitcnt     vmcnt(0)
    v_mov_b32     v4, v9
    v_mov_b32     v8, v3
    v_mov_b32     v9, v4
    v_add_u32     v8, vcc, v8, v1
    v_addc_u32    v9, vcc, v9, v2, vcc
    v_mov_b32     v3, v8
    v_mov_b32     v4, v9
    s_waitcnt     vmcnt(0) & lgkmcnt(0)
    v_add_u32     v7, vcc, v8, v7
    buffer_load_dword  v8, v0, s[0:3], s13 offen offset:16
    s_nop         0x0000
    buffer_load_dword  v9, v0, s[0:3], s13 offen offset:20
    s_waitcnt     vmcnt(1)
    v_mov_b32     v3, v8
    s_waitcnt     vmcnt(0)
    v_mov_b32     v4, v9
    v_mov_b32     v8, v3
    v_mov_b32     v9, v4
    v_add_u32     v1, vcc, v8, v1
    v_addc_u32    v2, vcc, v9, v2, vcc
    v_mov_b32     v3, v1
    v_mov_b32     v4, v2
    s_waitcnt     vmcnt(0) & lgkmcnt(0)
label_00EC:
    s_or_b64      exec, exec, s[4:5]
    v_readlane_b32  s4, v10, 18
    v_readlane_b32  s5, v10, 19
    s_nop         0x0000
    s_endpgm




// DIS: .hsa_code_object_version 2,0
// DIS: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"
// DIS: .amdgpu_hsa_kernel test

// DIS: test:
// DIS: .amd_kernel_code_t
// DIS:     amd_code_version_major = 1
// DIS:     amd_code_version_minor = 0
// DIS:     amd_machine_kind = 1
// DIS:     amd_machine_version_major = 8
// DIS:     amd_machine_version_minor = 0
// DIS:     amd_machine_version_stepping = 1
// DIS:     kernel_code_entry_byte_offset = 256
// DIS:     kernel_code_prefetch_byte_size = 0
// DIS:     max_scratch_backing_memory_byte_size = 0
// DIS:     granulated_workitem_vgpr_count = 3
// DIS:     granulated_wavefront_sgpr_count = 3
// DIS:     priority = 0
// DIS:     float_mode = 0
// DIS:     priv = 0
// DIS:     enable_dx10_clamp = 1
// DIS:     debug_mode = 0
// DIS:     enable_ieee_mode = 0
// DIS:     enable_sgpr_private_segment_wave_byte_offset = 1
// DIS:     user_sgpr_count = 10
// DIS:     enable_sgpr_workgroup_id_x = 1
// DIS:     enable_sgpr_workgroup_id_y = 1
// DIS:     enable_sgpr_workgroup_id_z = 1
// DIS:     enable_sgpr_workgroup_info = 0
// DIS:     enable_vgpr_workitem_id = 2
// DIS:     enable_exception_msb = 0
// DIS:     granulated_lds_size = 0
// DIS:     enable_exception = 0
// DIS:     enable_sgpr_private_segment_buffer = 1
// DIS:     enable_sgpr_dispatch_ptr = 1
// DIS:     enable_sgpr_queue_ptr = 0
// DIS:     enable_sgpr_kernarg_segment_ptr = 1
// DIS:     enable_sgpr_dispatch_id = 0
// DIS:     enable_sgpr_flat_scratch_init = 1
// DIS:     enable_sgpr_private_segment_size = 0
// DIS:     enable_sgpr_grid_workgroup_count_x = 0
// DIS:     enable_sgpr_grid_workgroup_count_y = 0
// DIS:     enable_sgpr_grid_workgroup_count_z = 0
// DIS:     enable_ordered_append_gds = 0
// DIS:     private_element_size = 1
// DIS:     is_ptr64 = 1
// DIS:     is_dynamic_callstack = 0
// DIS:     is_debug_enabled = 1
// DIS:     is_xnack_enabled = 0
// DIS:     workitem_private_segment_byte_size = 88
// DIS:     workgroup_group_segment_byte_size = 0
// DIS:     gds_segment_byte_size = 0
// DIS:     kernarg_segment_byte_size = 28
// DIS:     workgroup_fbarrier_count = 0
// DIS:     wavefront_sgpr_count = 25
// DIS:     workitem_vgpr_count = 15
// DIS:     reserved_vgpr_first = 11
// DIS:     reserved_vgpr_count = 4
// DIS:     reserved_sgpr_first = 0
// DIS:     reserved_sgpr_count = 0
// DIS:     debug_wavefront_private_segment_offset_sgpr = 0
// DIS:     debug_private_segment_buffer_sgpr = 0
// DIS:     kernarg_segment_alignment = 4
// DIS:     group_segment_alignment = 4
// DIS:     private_segment_alignment = 4
// DIS:     wavefront_size = 6
// DIS:     call_convention = -1
// DIS:     runtime_loader_kernel_symbol = 0
// DIS: .end_amd_kernel_code_t

// DIS: // Disassembly:
// DIS:     s_mov_b32 flat_scratch_lo, s9                              // 000000000100: BEE60009
// DIS:     s_add_u32 s8, s8, s13                                      // 000000000104: 80080D08
// DIS:     s_lshr_b32 flat_scratch_hi, s8, 8                          // 000000000108: 8F678808
// DIS:     v_mov_b32_e32 v3, s10                                      // 00000000010C: 7E06020A
// DIS:     s_nop 0                                                    // 000000000110: BF800000
// DIS:     v_mov_b32_e32 v4, s11                                      // 000000000114: 7E08020B
// DIS:     s_nop 0                                                    // 000000000118: BF800000
// DIS:     v_mov_b32_e32 v5, s12                                      // 00000000011C: 7E0A020C
// DIS:     s_nop 0                                                    // 000000000120: BF800000
// DIS:     s_waitcnt vmcnt(2) expcnt(0)                               // 000000000124: BF8C0F02
// DIS:     v_mov_b32_e32 v1, 32                                       // 000000000128: 7E0202A0
// DIS:     s_load_dwordx2 s[8:9], s[6:7], 0x0                         // 00000000012C: C0060203 C0060203
// DIS:     s_load_dwordx2 s[14:15], s[6:7], 0x8                       // 000000000134: C0060383 C0060383
// DIS:     s_load_dwordx2 s[16:17], s[6:7], 0x10                      // 00000000013C: C0060403 C0060403
// DIS:     s_load_dword s11, s[6:7], 0x18                             // 000000000144: C00202C3 C00202C3
// DIS:     s_waitcnt lgkmcnt(0)                                       // 00000000014C: BF8C007F
// DIS:     s_mov_b32 s12, s11                                         // 000000000150: BE8C000B
// DIS:     s_mov_b32 s18, s9                                          // 000000000154: BE920009
// DIS:     s_waitcnt vmcnt(0)                                         // 000000000158: BF8C0F70
// DIS:     v_mov_b32_e32 v2, s18                                      // 00000000015C: 7E040212
// DIS:     v_mov_b32_e32 v6, 32                                       // 000000000160: 7E0C02A0
// DIS:     buffer_store_dword v2, v6, s[0:3], s13 offen offset:4      // 000000000164: E0701004 E0701004
// DIS:     s_mov_b32 s18, s8                                          // 00000000016C: BE920008
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 000000000170: BF8C0F00
// DIS:     v_mov_b32_e32 v2, s18                                      // 000000000174: 7E040212
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen               // 000000000178: E0701000 E0701000
// DIS:     s_mov_b32 s18, s15                                         // 000000000180: BE92000F
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 000000000184: BF8C0F00
// DIS:     v_mov_b32_e32 v2, s18                                      // 000000000188: 7E040212
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen offset:12     // 00000000018C: E070100C E070100C
// DIS:     s_mov_b32 s18, s14                                         // 000000000194: BE92000E
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 000000000198: BF8C0F00
// DIS:     v_mov_b32_e32 v2, s18                                      // 00000000019C: 7E040212
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen offset:8      // 0000000001A0: E0701008 E0701008
// DIS:     s_mov_b32 s18, s17                                         // 0000000001A8: BE920011
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 0000000001AC: BF8C0F00
// DIS:     v_mov_b32_e32 v2, s18                                      // 0000000001B0: 7E040212
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen offset:20     // 0000000001B4: E0701014 E0701014
// DIS:     s_mov_b32 s18, s16                                         // 0000000001BC: BE920010
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 0000000001C0: BF8C0F00
// DIS:     v_mov_b32_e32 v2, s18                                      // 0000000001C4: 7E040212
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen offset:16     // 0000000001C8: E0701010 E0701010
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 0000000001D0: BF8C0F00
// DIS:     v_mov_b32_e32 v2, s11                                      // 0000000001D4: 7E04020B
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen offset:24     // 0000000001D8: E0701018 E0701018
// DIS:     s_nop 0                                                    // 0000000001E0: BF800000
// DIS:     s_load_dword s11, s[4:5], 0x4                              // 0000000001E4: C00202C2 C00202C2
// DIS:     s_mov_b32 s18, 0xffff                                      // 0000000001EC: BE9200FF BE9200FF
// DIS:     s_waitcnt lgkmcnt(0)                                       // 0000000001F4: BF8C007F
// DIS:     s_and_b32 s11, s11, s18                                    // 0000000001F8: 860B120B
// DIS:     s_mul_i32 s10, s11, s10                                    // 0000000001FC: 920A0A0B
// DIS:     v_add_i32_e32 v0, vcc, s10, v0                             // 000000000200: 3200000A
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 000000000204: BF8C0F00
// DIS:     v_mov_b32_e32 v2, v0                                       // 000000000208: 7E040300
// DIS:     buffer_store_dword v2, v1, s[0:3], s13 offen offset:28     // 00000000020C: E070101C E070101C
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 000000000214: BF8C0F00
// DIS:     s_nop 0                                                    // 000000000218: BF800000
// DIS:     buffer_load_dword v2, v1, s[0:3], s13 offen offset:24      // 00000000021C: E0501018 E0501018
// DIS:     s_waitcnt vmcnt(0)                                         // 000000000224: BF8C0F70
// DIS:     v_cmp_lt_i32_e64 s[4:5], v0, v2                            // 000000000228: D0C10004 D0C10004
// DIS:     v_writelane_b32 v10, s12, 17                               // 000000000230: D28A000A D28A000A
// DIS:     v_writelane_b32 v10, s4, 18                                // 000000000238: D28A000A D28A000A
// DIS:     v_writelane_b32 v10, s5, 19                                // 000000000240: D28A000A D28A000A
// DIS:     s_waitcnt vmcnt(0) expcnt(0)                               // 000000000248: BF8C0F00
// DIS:     s_and_saveexec_b64 s[4:5], s[4:5]                          // 00000000024C: BE842004
// DIS:     s_xor_b64 s[4:5], exec, s[4:5]                             // 000000000250: 8884047E
// DIS:     s_cbranch_execz label_00EC                                 // 000000000254: BF88003E
// DIS:     s_nop 0                                                    // 000000000258: BF800000
// DIS:     s_waitcnt vmcnt(0)                                         // 00000000025C: BF8C0F70
// DIS:     buffer_load_dword v1, v0, s[0:3], s13 offen                // 000000000260: E0501000 E0501000
// DIS:     s_nop 0                                                    // 000000000268: BF800000
// DIS:     buffer_load_dword v2, v0, s[0:3], s13 offen offset:4       // 00000000026C: E0501004 E0501004
// DIS:     s_waitcnt vmcnt(1)                                         // 000000000274: BF8C0F71
// DIS:     v_mov_b32_e32 v3, v1                                       // 000000000278: 7E060301
// DIS:     s_waitcnt vmcnt(0)                                         // 00000000027C: BF8C0F70
// DIS:     v_mov_b32_e32 v4, v2                                       // 000000000280: 7E080302
// DIS:     buffer_load_dword v1, v0, s[0:3], s13 offen offset:28      // 000000000284: E050101C E050101C
// DIS:     s_waitcnt vmcnt(0)                                         // 00000000028C: BF8C0F70
// DIS:     v_mov_b32_e32 v2, v1                                       // 000000000290: 7E040301
// DIS:     v_ashrrev_i32_e32 v2, 31, v2                               // 000000000294: 2204049F
// DIS:     v_mov_b32_e32 v5, v1                                       // 000000000298: 7E0A0301
// DIS:     v_mov_b32_e32 v6, v2                                       // 00000000029C: 7E0C0302
// DIS:     s_mov_b32 s4, 2                                            // 0000000002A0: BE840082
// DIS:     v_mov_b32_e32 v1, v5                                       // 0000000002A4: 7E020305
// DIS:     v_mov_b32_e32 v2, v6                                       // 0000000002A8: 7E040306
// DIS:     v_mov_b32_e32 v7, v3                                       // 0000000002AC: 7E0E0303
// DIS:     v_mov_b32_e32 v8, v4                                       // 0000000002B0: 7E100304
// DIS:     v_add_i32_e32 v7, vcc, v1, v7                              // 0000000002B4: 320E0F01
// DIS:     v_mov_b32_e32 v9, v2                                       // 0000000002B8: 7E120302
// DIS:     v_addc_u32_e32 v8, vcc, v8, v9, vcc                        // 0000000002BC: 38101308
// DIS:     v_mov_b32_e32 v3, v7                                       // 0000000002C0: 7E060307
// DIS:     v_mov_b32_e32 v4, v8                                       // 0000000002C4: 7E080308
// DIS:     s_nop 0                                                    // 0000000002C8: BF800000
// DIS:     buffer_load_dword v8, v0, s[0:3], s13 offen offset:8       // 0000000002CC: E0501008 E0501008
// DIS:     s_nop 0                                                    // 0000000002D4: BF800000
// DIS:     buffer_load_dword v9, v0, s[0:3], s13 offen offset:12      // 0000000002D8: E050100C E050100C
// DIS:     s_waitcnt vmcnt(1)                                         // 0000000002E0: BF8C0F71
// DIS:     v_mov_b32_e32 v3, v8                                       // 0000000002E4: 7E060308
// DIS:     s_waitcnt vmcnt(0)                                         // 0000000002E8: BF8C0F70
// DIS:     v_mov_b32_e32 v4, v9                                       // 0000000002EC: 7E080309
// DIS:     v_mov_b32_e32 v8, v3                                       // 0000000002F0: 7E100303
// DIS:     v_mov_b32_e32 v9, v4                                       // 0000000002F4: 7E120304
// DIS:     v_add_i32_e32 v8, vcc, v8, v1                              // 0000000002F8: 32100308
// DIS:     v_addc_u32_e32 v9, vcc, v9, v2, vcc                        // 0000000002FC: 38120509
// DIS:     v_mov_b32_e32 v3, v8                                       // 000000000300: 7E060308
// DIS:     v_mov_b32_e32 v4, v9                                       // 000000000304: 7E080309
// DIS:     s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000000308: BF8C0070
// DIS:     v_add_i32_e32 v7, vcc, v8, v7                              // 00000000030C: 320E0F08
// DIS:     buffer_load_dword v8, v0, s[0:3], s13 offen offset:16      // 000000000310: E0501010 E0501010
// DIS:     s_nop 0                                                    // 000000000318: BF800000
// DIS:     buffer_load_dword v9, v0, s[0:3], s13 offen offset:20      // 00000000031C: E0501014 E0501014
// DIS:     s_waitcnt vmcnt(1)                                         // 000000000324: BF8C0F71
// DIS:     v_mov_b32_e32 v3, v8                                       // 000000000328: 7E060308
// DIS:     s_waitcnt vmcnt(0)                                         // 00000000032C: BF8C0F70
// DIS:     v_mov_b32_e32 v4, v9                                       // 000000000330: 7E080309
// DIS:     v_mov_b32_e32 v8, v3                                       // 000000000334: 7E100303
// DIS:     v_mov_b32_e32 v9, v4                                       // 000000000338: 7E120304
// DIS:     v_add_i32_e32 v1, vcc, v8, v1                              // 00000000033C: 32020308
// DIS:     v_addc_u32_e32 v2, vcc, v9, v2, vcc                        // 000000000340: 38040509
// DIS:     v_mov_b32_e32 v3, v1                                       // 000000000344: 7E060301
// DIS:     v_mov_b32_e32 v4, v2                                       // 000000000348: 7E080302
// DIS:     s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000000034C: BF8C0070
// DIS: label_00EC:
// DIS:     s_or_b64 exec, exec, s[4:5]                                // 000000000350: 87FE047E
// DIS:     v_readlane_b32 s4, v10, 18                                 // 000000000354: D2890004 D2890004
// DIS:     v_readlane_b32 s5, v10, 19                                 // 00000000035C: D2890005 D2890005
// DIS:     s_nop 0                                                    // 000000000364: BF800000
// DIS:     s_endpgm                                                   // 000000000368: BF810000
