// RUN: llvm-mc -triple amdgcn--amdpal -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdpal -mcpu=kaveri -show-encoding %s | llvm-readobj --symbols -S --sd | FileCheck %s --check-prefix=ELF

	.amdgpu_pal_metadata
---
amdpal.pipelines: 
  - .hardware_stages: 
      .ps:             
        .entry_point:    ps_amdpal
        .scratch_memory_size: 0
        .sgpr_count:     0x1
        .vgpr_count:     0x1
    .internal_pipeline_hash: 
      - 0x123456789abcdef0
      - 0xfedcba9876543210
    .registers:      
      0x2c0a (SPI_SHADER_PGM_RSRC1_PS): 0
      0x2c0b (SPI_SHADER_PGM_RSRC2_PS): 0x42000000
      0xa1b3 (SPI_PS_INPUT_ENA): 0x1
      0xa1b4 (SPI_PS_INPUT_ADDR): 0x1
...
	.end_amdgpu_pal_metadata

// ASM: 	.amdgpu_pal_metadata
// ASM: ---
// ASM: amdpal.pipelines: 
// ASM:   - .hardware_stages: 
// ASM:       .ps:             
// ASM:         .entry_point:    ps_amdpal
// ASM:         .scratch_memory_size: 0
// ASM:         .sgpr_count:     0x1
// ASM:         .vgpr_count:     0x1
// ASM:     .internal_pipeline_hash: 
// ASM:       - 0x123456789abcdef0
// ASM:       - 0xfedcba9876543210
// ASM:     .registers:      
// ASM:       0x2c0a (SPI_SHADER_PGM_RSRC1_PS): 0
// ASM:       0x2c0b (SPI_SHADER_PGM_RSRC2_PS): 0x42000000
// ASM:       0xa1b3 (SPI_PS_INPUT_ENA): 0x1
// ASM:       0xa1b4 (SPI_PS_INPUT_ADDR): 0x1
// ASM: ...
// ASM: 	.end_amdgpu_pal_metadata

// ELF: SHT_NOTE
// ELF:       0000: 07000000 BD000000 20000000 414D4447  |........ ...AMDG|
// ELF:       0010: 50550000 81B0616D 6470616C 2E706970  |PU....amdpal.pip|
// ELF:       0020: 656C696E 65739183 B02E6861 72647761  |elines....hardwa|
// ELF:       0030: 72655F73 74616765 7381A32E 707384AC  |re_stages...ps..|
// ELF:       0040: 2E656E74 72795F70 6F696E74 A970735F  |.entry_point.ps_|
// ELF:       0050: 616D6470 616CB42E 73637261 7463685F  |amdpal..scratch_|
// ELF:       0060: 6D656D6F 72795F73 697A6500 AB2E7367  |memory_size...sg|
// ELF:       0070: 70725F63 6F756E74 01AB2E76 6770725F  |pr_count...vgpr_|
// ELF:       0080: 636F756E 7401B72E 696E7465 726E616C  |count...internal|
// ELF:       0090: 5F706970 656C696E 655F6861 736892CF  |_pipeline_hash..|
// ELF:       00A0: 12345678 9ABCDEF0 CFFEDCBA 98765432  |.4Vx.........vT2|
// ELF:       00B0: 10AA2E72 65676973 74657273 84CD2C0A  |...registers..,.|
// ELF:       00C0: 00CD2C0B CE420000 00CDA1B3 01CDA1B4  |..,..B..........|
// ELF:       00D0: 01000000                             |....|


