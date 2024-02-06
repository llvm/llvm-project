
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global | FileCheck --check-prefix=HSA-CI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -filetype=obj | llvm-readobj -S --sd --syms - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=4 | llvm-readobj -S --sd --syms - | FileCheck %s --check-prefix=ELF
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 | FileCheck --check-prefix=GFX10 --check-prefix=GFX10-W32 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 | FileCheck --check-prefix=GFX10 --check-prefix=GFX10-W64 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 | FileCheck --check-prefix=GFX10 --check-prefix=GFX10-W32 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 | FileCheck --check-prefix=GFX10 --check-prefix=GFX10-W64 %s

; The SHT_NOTE section contains the output from the .hsa_code_object_*
; directives.

; ELF: Section {
; ELF: Name: .text
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0x6)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_EXECINSTR (0x4)
; ELF: }

; ELF: SHT_NOTE
; ELF: Flags [ (0x2)
; ELF: SHF_ALLOC (0x2)
; ELF: ]
; ELF: SectionData (
; ELF:   0000: 07000000 A8020000 20000000 414D4447
; ELF:   0010: 50550000 83AE616D 64687361 2E6B6572
; ELF:   0020: 6E656C73 928DA52E 61726773 9185AE2E
; ELF:   0030: 61646472 6573735F 73706163 65A6676C
; ELF:   0040: 6F62616C A52E6E61 6D65A36F 7574A72E
; ELF:   0050: 6F666673 657400A5 2E73697A 6508AB2E
; ELF:   0060: 76616C75 655F6B69 6E64AD67 6C6F6261
; ELF:   0070: 6C5F6275 66666572 B92E6772 6F75705F
; ELF:   0080: 7365676D 656E745F 66697865 645F7369
; ELF:   0090: 7A6500B6 2E6B6572 6E617267 5F736567
; ELF:   00A0: 6D656E74 5F616C69 676E08B5 2E6B6572
; ELF:   00B0: 6E617267 5F736567 6D656E74 5F73697A
; ELF:   00C0: 6508B82E 6D61785F 666C6174 5F776F72
; ELF:   00D0: 6B67726F 75705F73 697A65CD 0400A52E
; ELF:   00E0: 6E616D65 A673696D 706C65BB 2E707269
; ELF:   00F0: 76617465 5F736567 6D656E74 5F666978
; ELF:   0100: 65645F73 697A6500 AB2E7367 70725F63
; ELF:   0110: 6F756E74 06B12E73 6770725F 7370696C
; ELF:   0120: 6C5F636F 756E7400 A72E7379 6D626F6C
; ELF:   0130: A973696D 706C652E 6B64AB2E 76677072
; ELF:   0140: 5F636F75 6E7403B1 2E766770 725F7370
; ELF:   0150: 696C6C5F 636F756E 7400AF2E 77617665
; ELF:   0160: 66726F6E 745F7369 7A65408D A52E6172
; ELF:   0170: 677390B9 2E67726F 75705F73 65676D65
; ELF:   0180: 6E745F66 69786564 5F73697A 6500B62E
; ELF:   0190: 6B65726E 6172675F 7365676D 656E745F
; ELF:   01A0: 616C6967 6E04B52E 6B65726E 6172675F
; ELF:   01B0: 7365676D 656E745F 73697A65 00B82E6D
; ELF:   01C0: 61785F66 6C61745F 776F726B 67726F75
; ELF:   01D0: 705F7369 7A65CD04 00A52E6E 616D65B2
; ELF:   01E0: 73696D70 6C655F6E 6F5F6B65 726E6172
; ELF:   01F0: 6773BB2E 70726976 6174655F 7365676D
; ELF:   0200: 656E745F 66697865 645F7369 7A6500AB
; ELF:   0210: 2E736770 725F636F 756E7400 B12E7367
; ELF:   0220: 70725F73 70696C6C 5F636F75 6E7400A7
; ELF:   0230: 2E73796D 626F6CB5 73696D70 6C655F6E
; ELF:   0240: 6F5F6B65 726E6172 67732E6B 64AB2E76
; ELF:   0250: 6770725F 636F756E 7402B12E 76677072
; ELF:   0260: 5F737069 6C6C5F63 6F756E74 00AF2E77
; ELF:   0270: 61766566 726F6E74 5F73697A 6540AD61
; ELF:   0280: 6D646873 612E7461 72676574 BD616D64
; ELF:   0290: 67636E2D 756E6B6E 6F776E2D 616D6468
; ELF:   02A0: 73612D2D 67667837 3030AE61 6D646873
; ELF:   02B0: 612E7665 7273696F 6E920101
; ELF: )

; ELF: Symbol {
; ELF: Name: simple
; ELF: Size: 32
; ELF: }

; HSA-NOT: .AMDGPU.config
; HSA: .text
; HSA-CI: .amdgcn_target "amdgcn-unknown-amdhsa--gfx700"
; HSA-VI: .amdgcn_target "amdgcn-unknown-amdhsa--gfx801"

; HSA-LABEL: {{^}}simple:

; PRE-GFX10: wavefront_size = 6

; HSA: s_load_{{dwordx2|b64}} s[{{[0-9]+:[0-9]+}}], s[4:5], 0x0

; Make sure we are setting the ATC bit:
; HSA-CI: s_mov_b32 s[[HI:[0-9]]], 0x100f000
; On VI+ we also need to set MTYPE = 2
; HSA-VI: s_mov_b32 s[[HI:[0-9]]], 0x1100f000
; Make sure we generate flat store for HSA
; PRE-GFX10: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}
; GFX10: global_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, off

; HSA: .amdhsa_user_sgpr_private_segment_buffer 1
; HSA: .amdhsa_user_sgpr_kernarg_segment_ptr 1

; PRE-GFX10-NOT: .amdhsa_wavefront_size32
; GFX10-W32: .amdhsa_wavefront_size32 1
; GFX10-W64: .amdhsa_wavefront_size32 0

; HSA: .Lfunc_end0:
; HSA: .size   simple, .Lfunc_end0-simple

define amdgpu_kernel void @simple(ptr addrspace(1) %out) {
entry:
  store i32 0, ptr addrspace(1) %out
  ret void
}

; HSA-LABEL: {{^}}simple_no_kernargs:
; HSA: .amdhsa_user_sgpr_kernarg_segment_ptr 0
define amdgpu_kernel void @simple_no_kernargs() {
entry:
  store volatile i32 0, ptr addrspace(1) undef
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
