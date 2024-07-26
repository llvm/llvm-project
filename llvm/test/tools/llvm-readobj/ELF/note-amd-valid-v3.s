## This test is checking the handling of valid note entries for AMDGPU code
## object v3.

# REQUIRES: amdgpu-registered-target

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj < %s | llvm-readobj --notes - | FileCheck %s --match-full-lines --check-prefix=LLVM
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj < %s | llvm-readelf --notes - | FileCheck %s --match-full-lines --check-prefix=GNU

#LLVM:       NoteSections [
#LLVM-NEXT:    NoteSection {
#LLVM-NEXT:      Name: .note
#LLVM-NEXT:      Offset: 0x40
#LLVM-NEXT:      Size: 0x110
#LLVM-NEXT:      Notes [
#LLVM-NEXT:      {
#LLVM-NEXT:        Owner: AMDGPU
#LLVM-NEXT:        Data size: 0xFC
#LLVM-NEXT:        Type: NT_AMDGPU_METADATA (AMDGPU Metadata)
#LLVM-NEXT:        AMDGPU Metadata: ---
#LLVM-NEXT:  amdhsa.kernels:
#LLVM-NEXT:    - .group_segment_fixed_size: 16
#LLVM-NEXT:      .kernarg_segment_align: 64
#LLVM-NEXT:      .kernarg_segment_size: 8
#LLVM-NEXT:      .max_flat_workgroup_size: 256
#LLVM-NEXT:      .name:           test_kernel
#LLVM-NEXT:      .private_segment_fixed_size: 32
#LLVM-NEXT:      .sgpr_count:     14
#LLVM-NEXT:      .symbol:         'test_kernel@kd'
#LLVM-NEXT:      .vgpr_count:     40
#LLVM-NEXT:      .wavefront_size: 128
#LLVM-NEXT:  amdhsa.version:
#LLVM-NEXT:    - 1
#LLVM-NEXT:    - 0
#LLVM-NEXT:  ...
#LLVM-EMPTY:
#LLVM-NEXT:      }
#LLVM-NEXT:     ]
#LLVM-NEXT:    }
#LLVM-NEXT:  ]

# GNU:      Displaying notes found in: .note
# GNU-NEXT:   Owner                Data size        Description
# GNU-NEXT:   AMDGPU               0x000000fc       NT_AMDGPU_METADATA (AMDGPU Metadata)
# GNU-NEXT:     AMDGPU Metadata:
# GNU-NEXT:         ---
# GNU-NEXT: amdhsa.kernels:
# GNU-NEXT:   - .group_segment_fixed_size: 16
# GNU-NEXT:     .kernarg_segment_align: 64
# GNU-NEXT:     .kernarg_segment_size: 8
# GNU-NEXT:     .max_flat_workgroup_size: 256
# GNU-NEXT:     .name:           test_kernel
# GNU-NEXT:     .private_segment_fixed_size: 32
# GNU-NEXT:     .sgpr_count:     14
# GNU-NEXT:     .symbol:         'test_kernel@kd'
# GNU-NEXT:     .vgpr_count:     40
# GNU-NEXT:     .wavefront_size: 128
# GNU-NEXT: amdhsa.version:
# GNU-NEXT:   - 1
# GNU-NEXT:   - 0
# GNU-NEXT: ...

.amdgpu_metadata
  amdhsa.version:
    - 1
    - 0
  amdhsa.kernels:
    - .name:                       test_kernel
      .symbol:                     test_kernel@kd
      .group_segment_fixed_size:   16
      .kernarg_segment_align:      64
      .kernarg_segment_size:       8
      .max_flat_workgroup_size:    256
      .private_segment_fixed_size: 32
      .sgpr_count:                 14
      .vgpr_count:                 40
      .wavefront_size:             128
.end_amdgpu_metadata
