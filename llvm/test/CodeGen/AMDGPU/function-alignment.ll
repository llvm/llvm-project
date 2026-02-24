; Test preferred alignment of non-entry functions on different AMDGPU
; architectures. Cache-line alignment (default) and fetch-only alignment
; (via -amdgpu-align-functions-for-fetch-only) are both tested.
;
; GFX9:  cache line = 64B  (.p2align 6), fetch = 32B (.p2align 5)
; GFX10: cache line = 64B  (.p2align 6), fetch = 4B  (.p2align 2)
; GFX11: cache line = 128B (.p2align 7), fetch = 4B  (.p2align 2)
; GFX12: cache line = 128B (.p2align 7), fetch = 4B  (.p2align 2)

; --- Default (cache line alignment) ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9-CACHE %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 < %s | FileCheck -check-prefix=GFX10-CACHE %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11-CACHE %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefix=GFX12-CACHE %s

; --- Fetch-only alignment ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-align-functions-for-fetch-only < %s | FileCheck -check-prefix=GFX9-FETCH %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -amdgpu-align-functions-for-fetch-only < %s | FileCheck -check-prefix=GFX10-FETCH %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-align-functions-for-fetch-only < %s | FileCheck -check-prefix=GFX11-FETCH %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -amdgpu-align-functions-for-fetch-only < %s | FileCheck -check-prefix=GFX12-FETCH %s

; --- Optsize: alignment drops to minimum (Align(4) = .p2align 2) ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=OPTSIZE %s

; --- IR align attribute: ensureAlignment must not lower explicit alignment ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=EXPLICIT-ALIGN %s

; --- -align-all-functions=1 with optsize: verify floor at Align(4) ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -align-all-functions=1 < %s | FileCheck -check-prefix=ALIGN-ALL %s

; --- Entry function: 256B alignment unchanged ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=ENTRY %s


; Non-entry function: alignment depends on architecture and option.
define void @non_entry_func() {
; GFX9-CACHE:       .p2align 6{{$}}
; GFX9-CACHE:       non_entry_func:

; GFX10-CACHE:      .p2align 6{{$}}
; GFX10-CACHE:      non_entry_func:

; GFX11-CACHE:      .p2align 7{{$}}
; GFX11-CACHE:      non_entry_func:

; GFX12-CACHE:      .p2align 7{{$}}
; GFX12-CACHE:      non_entry_func:

; GFX9-FETCH:       .p2align 5{{$}}
; GFX9-FETCH:       non_entry_func:

; GFX10-FETCH:      .p2align 2{{$}}
; GFX10-FETCH:      non_entry_func:

; GFX11-FETCH:      .p2align 2{{$}}
; GFX11-FETCH:      non_entry_func:

; GFX12-FETCH:      .p2align 2{{$}}
; GFX12-FETCH:      non_entry_func:
  ret void
}

; Non-entry function with optsize: must still be at least Align(4).
define void @optsize_func() optsize {
; OPTSIZE:          .globl optsize_func
; OPTSIZE-NEXT:     .p2align 2{{$}}
  ret void
}

; Non-entry function with explicit IR align 128: ensureAlignment must not lower
; it. On GFX9 default is 64 (cache line), so 128 from IR must be preserved.
define void @explicit_align_func() align 128 {
; EXPLICIT-ALIGN:   .globl explicit_align_func
; EXPLICIT-ALIGN-NEXT: .p2align 7{{$}}
  ret void
}

; Non-entry function with explicit IR align 32 on GFX9: lower than preferred
; (64), so preferred alignment wins. Result: .p2align 6.
define void @low_align_func() align 32 {
; GFX9-CACHE:       .globl low_align_func
; GFX9-CACHE-NEXT:  .p2align 6{{$}}
  ret void
}

; Optsize + -align-all-functions=1: MachineFunction::init sets Align(2), but
; ensureAlignment(4) in AsmPrinter restores the floor. With optsize,
; getPreferredAlignment returns max(Align(1), Align(4)) = Align(4).
define void @align_all_optsize_func() optsize {
; ALIGN-ALL:        .globl align_all_optsize_func
; ALIGN-ALL-NEXT:   .p2align 2{{$}}
  ret void
}

; Entry function: must be 256B aligned regardless of our changes.
define amdgpu_kernel void @entry_func() {
; ENTRY:            .globl entry_func
; ENTRY-NEXT:       .p2align 8{{$}}
  ret void
}
