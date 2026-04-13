; Test preferred alignment of non-entry functions on different AMDGPU
; architectures. Preferred alignment matches the instruction cache line size:
;
; GFX9  - cache line = 64B  (.p2align 6)
; GFX10 - cache line = 64B  (.p2align 6)
; GFX11 - cache line = 128B (.p2align 7)
; GFX12 - cache line = 128B (.p2align 7)

; --- Default (cache line alignment) ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefix=GFX12 %s

; --- Optsize: alignment drops to minimum (Align(4) = .p2align 2) ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=OPTSIZE %s

; --- IR align attribute: ensureAlignment must not lower explicit alignment ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=EXPLICIT-ALIGN %s

; --- -align-all-functions=1 with optsize: verify floor at Align(4) ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -align-all-functions=1 < %s | FileCheck -check-prefix=ALIGN-ALL %s

; --- prefalign attribute: overrides target preferred alignment ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=PREFALIGN %s

; --- Entry function: 256B alignment unchanged ---

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=ENTRY %s


; Non-entry function: alignment matches instruction cache line size.
define void @non_entry_func() {
; GFX9:       .p2align 6{{$}}
; GFX9:       non_entry_func:

; GFX10:      .p2align 6{{$}}
; GFX10:      non_entry_func:

; GFX11:      .p2align 7{{$}}
; GFX11:      non_entry_func:

; GFX12:      .p2align 7{{$}}
; GFX12:      non_entry_func:
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

; Non-entry function with explicit IR align 32 on gfx900 -- lower than
; preferred (64), so preferred alignment wins. Result: .p2align 6.
define void @low_align_func() align 32 {
; GFX9:       .globl low_align_func
; GFX9-NEXT:  .p2align 6{{$}}
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

; prefalign(16) on gfx900 overrides target preferred (64) with 16.
; getPreferredAlignment uses prefalign directly instead of getPrefFunctionAlignment.
; Result: max(16, 4) = 16 -> .p2align 4.
define void @prefalign_low_func() prefalign(16) {
; PREFALIGN:        .globl prefalign_low_func
; PREFALIGN-NEXT:   .p2align 4{{$}}
  ret void
}

; prefalign(256) on gfx900 -- higher than target preferred (64).
; Result: max(256, 4) = 256 -> .p2align 8.
define void @prefalign_high_func() prefalign(256) {
; PREFALIGN:        .globl prefalign_high_func
; PREFALIGN-NEXT:   .p2align 8{{$}}
  ret void
}

; prefalign(2) on gfx900 -- below the 4-byte instruction alignment floor.
; ensureAlignment(4) in AsmPrinter guarantees the minimum.
; Result: max(2, 4) = 4 -> .p2align 2.
define void @prefalign_floor_func() prefalign(2) {
; PREFALIGN:        .globl prefalign_floor_func
; PREFALIGN-NEXT:   .p2align 2{{$}}
  ret void
}

; Entry function: must be 256B aligned regardless of our changes.
define amdgpu_kernel void @entry_func() {
; ENTRY:            .globl entry_func
; ENTRY-NEXT:       .p2align 8{{$}}
  ret void
}
