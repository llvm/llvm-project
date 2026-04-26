; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck --implicit-check-not=warning -check-prefix=WARN %s

; Check the effect that amdgpu-agpr-alloc has on register reservations.
;
; Asm clobbers will print a warning when they clobber reserved
; registers, and should be uniquely identified in the message from the
; !srcloc values.

; The occupancy target warnings should be a side effect of violating
; the register budget with asm.

; WARN: warning: inline asm clobber list contains reserved registers: a0 at line 1
define amdgpu_kernel void @min_num_agpr_0_0__amdgpu_no_agpr() #0 {
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 1}
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="0,0" }

; Check parse of single entry 0

; WARN: warning: inline asm clobber list contains reserved registers: a0 at line 2
define amdgpu_kernel void @min_num_agpr_0__amdgpu_no_agpr() #1 {
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 2}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 2}
  ret void
}

attributes #1 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="0" }


; Undefined use
define amdgpu_kernel void @min_num_agpr_1_1() #2 {
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 3}
  ret void
}

attributes #2 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="1,1" }

; Check parse of single entry 4, interpreted as the minimum. Total budget is 64.
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_4__amdgpu_no_agpr': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: v60 at line 4
define amdgpu_kernel void @min_num_agpr_4__amdgpu_no_agpr() #3 {
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 4}
  call void asm sideeffect "; clobber $0","~{a3}"(), !srcloc !{i32 4}
  call void asm sideeffect "; clobber $0","~{v59}"(), !srcloc !{i32 4}
  call void asm sideeffect "; clobber $0","~{v60}"(), !srcloc !{i32 4}
  ret void
}

attributes #3 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="4" }


; Allocation granularity requires rounding this to use 4 AGPRs, so the
; top 4 VGPRs are unavailable. The maximum agpr count is also padded
; up to the minimum of 4

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ8_1_1': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: a4 at line 5
; WARN: warning: inline asm clobber list contains reserved registers: v60 at line 5
; WARN: warning: inline asm clobber list contains reserved registers: v63 at line 5
define amdgpu_kernel void @min_num_agpr_occ8_1_1() #4 {
  call void asm sideeffect "; clobber $0","~{a3}"(), !srcloc !{i32 5}
  call void asm sideeffect "; clobber $0","~{a4}"(), !srcloc !{i32 5}
  call void asm sideeffect "; clobber $0","~{v59}"(), !srcloc !{i32 5}
  call void asm sideeffect "; clobber $0","~{v60}"(), !srcloc !{i32 5}
  call void asm sideeffect "; clobber $0","~{v63}"(), !srcloc !{i32 5}
  ret void
}

attributes #4 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="1,1" }


; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_64_64__amdgpu_no_agpr': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 6
define amdgpu_kernel void @min_num_agpr_64_64__amdgpu_no_agpr() #5 {
  call void asm sideeffect "; clobber $0","~{a63}"(), !srcloc !{i32 6}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 6}
  ret void
}

attributes #5 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="64,64" }

; No free VGPRs
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 7
define amdgpu_kernel void @min_num_agpr_64_64() #6 {
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 7}
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 7}
  ret void
}

attributes #6 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="64,64" }

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_63_64': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 8
; WARN: warning: inline asm clobber list contains reserved registers: v3 at line 8
define amdgpu_kernel void @min_num_agpr_63_64() #7 {
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 8}
  call void asm sideeffect "; clobber $0","~{v3}"(), !srcloc !{i32 8}
  call void asm sideeffect "; clobber $0","~{a59}"(), !srcloc !{i32 8}
  call void asm sideeffect "; clobber $0","~{a60}"(), !srcloc !{i32 8}
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 8}
  ret void
}

attributes #7 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="63,64" }


; No-op value.
define amdgpu_kernel void @min_num_agpr_occ8_0_64() #8 {
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 9}
  call void asm sideeffect "; clobber $0","~{v59}"(), !srcloc !{i32 9}
  ret void
}

attributes #8 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="0,64" }


; Register budget 64
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ8_11_59': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: a12 at line 10
; WARN: warning: inline asm clobber list contains reserved registers: v52 at line 10
define amdgpu_kernel void @min_num_agpr_occ8_11_59() #9 {
  call void asm sideeffect "; clobber $0","~{a11}"(), !srcloc !{i32 10}
  call void asm sideeffect "; clobber $0","~{a12}"(), !srcloc !{i32 10}
  call void asm sideeffect "; clobber $0","~{v51}"(), !srcloc !{i32 10}
  call void asm sideeffect "; clobber $0","~{v52}"(), !srcloc !{i32 10}
  ret void
}

attributes #9 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="11,59" }


; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ8_12_59': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: a12 at line 11
; WARN: warning: inline asm clobber list contains reserved registers: v52 at line 11
define amdgpu_kernel void @min_num_agpr_occ8_12_59() #10 {
  call void asm sideeffect "; clobber $0","~{a11}"(), !srcloc !{i32 11}
  call void asm sideeffect "; clobber $0","~{a12}"(), !srcloc !{i32 11}
  call void asm sideeffect "; clobber $0","~{v51}"(), !srcloc !{i32 11}
  call void asm sideeffect "; clobber $0","~{v52}"(), !srcloc !{i32 11}
  ret void
}

attributes #10 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="12,59" }


; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ8_12_20': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: a12 at line 12
; WARN: warning: inline asm clobber list contains reserved registers: v52 at line 12
define amdgpu_kernel void @min_num_agpr_occ8_12_20() #11 {
  call void asm sideeffect "; clobber $0","~{a11}"(), !srcloc !{i32 12}
  call void asm sideeffect "; clobber $0","~{a12}"(), !srcloc !{i32 12}
  call void asm sideeffect "; clobber $0","~{v51}"(), !srcloc !{i32 12}
  call void asm sideeffect "; clobber $0","~{v52}"(), !srcloc !{i32 12}
  ret void
}

attributes #11 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="12,20" }


; WARN: warning: inline asm clobber list contains reserved registers: a20 at line 13
define amdgpu_kernel void @min_num_agpr_occ1_12_20() #12 {
  call void asm sideeffect "; clobber $0","~{a12}"(), !srcloc !{i32 13}
  call void asm sideeffect "; clobber $0","~{a19}"(), !srcloc !{i32 13}
  call void asm sideeffect "; clobber $0","~{a20}"(), !srcloc !{i32 13}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 13}
  call void asm sideeffect "; clobber $0","~{v20}"(), !srcloc !{i32 13}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 13}
  ret void
}

attributes #12 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="12,20" }

; WARN: warning: inline asm clobber list contains reserved registers: a20 at line 14
define amdgpu_kernel void @min_num_agpr_occ1_13_20() #13 {
  call void asm sideeffect "; clobber $0","~{a11}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{a12}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{a13}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{a19}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{a20}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{v51}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{v20}"(), !srcloc !{i32 14}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 14}
  ret void
}

attributes #13 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="13,20" }


; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ2_13_20': desired occupancy was 2, final occupancy is 1
; WARN: warning: inline asm clobber list contains reserved registers: a16 at line 15
; WARN: warning: inline asm clobber list contains reserved registers: a20 at line 15
; WARN: warning: inline asm clobber list contains reserved registers: v240 at line 15
define amdgpu_kernel void @min_num_agpr_occ2_13_20() #14 {
  call void asm sideeffect "; clobber $0","~{a15}"(), !srcloc !{i32 15}
  call void asm sideeffect "; clobber $0","~{a16}"(), !srcloc !{i32 15}
  call void asm sideeffect "; clobber $0","~{a20}"(), !srcloc !{i32 15}

  call void asm sideeffect "; clobber $0","~{v239}"(), !srcloc !{i32 15}
  call void asm sideeffect "; clobber $0","~{v240}"(), !srcloc !{i32 15}

  ret void
}

attributes #14 = { "amdgpu-waves-per-eu"="2,2" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="13,20" }


; Test maximum exceeds the hardware limit.
define amdgpu_kernel void @min_num_agpr_occ1_13_257() #15 {
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 16}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 16}
  ret void
}

attributes #15 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="13,257" }


; Test min and max exceeds the hardware limit.
define amdgpu_kernel void @min_num_agpr_occ1_257_257() #16 {
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 17}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 17}
  ret void
}

attributes #16 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="257,257" }


; Test round up hits the hardware limit
define amdgpu_kernel void @min_num_agpr_occ1_255_255() #17 {
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 18}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 18}
  ret void
}

attributes #17 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="255,255" }


; Test round up hits the hardware limit
define amdgpu_kernel void @min_num_agpr_occ1_253_259() #18 {
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 19}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 19}
  ret void
}

attributes #18 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="253,259" }

; With a minimum of 0, we are not required to allocate any AGPRs
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ4_0_64': desired occupancy was 4, final occupancy is 2
; WARN: warning: inline asm clobber list contains reserved registers: a0 at line 20
; WARN: warning: inline asm clobber list contains reserved registers: a63 at line 20
; WARN: warning: inline asm clobber list contains reserved registers: a64 at line 20
; WARN: warning: inline asm clobber list contains reserved registers: v128 at line 20
define amdgpu_kernel void @min_num_agpr_occ4_0_64() #19 {
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 20}
  call void asm sideeffect "; clobber $0","~{a63}"(), !srcloc !{i32 20}
  call void asm sideeffect "; clobber $0","~{a64}"(), !srcloc !{i32 20}
  call void asm sideeffect "; clobber $0","~{v63}"(), !srcloc !{i32 20}
  call void asm sideeffect "; clobber $0","~{v64}"(), !srcloc !{i32 20}
  call void asm sideeffect "; clobber $0","~{v127}"(), !srcloc !{i32 20}
  call void asm sideeffect "; clobber $0","~{v128}"(), !srcloc !{i32 20}
  ret void
}

attributes #19 = { "amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="0,64" }


; With a non-0 minimum, we must allocate at least 4 AGPRs. The rest of
; the budget is for VGPRs.
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ4_1_64': desired occupancy was 4, final occupancy is 2
; WARN: warning: inline asm clobber list contains reserved registers: a4 at line 21
; WARN: warning: inline asm clobber list contains reserved registers: a63 at line 21
; WARN: warning: inline asm clobber list contains reserved registers: a64 at line 21
; WARN: warning: inline asm clobber list contains reserved registers: v124 at line 21
define amdgpu_kernel void @min_num_agpr_occ4_1_64() #20 {
  call void asm sideeffect "; clobber $0","~{a0}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{a3}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{a4}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{a63}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{a64}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{v63}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{v64}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{v123}"(), !srcloc !{i32 21}
  call void asm sideeffect "; clobber $0","~{v124}"(), !srcloc !{i32 21}
  ret void
}

attributes #20 = { "amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="1,64" }

; 128 vector registers
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ4_32_64': desired occupancy was 4, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a32 at line 22
; WARN: warning: inline asm clobber list contains reserved registers: v96 at line 22
define amdgpu_kernel void @min_num_agpr_occ4_32_64() #21 {
  call void asm sideeffect "; clobber $0","~{a31}"(), !srcloc !{i32 22}
  call void asm sideeffect "; clobber $0","~{a32}"(), !srcloc !{i32 22}
  call void asm sideeffect "; clobber $0","~{v95}"(), !srcloc !{i32 22}
  call void asm sideeffect "; clobber $0","~{v96}"(), !srcloc !{i32 22}
  ret void
}

attributes #21 = { "amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="32,64" }

; Evenly partition the 128 vector registers
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ4_64_64': desired occupancy was 4, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a64 at line 23
; WARN: warning: inline asm clobber list contains reserved registers: v64 at line 23
define amdgpu_kernel void @min_num_agpr_occ4_64_64() #22 {
  call void asm sideeffect "; clobber $0","~{a63}"(), !srcloc !{i32 23}
  call void asm sideeffect "; clobber $0","~{a64}"(), !srcloc !{i32 23}
  call void asm sideeffect "; clobber $0","~{v63}"(), !srcloc !{i32 23}
  call void asm sideeffect "; clobber $0","~{v64}"(), !srcloc !{i32 23}
  ret void
}

attributes #22 = { "amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="64,64" }

; We are not required to allocate any AGPRs, but they are available
; with a budget of 512 vector registers. We are artificially limiting
; to use 64.

; WARN: warning: inline asm clobber list contains reserved registers: a64 at line 24
define amdgpu_kernel void @min_num_agpr_occ1_0_64() #23 {
  call void asm sideeffect "; clobber $0","~{a63}"(), !srcloc !{i32 24}
  call void asm sideeffect "; clobber $0","~{a64}"(), !srcloc !{i32 24}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 24}
  ret void
}

attributes #23 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="0,64" }

; WARN: warning: inline asm clobber list contains reserved registers: a68 at line 25
define amdgpu_kernel void @min_num_agpr_occ1_0_68() #24 {
  call void asm sideeffect "; clobber $0","~{a67}"(), !srcloc !{i32 25}
  call void asm sideeffect "; clobber $0","~{a68}"(), !srcloc !{i32 25}
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 25}
  ret void
}

attributes #24 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="0,68" }


; The total vector register budget is 128, claim more than that for
; the minimum AGPRs. This checks for an assertion.
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ10__min_agpr_129': desired occupancy was 8, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a128 at line 26
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 26
define amdgpu_kernel void @min_num_agpr_occ10__min_agpr_129() #25 {
  call void asm sideeffect "; clobber $0","~{a127}"(), !srcloc !{i32 26}
  call void asm sideeffect "; clobber $0","~{a128}"(), !srcloc !{i32 26}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 26}
  ret void
}

attributes #25 = { "amdgpu-waves-per-eu"="8,10" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="129" }

; Check for another assertion, request beyond the budget.

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ10__min_agpr_129_129': desired occupancy was 8, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a128 at line 27
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 27
define amdgpu_kernel void @min_num_agpr_occ10__min_agpr_129_129() #26 {
  call void asm sideeffect "; clobber $0","~{a127}"(), !srcloc !{i32 27}
  call void asm sideeffect "; clobber $0","~{a128}"(), !srcloc !{i32 27}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 27}
  ret void
}

attributes #26 = { "amdgpu-waves-per-eu"="8,10" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="129,129" }

; The total vector register budget is 128, claim all of it for AGPRs.

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ10__min_agpr_128': desired occupancy was 8, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a128 at line 28
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 28

define amdgpu_kernel void @min_num_agpr_occ10__min_agpr_128() #27 {
  call void asm sideeffect "; clobber $0","~{a127}"(), !srcloc !{i32 28}
  call void asm sideeffect "; clobber $0","~{a128}"(), !srcloc !{i32 28}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 28}
  ret void
}

attributes #27 = { "amdgpu-waves-per-eu"="8,10" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="128" }

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ10__min_agpr_257': desired occupancy was 8, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a128 at line 29
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 29
define amdgpu_kernel void @min_num_agpr_occ10__min_agpr_257() #28 {
  call void asm sideeffect "; clobber $0","~{a127}"(), !srcloc !{i32 29}
  call void asm sideeffect "; clobber $0","~{a128}"(), !srcloc !{i32 29}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 29}
  ret void
}

attributes #28 = { "amdgpu-waves-per-eu"="8,10" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="257" }

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ10__min_agpr_257_257': desired occupancy was 8, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: a128 at line 30
; WARN: warning: inline asm clobber list contains reserved registers: v0 at line 30
define amdgpu_kernel void @min_num_agpr_occ10__min_agpr_257_257() #29 {
  call void asm sideeffect "; clobber $0","~{a127}"(), !srcloc !{i32 30}
  call void asm sideeffect "; clobber $0","~{a128}"(), !srcloc !{i32 30}
  call void asm sideeffect "; clobber $0","~{v0}"(), !srcloc !{i32 30}
  ret void
}

attributes #29 = { "amdgpu-waves-per-eu"="8,10" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="257,257" }


; The total vector register budget is 96

; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ5__min_agpr_8_256': desired occupancy was 5, final occupancy is 4
; WARN: warning: inline asm clobber list contains reserved registers: v88 at line 31
; WARN: warning: inline asm clobber list contains reserved registers: a8 at line 31
define amdgpu_kernel void @min_num_agpr_occ5__min_agpr_8_256() #30 {
  call void asm sideeffect "; clobber $0","~{v87}"(), !srcloc !{i32 31}
  call void asm sideeffect "; clobber $0","~{v88}"(), !srcloc !{i32 31}
  call void asm sideeffect "; clobber $0","~{a7}"(), !srcloc !{i32 31}
  call void asm sideeffect "; clobber $0","~{a8}"(), !srcloc !{i32 31}
  ret void
}

attributes #30 = { "amdgpu-waves-per-eu"="5,5" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="8,256" }

; The total vector register budget is 96
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ5__min_agpr_8': desired occupancy was 5, final occupancy is 4
; WARN: warning: inline asm clobber list contains reserved registers: v88 at line 32
; WARN: warning: inline asm clobber list contains reserved registers: a8 at line 32
define amdgpu_kernel void @min_num_agpr_occ5__min_agpr_8() #31 {
  call void asm sideeffect "; clobber $0","~{v87}"(), !srcloc !{i32 32}
  call void asm sideeffect "; clobber $0","~{v88}"(), !srcloc !{i32 32}
  call void asm sideeffect "; clobber $0","~{a7}"(), !srcloc !{i32 32}
  call void asm sideeffect "; clobber $0","~{a8}"(), !srcloc !{i32 32}
  ret void
}

; budget is 96
; WARN: warning: inline asm clobber list contains reserved registers: v88 at line 33
define amdgpu_kernel void @min_num_agpr_occ5__min_agpr_8_no_agpr_references() #31 {
  call void asm sideeffect "; clobber $0","~{v87}"(), !srcloc !{i32 33}
  call void asm sideeffect "; clobber $0","~{v88}"(), !srcloc !{i32 33}
  ret void
}

attributes #31 = { "amdgpu-waves-per-eu"="5,5" "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-agpr-alloc"="8" }


; register budget 256
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'min_num_agpr_occ2__min_agpr_93': desired occupancy was 2, final occupancy is 1
; WARN: warning: inline asm clobber list contains reserved registers: v160 at line 34
; WARN: warning: inline asm clobber list contains reserved registers: a96 at line 34
define amdgpu_kernel void @min_num_agpr_occ2__min_agpr_93() #33 {
  call void asm sideeffect "; clobber $0","~{v159}"(), !srcloc !{i32 34}
  call void asm sideeffect "; clobber $0","~{v160}"(), !srcloc !{i32 34}
  call void asm sideeffect "; clobber $0","~{a95}"(), !srcloc !{i32 34}
  call void asm sideeffect "; clobber $0","~{a96}"(), !srcloc !{i32 34}
  ret void
}

attributes #33 = { "amdgpu-waves-per-eu"="2,2" "amdgpu-flat-work-group-size"="1,256" "amdgpu-agpr-alloc"="93" }

; register budget 512, no warnings and fully allocated
define amdgpu_kernel void @min_num_agpr_occ1__min_agpr_93() #34 {
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 35}
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 35}
  ret void
}

attributes #34 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-agpr-alloc"="93" }

; register budget 256
; WARN: warning: inline asm clobber list contains reserved registers: a96 at line 36
define amdgpu_kernel void @min_num_agpr_occ1__min_agpr_93_93() #35 {
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 36}
  call void asm sideeffect "; clobber $0","~{a95}"(), !srcloc !{i32 36}
  call void asm sideeffect "; clobber $0","~{a96}"(), !srcloc !{i32 36}
  ret void
}

attributes #35 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-agpr-alloc"="93,93" }

; register budget 512, fully allocated and no warnings.
define amdgpu_kernel void @min_num_agpr_occ1__min_agpr_256() #36 {
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 37}
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 37}
  ret void
}

attributes #36 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-agpr-alloc"="256" }

; register budget 512, fully allocated and no warnings.
define amdgpu_kernel void @min_num_agpr_occ1__min_agpr_256_256() #37 {
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 38}
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 38}
  ret void
}

attributes #37 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-agpr-alloc"="256,256" }

; register budget 512, fully allocated and no warnings.
define amdgpu_kernel void @occ1_min_agpr_no_attr() #38 {
  call void asm sideeffect "; clobber $0","~{v255}"(), !srcloc !{i32 39}
  call void asm sideeffect "; clobber $0","~{a255}"(), !srcloc !{i32 39}
  ret void
}

attributes #38 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="256,256" }
