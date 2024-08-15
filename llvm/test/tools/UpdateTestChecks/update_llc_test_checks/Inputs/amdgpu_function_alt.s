	.text
	.amdgcn_target "amdgcn--amdpal--gfx1030"
	.globl	sample
	.p2align	2
	.type	sample,@function
sample:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mul_f32_e32 v0, v0, v0
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	sample, .Lfunc_end0-sample

	.p2alignl 6, 3214868480
	.fill 48, 4, 3214868480
	.section	".note.GNU-stack","",@progbits
	.amd_amdgpu_isa "amdgcn--amdpal--gfx1030"
	.amdgpu_pal_metadata
---
amdpal.pipelines:
  - .api:            !str ''
    .shader_functions:
      sample:
        .backend_stack_size: 0
        .lds_size:       0
        .sgpr_count:     0x20
        .stack_frame_size_in_bytes: 0
        .vgpr_count:     0x1
amdpal.version:
  - 0x3
  - 0
...
	.end_amdgpu_pal_metadata
