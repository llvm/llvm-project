// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 %s -o - -filetype=null 2>&1 | FileCheck %s

// Missing symbol name after .amdgpu_resource_usage.
	.amdgpu_resource_usage
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected symbol name after .amdgpu_resource_usage

// Duplicate field directive.
	.amdgpu_resource_usage fn2
		.num_vgpr 0
		.num_vgpr 1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: resource usage directives already declared
	.end_amdgpu_resource_usage

// Negative value.
	.amdgpu_resource_usage fn3
		.num_vgpr -1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: value must be non-negative
	.end_amdgpu_resource_usage

// Unknown field.
	.amdgpu_resource_usage fn4
		.num_vgpr 0
		.num_agpr 0
		.num_sgpr 0
		.named_barrier 0
		.private_seg_size 0
		.uses_vcc 0
		.uses_flat_scratch 0
		.has_dyn_sized_stack 0
		.bogus_field 42
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: unknown field '.bogus_field' in .amdgpu_resource_usage
	.end_amdgpu_resource_usage

// Missing required field (.num_sgpr omitted).
	.amdgpu_resource_usage fn5
		.num_vgpr 0
		.num_agpr 0
		.named_barrier 0
		.private_seg_size 0
		.uses_vcc 0
		.uses_flat_scratch 0
		.has_dyn_sized_stack 0
	.end_amdgpu_resource_usage
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: missing required .num_sgpr directive in .amdgpu_resource_usage
