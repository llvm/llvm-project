// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 %s -filetype=null 2>&1 | FileCheck %s

// Each error case aborts parsing of its enclosing .amdgpu_info block: the
// parser returns on the failing directive, which implicitly exits the block
// (there is no block-open state tracked at the top level), and the next
// test case starts fresh at top level. `.end_amdgpu_info` terminators are
// therefore intentionally omitted -- adding them here would themselves
// become "unknown directive" errors, since `.end_amdgpu_info` is only
// recognised inside the block.

// Missing function symbol after .amdgpu_info.
.amdgpu_info
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected symbol name after .amdgpu_info

// Unknown directive inside a .amdgpu_info block.
.amdgpu_info f_unknown_dir
	.amdgpu_bogus 1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: unknown .amdgpu_info directive '.amdgpu_bogus'

// .amdgpu_use with no resource symbol.
.amdgpu_info f_use_missing
	.amdgpu_use
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected resource symbol for .amdgpu_use

// .amdgpu_call with no callee symbol.
.amdgpu_info f_call_missing
	.amdgpu_call
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected callee symbol for .amdgpu_call

// .amdgpu_indirect_call with no type-ID string.
.amdgpu_info f_icall_missing
	.amdgpu_indirect_call
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected type ID string for .amdgpu_indirect_call

// .amdgpu_typeid with no type-ID string.
.amdgpu_info f_typeid_missing
	.amdgpu_typeid
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected type ID string for .amdgpu_typeid

// Non-identifier token where a directive or .end_amdgpu_info is expected.
.amdgpu_info f_bad_token
	123
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected directive or .end_amdgpu_info
