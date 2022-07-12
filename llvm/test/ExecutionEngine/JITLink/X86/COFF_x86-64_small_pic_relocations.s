# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-windows-msvc -relax-relocations=false \
# RUN:   -position-independent -filetype=obj -o %t/coff_sm_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:              -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:              -abs external_data=0xdeadbeef \
# RUN:              -abs extern_out_of_range32=0x7fff00000000 \
# RUN:              -check %s %t/coff_sm_reloc.o

	.text

	.def main;
	.scl 2;
	.type 32;
	.endef
	.globl main
	.p2align 4, 0x90
main:
	retq

# Check a IMAGE_REL_AMD64_REL32 relocation to local function symbol.
# jitlink-check: decode_operand(test_rel32_func, 0) = named_func - next_pc(test_rel32_func)
	.def test_rel32_func;
	.scl 2;
	.type 32;
	.endef
	.globl test_rel32_func
	.p2align 4, 0x90
test_rel32_func:
	callq named_func

# Check a IMAGE_REL_AMD64_REL32 relocation to local data symbol.
# jitlink-check: decode_operand(test_rel32_data, 4) = named_data - next_pc(test_rel32_data)
	.def test_rel32_data;
	.scl 2;
	.type 32;
	.endef
	.globl test_rel32_data
	.p2align 4, 0x90
test_rel32_data:
    leaq named_data(%rip), %rax

# Check that calls to external functions out-of-range from the callsite trigger 
# the generation of stubs and GOT entries. This produces a BranchPCRel32 edge, 
# but STUB table manager will create a STUB sequence because external function 
# is out-of-range from the callsite.
#
# jitlink-check: decode_operand(test_call_extern_out_of_range32, 0) = \
# jitlink-check:     stub_addr(coff_sm_reloc.o, extern_out_of_range32) - \
# jitlink-check:        next_pc(test_call_extern_out_of_range32)
# jitlink-check: *{8}(got_addr(coff_sm_reloc.o, extern_out_of_range32)) = \
# jitlink-check:     extern_out_of_range32
	.def test_call_extern_out_of_range32;
	.scl 2;
	.type 32;
	.endef
	.globl main
	.p2align 4, 0x90
test_call_extern_out_of_range32:
	callq extern_out_of_range32
	retq

# Local named data/func that is used in conjunction with other test cases
	.text
	.def named_func;
	.scl 2;
	.type 32;
	.endef
	.globl named_func
	.p2align 4, 0x90
named_func:
	retq

	.data
	.p2align 3
named_data:
	.quad 53

