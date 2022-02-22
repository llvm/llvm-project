# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu %s -filetype=obj -o - | \
# RUN:    llvm-objdump -D -r - | FileCheck %s
	.text
test:                                   # @test
        add 5, 3, 4
        extsw 3, 5
        .space 32776
lab2:
        lxv 5, (lab2-test)@l(4)
        ld 5, (lab2-test)@l(4)
        lwz 5, (lab2-test)@l(4)
        lxv 5, 8389632@l(4)
        ld 5, 8389632@l(4)
        lwz 5, 8389632@l(4)
        blr

# CHECK: lxv 5, -32752(4)
# CHECK: ld 5, -32752(4)
# CHECK: lwz 5, -32752(4)
# CHECK: lxv 5, 1024(4)
# CHECK: ld 5, 1024(4)
# CHECK: lwz 5, 1024(4)
