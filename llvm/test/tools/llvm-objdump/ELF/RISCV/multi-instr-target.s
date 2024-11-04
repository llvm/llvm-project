# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s | \
# RUN:     llvm-objdump -d -M no-aliases --no-show-raw-insn - | \
# RUN:     FileCheck %s

## Test multiple interleaved auipc/jalr pairs
# CHECK: auipc t0, 0
1: auipc t0, %pcrel_hi(bar)
# CHECK: auipc t1, 0
2: auipc t1, %pcrel_hi(bar)
# CHECK: jalr ra, {{[0-9]+}}(t0) <bar>
jalr %pcrel_lo(1b)(t0)
## Target should not be printed because the call above clobbers register state
# CHECK: jalr ra, {{[0-9]+}}(t1){{$}}
jalr %pcrel_lo(2b)(t1)

## Test that auipc+jalr with a write to the target register in between does not
## print the target
# CHECK: auipc t0, 0
1: auipc t0, %pcrel_hi(bar)
# CHECK: c.li t0, 0
li t0, 0
# CHECK: jalr ra, {{[0-9]+}}(t0){{$}}
jalr %pcrel_lo(1b)(t0)

## Test that auipc+jalr with a write to an unrelated register in between does
## print the target
# CHECK: auipc t0, 0
1: auipc t0, %pcrel_hi(bar)
# CHECK: c.li t1, 0
li t1, 0
# CHECK: jalr ra, {{[0-9]+}}(t0) <bar>
jalr %pcrel_lo(1b)(t0)

## Test that auipc+jalr with a terminator in between does not print the target
# CHECK: auipc t0, 0
1: auipc t0, %pcrel_hi(bar)
# CHECK: c.j {{.*}} <bar>
j bar
# CHECK: jalr ra, {{[0-9]+}}(t0){{$}}
jalr %pcrel_lo(1b)(t0)

# CHECK-LABEL: <bar>:
bar:
# CHECK: c.nop
nop
