# RUN: llvm-objdump -d %p/Inputs/riscv-ar | FileCheck %s

# CHECK:   auipc a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   ld a0, {{-?0x[0-9a-fA-F]+}}(a0) <ldata+0xfa4>
# CHECK:   auipc a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addi a0, a0, {{-?0x[0-9a-fA-F]+}} <gdata>
# CHECK:   auipc	a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addi a0, a0, {{-?0x[0-9a-fA-F]+}} <gdata>
# CHECK:   auipc	a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   lw a0, {{-?0x[0-9a-fA-F]+}}(a0) <gdata>
# CHECK:   auipc	a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addi a0, a0, {{-?0x[0-9a-fA-F]+}} <ldata>
# CHECK:   auipc	ra, {{-?0x[0-9a-fA-F]+}}
# CHECK:   jalr {{-?0x[0-9a-fA-F]+}}(ra) <func>
# CHECK:   auipc	t1, {{-?0x[0-9a-fA-F]+}}
# CHECK:   jr {{-?0x[0-9a-fA-F]+}}(t1) <func>
# CHECK:   lui a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addiw a0, a0, {{-?0x[0-9a-fA-F]+}} <gdata+0x12242678>
# CHECK:   lui a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addiw	a0, a0, {{-?0x[0-9a-fA-F]+}} <gdata+0x1438ad>
# CHECK:   slli a0, a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addi a0, a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   slli a0, a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addi a0, a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   slli a0, a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addi a0, a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   lui a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   lui a0, {{-?0x[0-9a-fA-F]+}}
# CHECK:   addiw a0, a0, {{-?0x[0-9a-fA-F]+}} <_start+0xfefff>

.global _start
.text
_start:
  la a0, gdata
  lla a0, gdata
  lla a0, gdata
  lw a0, gdata
  lla a0, ldata

  call func
  tail func

  li a0, 0x12345678
  li a0, 0x1234567890abcdef
  li a0, 0x10000
  li a0, 0xfffff

  .skip 0x100000
func:
  ret

ldata:
  .int 0

.data
gdata:
  .int 0
