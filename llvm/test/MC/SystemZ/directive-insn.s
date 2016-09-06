# RUN: llvm-mc -triple s390x-linux-gnu -filetype=obj %s | \
# RUN: llvm-objdump -mcpu=zEC12 -disassemble - | FileCheck %s

# Test the .insn directive which provides a way of encoding an instruction
# directly. It takes a format, encoding, and operands based on the format.

#CHECK: 01 01                 pr
  .insn e,0x0101

#CHECK: a7 18 12 34           lhi %r1, 4660
  .insn ri,0xa7080000,%r1,0x1234

# GAS considers this instruction's immediate operand to be PC relative.
#CHECK: ec 12 00 06 00 76     crj %r1, %r2, 0, 0x12
  .insn rie,0xec0000000076,%r1,%r2,12
#CHECK: ec 12 00 03 00 64     cgrj %r1, %r2, 0, 0x12
  .insn rie,0xec0000000064,%r1,%r2,label.rie
#CHECK: label.rie:
label.rie:

# GAS considers this instruction's immediate operand to be PC relative.
#CHECK: c6 1d 00 00 00 06     crl %r1, 0x1e
  .insn ril,0xc60d00000000,%r1,12
#CHECK: c6 18 00 00 00 03     cgrl %r1, 0x1e
  .insn ril,0xc60800000000,%r1,label.ril
#CHECK: label.ril:
label.ril:

#CHECK: c2 2b 80 00 00 00     alfi %r2, 2147483648
  .insn rilu,0xc20b00000000,%r2,0x80000000

#CHECK: ec 1c f0 a0 34 fc     cgible %r1, 52, 160(%r15)
  .insn ris,0xec00000000fc,%r1,0x34,0xc,160(%r15)

# Test using an integer in place of a register.
#CHECK: 18 23                 lr %r2, %r3
  .insn rr,0x1800,2,3

#CHECK: b9 14 00 45           lgfr %r4, %r5
  .insn rre,0xb9140000,%r4,%r5

# Test FP and GR registers in a single directive.
#CHECK: b3 c1 00 fe           ldgr %f15, %r14
  .insn rre,0xb3c10000,%f15,%r14

# Test using an integer in place of a register.
#CHECK: b3 44 34 12           ledbra %f1, 3, %f2, 4
  .insn rrf,0xb3440000,%f1,2,%f3,4

#CHECK: ec 34 f0 b4 a0 e4     cgrbhe %r3, %r4, 180(%r15)
  .insn rrs,0xec00000000e4,%r3,%r4,0xa,180(%r15)

#CHECK: ba 01 f0 a0           cs %r0, %r1, 160(%r15)
  .insn rs,0xba000000,%r0,%r1,160(%r15)

# GAS considers this instruction's immediate operand to be PC relative.
#CHECK: 84 13 00 04           brxh %r1, %r3, 0x4a
  .insn rsi,0x84000000,%r1,%r3,8
#CHECK: 84 13 00 02           brxh %r1, %r3, 0x4a
  .insn rsi,0x84000000,%r1,%r3,label.rsi
#CHECK: label.rsi:
label.rsi:

# RSE formats are short displacement versions of the RSY formats.
#CHECK: eb 12 f0 a0 00 f8     laa %r1, %r2, 160(%r15)
  .insn rse,0xeb00000000f8,%r1,%r2,160(%r15)

#CHECK: eb 12 f3 45 12 30     csg %r1, %r2, 74565(%r15)
  .insn rsy,0xeb0000000030,%r1,%r2,74565(%r15)

#CHECK: 59 13 f0 a0           c %r1, 160(%r3,%r15)
  .insn rx,0x59000000,%r1,160(%r3,%r15)

#CHECK: ed 13 f0 a0 00 19     cdb %f1, 160(%r3,%r15)
  .insn rxe,0xed0000000019,%f1,160(%r3,%r15)

#CHECK: ed 23 f0 a0 10 1e     madb %f1, %f2, 160(%r3,%r15)
  .insn rxf,0xed000000001e,%f1,%f2,160(%r3,%r15)

#CHECK: ed 12 f1 23 90 65     ldy %f1, -458461(%r2,%r15)
  .insn rxy,0xed0000000065,%f1,-458461(%r2,%r15)

#CHECK: b2 fc f0 a0           tabort 160(%r15)
  .insn s,0xb2fc0000,160(%r15)

#CHECK: 91 34 f0 a0           tm 160(%r15), 52
  .insn si,0x91000000,160(%r15),52

#CHECK: eb f0 fc de ab 51     tmy -344866(%r15), 240
  .insn siy,0xeb0000000051,-344866(%r15),240

#CHECK: e5 60 f0 a0 12 34     tbegin 160(%r15), 4660
  .insn sil,0xe56000000000,160(%r15),0x1234

#CHECK: d9 13 f1 23 e4 56     mvck 291(%r1,%r15), 1110(%r14), %r3
  .insn ss,0xd90000000000,291(%r1,%r15),1110(%r14),%r3

#CHECK: e5 02 10 a0 21 23     strag 160(%r1), 291(%r2)
  .insn sse,0xe50200000000,160(%r1),291(%r2)

#CHECK: c8 31 f0 a0 e2 34     ectg 160(%r15), 564(%r14), %r3
  .insn ssf,0xc80100000000,160(%r15),564(%r14),%r3
