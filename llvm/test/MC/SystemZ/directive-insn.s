# RUN: llvm-mc -triple s390x-linux-gnu -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=zEC12 -d - | FileCheck %s

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
#CHECK: <label.rie>:
label.rie:

# GAS considers this instruction's immediate operand to be PC relative.
#CHECK: c6 1d 00 00 00 06     crl %r1, 0x1e
  .insn ril,0xc60d00000000,%r1,12
#CHECK: c6 18 00 00 00 03     cgrl %r1, 0x1e
  .insn ril,0xc60800000000,%r1,label.ril
#CHECK: <label.ril>:
label.ril:

#CHECK: c2 2b 80 00 00 00     alfi %r2, 2147483648
  .insn rilu,0xc20b00000000,%r2,0x80000000

#CHECK: ec 1c f0 a0 34 fc     cgible %r1, 52, 160(%r15)
  .insn ris,0xec00000000fc,%r1,0x34,0xc,160(%r15)

#CHECK: ec 1c f0 a0 ff fc     cgible %r1, -1, 160(%r15)
  .insn ris,0xec00000000fc,%r1,255,0xc,160(%r15)

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
#CHECK: 84 13 00 04           brxh %r1, %r3, 0x50
  .insn rsi,0x84000000,%r1,%r3,8
#CHECK: 84 13 00 02           brxh %r1, %r3, 0x50
  .insn rsi,0x84000000,%r1,%r3,label.rsi
#CHECK: <label.rsi>:
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

#CHECK: 91 ff f0 a0           tm 160(%r15), 255
  .insn si,0x91000000,160(%r15),-1

#CHECK: eb f0 fc de ab 51     tmy -344866(%r15), 240
  .insn siy,0xeb0000000051,-344866(%r15),240

#CHECK: eb ff fc de ab 51     tmy -344866(%r15), 255
  .insn siy,0xeb0000000051,-344866(%r15),-1

#CHECK: e5 60 f0 a0 12 34     tbegin 160(%r15), 4660
  .insn sil,0xe56000000000,160(%r15),0x1234

#CHECK: e5 60 f0 a0 ff ff     tbegin 160(%r15), 65535
  .insn sil,0xe56000000000,160(%r15),-1

#CHECK: d9 13 f1 23 e4 56     mvck 291(%r1,%r15), 1110(%r14), %r3
  .insn ss,0xd90000000000,291(%r1,%r15),1110(%r14),%r3

#CHECK: e5 02 10 a0 21 23     strag 160(%r1), 291(%r2)
  .insn sse,0xe50200000000,160(%r1),291(%r2)

#CHECK: c8 31 f0 a0 e2 34     ectg 160(%r15), 564(%r14), %r3
  .insn ssf,0xc80100000000,160(%r15),564(%r14),%r3

#CHECK: 0a bc                 svc 188
      .insn i,0x0a00,0xbc

#CHECK: b2 fa 00 12           niai 1, 2
      .insn ie,0xb2fa0000,0x1,0x2

#CHECK: c5 f1 00 00 10 00 bprp 15, 0x2ae, 0x20ae
      .insn mii,0xc50000000000,15,512,8192

#CHECK: a7 1a af fe           ahi %r1, -20482
      .insn ri,0xa70a0000,%r1,-20482

#CHECK: a7 12 ff ff           tmhh %r1, 65535
      .insn ri_a,0xa7020000,%r1,65535

#CHECK: a7 15 ff fc           bras %r1, 0xb4
      .insn ri_b,0xa7050000,%r1,-8

#CHECK: a7 34 ff fc           jnle 0xb8
      .insn ri_c,0xa7040000,3,-8

#CHECK: ec 10 ff ff 70 72 cit %r1, -1, 7
      .insn rie_a,0xec0000000072,%r1,65535,7

#CHECK: ec 10 23 45 70 72    	cit %r1, 9029, 7
      .insn rie_a,0xec0000000072,%r1,0x2345,7

#CHECK: ec 12 00 00 70 77    	clrj %r1, %r2, 7, 0xd0
      .insn rie_b,0xec0000000077,%r1,%r2,7,0

#CHECK: ec 12 ff fc 34 7e    	cijh %r1, 52, 0xce
      .insn rie_c,0xec000000007e,%r1,52,2,-8

#CHECK: ec 12 ff fc ff 7e     cijh %r1, -1, 0xd4
      .insn rie_c,0xec000000007e,%r1,255,2,-8

#CHECK: ec 12 af fe 00 d8    	ahik %r1, %r2, -20482
      .insn rie_d,0xec00000000d8,%r1,-20482,%r2

#CHECK: ec 12 ff fc 00 44    	brxhg %r1, %r2, 0xe0
      .insn rie_e,0xec0000000044,%r1,-8,%r2

#CHECK: ec 12 fa ca de 5d    	risbhg	%r1, %r2, 250, 202, 222
      .insn rie_f,0xec000000005d,%r1,%r2,250,202,222

#CHECK: c2 15 00 fa ca de    	slfi	%r1, 16435934
      .insn ril_a,0xc20500000000,%r1,16435934

#CHECK: c2 15 ff ff ff ff slfi %r1, 4294967295
      .insn ril_a,0xc20500000000,%r1,4294967295

#CHECK: c0 15 ff ff ff fc    	brasl	%r1, 0xf8
      .insn ril_b,0xc00500000000,%r1,-8

#CHECK: c0 04 ff ff ff fc    	jgnop	0xfe
      .insn ril_c,0xc00400000000,0,-8

#CHECK: b3 3e 10 23  	madr	%f1, %f2, %f3
      .insn rrd,0xb33e0000,%f1,%f3,%f2

#CHECK: b3 d2 30 12  	adtr	%f1, %f2, %f3
      .insn rrf,0xb3d20000,%f1,%f2,%f3,0

#CHECK: b3 d2 31 12  	adtra	%f1, %f2, %f3, 1
      .insn rrf_a,0xb3d20000,%f1,%f2,%f3,1

#CHECK: b3 f6 20 13  	iedtr	%f1, %f2, %r3
      .insn rrf_b,0xb3f60000,%f1,%r3,%f2,0

#CHECK: b9 72 10 23  	crt	%r2, %r3, 1
      .insn rrf_c,0xb9720000,%r2,%r3,1

#CHECK: b3 e3 01 23  	csdtr	%r2, %f3, 1
      .insn rrf_d,0xb3e30000,%r2,%f3,1

#CHECK: b3 d7 12 34  	fidtr	%f3, 1, %f4, 2
      .insn rrf_e,0xb3d70000,%r3,%f4,1,2

#CHECK: 86 12 34 56  	bxh	%r1, %r2, 1110(%r3)
      .insn rs_a,0x86000000,%r1,1110(%r3),%r2

#CHECK: bd 12 34 56  	clm	%r1, 2, 1110(%r3)
      .insn rs_b,0xbd000000,%r1,1110(%r3),2

#CHECK: eb 10 23 45 00 c0    	tp	837(2,%r2)
      .insn rsl_a,0xeb00000000c0,837(2,%r2)

#CHECK: eb 12 34 56 78 44    	bxhg	%r1, %r2, 492630(%r3)
      .insn rsy_a,0xeb0000000044,%r1,492630(%r3),%r2

#CHECK: 5a 12 34 56  	a	%r1, 1110(%r2,%r3)
      .insn rx_a,0x5a000000,%r1,1110(%r2,%r3)

#CHECK: 47 12 34 56  	bo	1110(%r2,%r3)
      .insn rx_b,0x47000000,1,1110(%r2,%r3)

#CHECK: e3 12 34 56 78 16    	llgf	%r1, 492630(%r2,%r3)
      .insn rxy_a,0xe30000000016,%r1,492630(%r2,%r3)

#CHECK: e3 12 34 56 78 36     pfd 1, 492630(%r2,%r3)
      .insn rxy_b,0xe30000000036,1,492630(%r2,%r3)

#CHECK: d5 12 34 56 78 90     clc 1110(19,%r3), 2192(%r7)
      .insn ss_a,0xd50000000000,1110(19,%r3),2192(%r7)

#CHECK: f1 12 34 56 78 90      mvo 1110(2,%r3), 2192(3,%r7)
      .insn ss_b,0xf10000000000,1110(2,%r3),2192(3,%r7)

#CHECK: f0 12 34 56 78 90      srp 1110(2,%r3), 2192(%r7), 2
      .insn ss_c,0xf00000000000,1110(2,%r3),2192(%r7),2

#CHECK: da 12 34 56 78 90      mvcp 1110(%r1,%r3), 2192(%r7), %r2
      .insn ss_d,0xda0000000000,1110(%r1,%r3),2192(%r7),%r2

#CHECK: ef 12 34 56 78 90      lmd %r1, %r2, 1110(%r3), 2192(%r7)
      .insn ss_e,0xef0000000000,%r1,1110(%r3),%r2,2192(%r7)

#CHECK: e9 12 34 56 78 90      pka 1110(%r3), 2192(19,%r7)
      .insn ss_f,0xe90000000000,1110(%r3),2192(19,%r7)
