# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32 -mattr=+soft-float 2>&1 |\
# RUN:   FileCheck %s --check-prefix=32
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips64 -mattr=+soft-float 2>&1 |\
# RUN:   FileCheck %s --check-prefix=64
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -mattr=+soft-float 2>&1 |\
# RUN:   FileCheck %s --check-prefix=R2
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r6 -mattr=+soft-float 2>&1 |\
# RUN:   FileCheck %s --check-prefix=R6

foo:
  dmfc1      $7, $f2
  # 64: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  dmtc1      $6, $f2
  # 64: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled

  ceil.l.d   $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  ceil.l.s   $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.d.l    $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.l.d    $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.l.s    $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.s.l    $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  floor.l.d  $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  floor.l.s  $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  ldxc1      $f2, $4($6)
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  luxc1      $f2, $4($6)
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  lwxc1      $f2, $4($6)
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mfhc1      $7, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  msub.s     $f2, $f2, $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mthc1      $7, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  nmadd.s    $f2, $f2, $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  nmsub.s    $f2, $f2, $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  round.l.s  $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sdxc1      $f2, $4($6)
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  suxc1      $f2, $4($6)
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  swxc1      $f2, $4($6)
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  trunc.l.d  $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  trunc.l.s  $f2, $f2
  # R2: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled

  bc1eqz     $f2, 123
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  bc1nez     $f2, 456
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  class.d    $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  class.s    $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.af.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.af.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.eq.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.eq.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.le.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.le.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.lt.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.lt.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.saf.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.saf.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.seq.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.seq.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sle.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sle.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.slt.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.slt.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sueq.d $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sueq.s $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sule.d $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sule.s $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sult.d $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sult.s $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sun.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.sun.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.ueq.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.ueq.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.ule.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.ule.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.ult.d  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.ult.s  $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.un.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cmp.un.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  maddf.d    $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  maddf.s    $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  max.d      $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  max.s      $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  maxa.d     $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  maxa.s     $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  min.d      $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  min.s      $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mina.d     $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mina.s     $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  msubf.d    $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  msubf.s    $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  rint.d     $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  rint.s     $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sel.d      $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sel.s      $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  seleqz.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  seleqz.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  selnez.d   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  selnez.s   $f2, $f2, $f2
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled

  abs.d      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  abs.s      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  add.d      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  add.s      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.eq.d     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.eq.s     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.f.d      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.f.s      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.le.d     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.le.s     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.lt.d     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.lt.s     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.nge.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.nge.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ngl.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ngl.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ngle.d   $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ngle.s   $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ngt.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ngt.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ole.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ole.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.olt.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.olt.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.seq.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.seq.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.sf.d     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.sf.s     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ueq.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ueq.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ule.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ule.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ult.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.ult.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.un.d     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  c.un.s     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  ceil.w.d   $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  ceil.w.s   $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.d.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.d.w    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.s.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.s.w    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.w.d    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  cvt.w.s    $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  div.d      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  div.s      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  floor.w.d  $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  floor.w.s  $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  ldc1       $f2, 16($7)
  # 32: :[[#@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  lwc1       $f2, 16($7)
  # 32: :[[#@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  madd.s     $f2, $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mfc1       $7, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mov.d      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mov.s      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movf.d     $f2, $f2, $fcc2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movf.s     $f2, $f2, $fcc5
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movn.d     $f2, $f2, $6
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movn.s     $f2, $f2, $6
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movt.d     $f2, $f2, $fcc0
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movt.s     $f2, $f2, $fcc1
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movz.d     $f2, $f2, $6
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  movz.s     $f2, $f2, $6
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mtc1       $7, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mul.d      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  mul.s      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  neg.d      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  neg.s      $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  round.w.d  $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  round.w.s  $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sdc1       $f2, 16($7)
  # 32: :[[#@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sqrt.d     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sqrt.s     $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sub.d      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sub.s      $f2, $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  swc1       $f2, 16($7)
  # 32: :[[#@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  trunc.w.d  $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  trunc.w.s  $f2, $f2
  # 32: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
