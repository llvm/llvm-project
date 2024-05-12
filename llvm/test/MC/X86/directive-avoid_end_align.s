# RUN: llvm-mc -triple=x86_64 -filetype=obj %s | llvm-objdump --no-show-raw-insn -d - | FileCheck %s
# RUN: not llvm-mc -triple=x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# avoid_end_align has no effect since test doesn't end at alignment boundary:
.avoid_end_align 64
# CHECK-NOT: nop
  testl %eax, %eax
# CHECK: testl %eax, %eax
  je  .LBB0

.fill 58, 1, 0x00
# NeverAlign followed by MCDataFragment:
# avoid_end_align inserts nop because `test` would end at alignment boundary:
.avoid_end_align 64
# CHECK: 			3e: nop
  testl %eax, %eax
# CHECK-NEXT: 3f: testl %eax, %eax
  je  .LBB0
# CHECK-NEXT: 41: je
.LBB0:
  retq

.p2align 6
.L0:
.nops 57
  int3
# NeverAlign followed by RelaxableFragment:
.avoid_end_align 64
# CHECK: 			ba: nop
  cmpl $(.L1-.L0), %eax
# CHECK-NEXT: bb: cmpl
  je  .L0
# CHECK-NEXT: c1: je
.nops 65
.L1:

###############################################################################
# Experiment A:
# Check that NeverAlign doesn't introduce infinite loops in layout.
# Control:
# 1. NeverAlign fragment is not added,
# 2. Short formats of cmp and jcc are used (3 and 2 bytes respectively),
# 3. cmp and jcc are placed such that to be split by 64B alignment boundary.
# 4. jcc would be relaxed to a longer format if at least one byte is added
#    between .L10 and je itself, e.g. by adding a NeverAlign padding byte,
#    or relaxing cmp instruction.
# 5. cmp would be relaxed to a longer format if at least one byte is added
#    between .L11 and .L12, e.g. due to relaxing jcc instruction.
.p2align 6
# CHECK:      140: int3
.fill 2, 1, 0xcc
.L10:
.nops 122
  int3
# CHECK:      1bc: int3
# no avoid_end_align here
# CHECK-NOT:  nop
  cmp $(.L12-.L11), %eax
# CHECK:      1bd: cmpl
.L11:
  je  .L10
# CHECK-NEXT: 1c0: je
.nops 125
.L12:

# Experiment:
# Same setup as control, except NeverAlign fragment is added before cmp.
# Expected effect:
# 1. NeverAlign pads cmp+jcc by one byte since cmp and jcc are split by a 64B
#    alignment boundary,
# 2. This extra byte forces jcc relaxation to a longer format (Control rule #4),
# 3. This results in an cmp relaxation (Control rule #5),
# 4. Which in turn makes NeverAlign fragment unnecessary as cmp and jcc
#    are no longer split by an alignment boundary (cmp crosses the boundary).
# 5. NeverAlign padding is removed.
# 6. cmp and jcc instruction remain in relaxed form.
# 7. Relaxation converges, layout succeeds.
.p2align 6
# CHECK:      240: int3
.fill 2, 1, 0xcc
.L20:
.nops 122
  int3
# CHECK:      2bc: int3
.avoid_end_align 64
# CHECK-NOT: 	nop
  cmp $(.L22-.L21), %eax
# CHECK-NEXT: 2bd: cmpl
.L21:
  je  .L20
# CHECK-NEXT: 2c3: je
.nops 125
.L22:

###############################################################################
# Experiment B: similar to exp A, but we check that once NeverAlign padding is
# removed from the layout (exp A, experiment step 5), the increased distance
# between the symbols L33 and L34 triggers the relaxation of instruction at
# label L32.
#
# Control 1: using a one-byte instruction at L33 (site of NeverAlign) leads to
# steps 2-3 of exp A, experiment:
# 2. This extra byte forces jcc relaxation to a longer format (Control rule #4),
# 3. This results in an cmp relaxation (Control rule #5),
# => short cmp under L32
.p2align 6
# CHECK:      380: int3
.fill 2, 1, 0xcc
.L30:
.nops 122
  int3
# CHECK:      3fc: int3
  hlt
#.avoid_end_align 64
.L33:
  cmp $(.L32-.L31), %eax
# CHECK:      3fe: cmpl
.L31:
  je  .L30
# CHECK-NEXT: 404: je
.nops 114
.p2align 1
  int3
  int3
# CHECK:      47c: int3
.L34:
.nops 9
.L32:
  cmp $(.L33-.L34), %eax
# CHECK:      487: cmp
# note that the size of cmp is 48a-487 == 3 bytes (distance is exactly -128)
  int3
# CHECK-NEXT: 48a: int3

# Control 2: leaving out a byte at L43 (site of NeverAlign), plus
# relaxed jcc and cmp leads to a relaxed cmp under L42 (-129 as cmp's immediate)
.p2align 6
# CHECK:      4c0: int3
.fill 2, 1, 0xcc
.L40:
.nops 122
  int3
# CHECK:      53c: int3
#  int3
#.avoid_end_align 64
.L43:
  cmp $(.L42-.L41+0x100), %eax
# CHECK:      53d: cmpl
.L41:
  je  .L40+0x100
# CHECK-NEXT: 543: je
.nops 114
.p2align 1
  int3
  int3
# CHECK:      5bc: int3
.L44:
.nops 9
.L42:
  cmp $(.L43-.L44), %eax
# CHECK:      5c7: cmp
# note that the size of cmp is 5cd-5c7 == 6 bytes (distance is exactly -129)
  int3
# CHECK-NEXT: 5cd: int3

# Experiment
# Checking if removing NeverAlign padding at L53 as a result of alignment and
# relaxation of cmp and jcc following it (see exp A), thus reproducing the case
# in Control 2 (getting a relaxed cmp under L52), is handled correctly.
.p2align 6
# CHECK:      600: int3
.fill 2, 1, 0xcc
.L50:
.nops 122
  int3
# CHECK:      67c: int3
.avoid_end_align 64
.L53:
# CHECK-NOT: 	nop
  cmp $(.L52-.L51), %eax
# CHECK-NEXT: 67d: cmpl
.L51:
  je  .L50
# CHECK-NEXT: 683: je
.nops 114
.p2align 1
  int3
  int3
# CHECK:      6fc: int3
.L54:
.nops 9
.L52:
  cmp $(.L53-.L54), %eax
# CHECK:      707: cmp
# note that the size of cmp is 70d-707 == 6 bytes (distance is exactly -129)
  int3
# CHECK-NEXT: 70d: int3

.ifdef ERR
# ERR: {{.*}}.s:[[#@LINE+1]]:17: error: unknown token in expression
.avoid_end_align
# ERR: {{.*}}.s:[[#@LINE+1]]:18: error: expected absolute expression
.avoid_end_align x
# ERR: {{.*}}.s:[[#@LINE+1]]:18: error: expected a positive alignment
.avoid_end_align 0
# ERR: {{.*}}.s:[[#@LINE+1]]:20: error: unexpected token in directive
.avoid_end_align 64, 0
.endif
