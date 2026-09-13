// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa -filetype=obj %s -o %t
// RUN: llvm-readobj --hex-dump=.data %t | FileCheck %s

// Both forward DAGs contain only 33 nodes but have over four billion paths.
// Evaluate shared subexpressions once, with a fresh state for every query.
// All maximum operands are negative; OR operands are arbitrary bitsets.
// CHECK:      Hex dump of section '.data':
// CHECK-NEXT: 0x00000000 f9ffffff ffff0000 feffffff 0a000000
// CHECK-NEXT: 0x00000010 3a000000 fdffffff 40210000 ffffffff
// CHECK-NEXT: 0x00000020 f7ffffff ffffffff 00000000 01000080
// CHECK-NEXT: 0x00000030 02000000 07000000 00000000 00000000
// CHECK-NEXT: 0x00000040 02000000 02000000 07000000 00000000
// CHECK-NEXT: 0x00000050 21000000 00000000 fcffffff ffffffff

.set .Lmax0, max(.Lmax1, .Lmax1, -9)
.set .Lmax1, max(.Lmax2, .Lmax2, -9)
.set .Lmax2, max(.Lmax3, .Lmax3, -9)
.set .Lmax3, max(.Lmax4, .Lmax4, -9)
.set .Lmax4, max(.Lmax5, .Lmax5, -9)
.set .Lmax5, max(.Lmax6, .Lmax6, -9)
.set .Lmax6, max(.Lmax7, .Lmax7, -9)
.set .Lmax7, max(.Lmax8, .Lmax8, -9)
.set .Lmax8, max(.Lmax9, .Lmax9, -9)
.set .Lmax9, max(.Lmax10, .Lmax10, -9)
.set .Lmax10, max(.Lmax11, .Lmax11, -9)
.set .Lmax11, max(.Lmax12, .Lmax12, -9)
.set .Lmax12, max(.Lmax13, .Lmax13, -9)
.set .Lmax13, max(.Lmax14, .Lmax14, -9)
.set .Lmax14, max(.Lmax15, .Lmax15, -9)
.set .Lmax15, max(.Lmax16, .Lmax16, -9)
.set .Lmax16, max(.Lmax17, .Lmax17, -9)
.set .Lmax17, max(.Lmax18, .Lmax18, -9)
.set .Lmax18, max(.Lmax19, .Lmax19, -9)
.set .Lmax19, max(.Lmax20, .Lmax20, -9)
.set .Lmax20, max(.Lmax21, .Lmax21, -9)
.set .Lmax21, max(.Lmax22, .Lmax22, -9)
.set .Lmax22, max(.Lmax23, .Lmax23, -9)
.set .Lmax23, max(.Lmax24, .Lmax24, -9)
.set .Lmax24, max(.Lmax25, .Lmax25, -9)
.set .Lmax25, max(.Lmax26, .Lmax26, -9)
.set .Lmax26, max(.Lmax27, .Lmax27, -9)
.set .Lmax27, max(.Lmax28, .Lmax28, -9)
.set .Lmax28, max(.Lmax29, .Lmax29, -9)
.set .Lmax29, max(.Lmax30, .Lmax30, -9)
.set .Lmax30, max(.Lmax31, .Lmax31, -9)
.set .Lmax31, max(.Lmax32, .Lmax32, -9)
.set .Lmax32, -7

.set .Lor0, or(.Lor1, .Lor1, 1)
.set .Lor1, or(.Lor2, .Lor2, 2)
.set .Lor2, or(.Lor3, .Lor3, 4)
.set .Lor3, or(.Lor4, .Lor4, 8)
.set .Lor4, or(.Lor5, .Lor5, 16)
.set .Lor5, or(.Lor6, .Lor6, 32)
.set .Lor6, or(.Lor7, .Lor7, 64)
.set .Lor7, or(.Lor8, .Lor8, 128)
.set .Lor8, or(.Lor9, .Lor9, 256)
.set .Lor9, or(.Lor10, .Lor10, 512)
.set .Lor10, or(.Lor11, .Lor11, 1024)
.set .Lor11, or(.Lor12, .Lor12, 2048)
.set .Lor12, or(.Lor13, .Lor13, 4096)
.set .Lor13, or(.Lor14, .Lor14, 8192)
.set .Lor14, or(.Lor15, .Lor15, 16384)
.set .Lor15, or(.Lor16, .Lor16, 32768)
.set .Lor16, or(.Lor17, .Lor17, 1)
.set .Lor17, or(.Lor18, .Lor18, 2)
.set .Lor18, or(.Lor19, .Lor19, 4)
.set .Lor19, or(.Lor20, .Lor20, 8)
.set .Lor20, or(.Lor21, .Lor21, 16)
.set .Lor21, or(.Lor22, .Lor22, 32)
.set .Lor22, or(.Lor23, .Lor23, 64)
.set .Lor23, or(.Lor24, .Lor24, 128)
.set .Lor24, or(.Lor25, .Lor25, 256)
.set .Lor25, or(.Lor26, .Lor26, 512)
.set .Lor26, or(.Lor27, .Lor27, 1024)
.set .Lor27, or(.Lor28, .Lor28, 2048)
.set .Lor28, or(.Lor29, .Lor29, 4096)
.set .Lor29, or(.Lor30, .Lor30, 8192)
.set .Lor30, or(.Lor31, .Lor31, 16384)
.set .Lor31, or(.Lor32, .Lor32, 32768)
.set .Lor32, 0

// Multiple aliases share the same expression. A completed shared value is
// distinct from a value that is still active in the current traversal.
.set .Lalias_a, .Lshared
.set .Lalias_b, .Lshared
.set .Lshared, max(.Llate_alias, -9)
.set .Laliases, max(.Lalias_a, .Lalias_b, .Lshared)
.set .Llate_alias, -2

// Preserve operation boundaries and the ordinary evaluation of binary leaves.
.set .Lmixed_max, max(or(.Lbits, 4), max(.Lmax_leaf, 3), .Lbinary + 2)
.set .Lbits, 2
.set .Lmax_leaf, 5
.set .Lbinary, 8
.set .Lmixed_or, or(max(.La, 4), or(.Lb, 2), .Lc + 1)
.set .La, 8
.set .Lb, 16
.set .Lc, 31

.data
.long .Lmax0, .Lor0, .Laliases, .Lmixed_max
.long .Lmixed_or, max(-9, -3, -2147483648), or(0x100, 0x40, 0x2000), max(-1)
.quad max(-9, -9223372036854775808)
.quad or(0x8000000000000000, 0x100000000)

// A later evaluation must observe a reassigned symbol's new value.
.set .Lmutable, 1
.long max(.Lmutable, 2)
.set .Lmutable, 7
.long max(.Lmutable, 2)
.long 0, 0

// An earlier expression retains its symbol version when the source spelling
// is reassigned. Following an alias must not switch to the latest named value.
.set .Ldelayed, max(.Lfuture, 2)
.set .Lfuture, 1
.long .Ldelayed
.set .Lfuture, 7
.long .Ldelayed
.long max(.Lfuture, 2), 0

// Private-stack expressions interleave addition and shared maxima. A separate
// state for each maximum would re-expand the same additive DAG exponentially.
.set .Ladd0, 1 + max(.Ladd1, .Ladd1)
.set .Ladd1, 1 + max(.Ladd2, .Ladd2)
.set .Ladd2, 1 + max(.Ladd3, .Ladd3)
.set .Ladd3, 1 + max(.Ladd4, .Ladd4)
.set .Ladd4, 1 + max(.Ladd5, .Ladd5)
.set .Ladd5, 1 + max(.Ladd6, .Ladd6)
.set .Ladd6, 1 + max(.Ladd7, .Ladd7)
.set .Ladd7, 1 + max(.Ladd8, .Ladd8)
.set .Ladd8, 1 + max(.Ladd9, .Ladd9)
.set .Ladd9, 1 + max(.Ladd10, .Ladd10)
.set .Ladd10, 1 + max(.Ladd11, .Ladd11)
.set .Ladd11, 1 + max(.Ladd12, .Ladd12)
.set .Ladd12, 1 + max(.Ladd13, .Ladd13)
.set .Ladd13, 1 + max(.Ladd14, .Ladd14)
.set .Ladd14, 1 + max(.Ladd15, .Ladd15)
.set .Ladd15, 1 + max(.Ladd16, .Ladd16)
.set .Ladd16, 1 + max(.Ladd17, .Ladd17)
.set .Ladd17, 1 + max(.Ladd18, .Ladd18)
.set .Ladd18, 1 + max(.Ladd19, .Ladd19)
.set .Ladd19, 1 + max(.Ladd20, .Ladd20)
.set .Ladd20, 1 + max(.Ladd21, .Ladd21)
.set .Ladd21, 1 + max(.Ladd22, .Ladd22)
.set .Ladd22, 1 + max(.Ladd23, .Ladd23)
.set .Ladd23, 1 + max(.Ladd24, .Ladd24)
.set .Ladd24, 1 + max(.Ladd25, .Ladd25)
.set .Ladd25, 1 + max(.Ladd26, .Ladd26)
.set .Ladd26, 1 + max(.Ladd27, .Ladd27)
.set .Ladd27, 1 + max(.Ladd28, .Ladd28)
.set .Ladd28, 1 + max(.Ladd29, .Ladd29)
.set .Ladd29, 1 + max(.Ladd30, .Ladd30)
.set .Ladd30, 1 + max(.Ladd31, .Ladd31)
.set .Ladd31, 1 + max(.Ladd32, .Ladd32)
.set .Ladd32, 1
.quad .Ladd0

// The cached absolute-addition path must preserve MC's 64-bit wrapping.
.set .Lwrapped, max(.Lsigned_max + 1, -4)
.set .Lsigned_max, 0x7fffffffffffffff
.quad .Lwrapped
