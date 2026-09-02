// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common \
// RUN:   -DOVERLAP %s 2>&1 | FileCheck %s --check-prefix=OVERLAP

// Check that registers spanning consecutive members of a register sequence are
// gathered into blocks, so that one enumerator reserves them all and none of
// them needs a name of its own, while tuples that hold registers that do not
// follow one another keep being named after every sub-register they contain.

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;
def sub3 : SubRegIndex<32, 96>;

defvar Indices = !range(0, 8);

foreach Index = Indices in {
  def R#Index : Register <"r"#Index>;
}

// The registers follow one another, which the tuples below rely on. Note that
// this says nothing about the register classes that hold them.
def : RegisterSequence<!foreach(i, Indices, !cast<Register>("R"#i))>;

// Registers of their own, belonging to no sequence.
def RA : Register<"ra">;
def RB : Register<"rb">;
def RC : Register<"rc">;
def RD : Register<"rd">;

def GPR32 : RegisterClass<"TestTarget", [i32], 32, (add (sequence "R%u", 0, 7))>;
def OtherGPR32 : RegisterClass<"TestTarget", [i32], 32, (add RA, RB, RC, RD)>;

// Pairs of consecutive registers, so named after the first of them and their
// width. Supplying asm names shows they are independent of the record names.
def R64 : RegisterTuples<[sub0, sub1],
                         [(decimate (shl GPR32, 0), 2),
                          (decimate (shl GPR32, 1), 2)],
                         ["r[0:1]", "r[2:3]", "r[4:5]", "r[6:7]"]>;

// Quads of consecutive registers. The width tells them from the pairs above,
// even though they start at the same registers.
def R128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                          [(decimate (shl GPR32, 0), 4),
                           (decimate (shl GPR32, 1), 4),
                           (decimate (shl GPR32, 2), 4),
                           (decimate (shl GPR32, 3), 4)]>;

// Spaced registers, such as R0 and R2, do not follow one another, so a name
// that only mentions the first of them would not describe them.
def R64Spaced : RegisterTuples<[sub0, sub1],
                               [(decimate (shl GPR32, 0), 2),
                                (decimate (shl GPR32, 2), 2)]>;

// Registers that belong to no sequence at all.
def RX64 : RegisterTuples<[sub0, sub1],
                          [(decimate (shl OtherGPR32, 0), 2),
                           (decimate (shl OtherGPR32, 1), 2)]>;

// A second sequence, whose last two registers follow one another the other way
// round. Registers spanning them are still spanning consecutive members, so
// they are still named after the first of them, but their sub-registers do not
// sit where the ones before them do.
foreach Index = Indices in {
  def T#Index : Register <"t"#Index>;
}

def : RegisterSequence<[T0, T1, T2, T3, T4, T5, T7, T6]>;

def TGPR32 : RegisterClass<"TestTarget", [i32], 32, (add (sequence "T%u", 0, 7))>;

// Pairs of consecutive members, the last of them spanning T7 and T6 in that
// order. Up to that pair each sub-register sits two registers further along
// than in the pair before; that pair breaks the pattern.
def T64 : RegisterTuples<[sub0, sub1],
                         [(add T0, T2, T4, T7),
                          (add T1, T3, T5, T6)]>;

def GPR64 : RegisterClass<"TestTarget", [v2i32], 64, (add R64)>;
def GPR128 : RegisterClass<"TestTarget", [v4i32], 128, (add R128)>;
def GPR64Spaced : RegisterClass<"TestTarget", [v2i32], 64, (add R64Spaced)>;
def OtherGPR64 : RegisterClass<"TestTarget", [v2i32], 64, (add RX64)>;
def TGPR64 : RegisterClass<"TestTarget", [v2i32], 64, (add T64)>;

#ifdef OVERLAP
// A register sits in one place of one sequence, so it may belong to no more
// than one of them.
def : RegisterSequence<[R2, R5]>;

// OVERLAP: error: Register 'R2' already belongs to a sequence.
#endif

// The four pairs of consecutive registers follow one another in the register
// enumeration, so a single enumerator reserves them all.
// CHECK:      R_64_SEQ = {{[0-9]+}}, // 4 registers.

// Spaced registers do not follow one another, so they keep the longer names.
// CHECK-NEXT: R0_R2 = {{[0-9]+}},
// CHECK-NEXT: R2_R4 = {{[0-9]+}},
// CHECK-NEXT: R4_R6 = {{[0-9]+}},

// Quads of consecutive registers form a block of their own, told from the
// pairs above by their width.
// CHECK-NEXT: R_128_SEQ = {{[0-9]+}}, // 2 registers.

// Registers that belong to no sequence keep the longer names as well.
// CHECK-NEXT: RA_RB = {{[0-9]+}},
// CHECK-NEXT: RC_RD = {{[0-9]+}},

// A block goes as far as its registers keep to the pattern and no further. The
// first three pairs of the second sequence have their sub-registers two
// registers further along than the pair before, so the first of them describes
// them all. T7_64 spans T7 and T6, whose sub-registers sit the other way
// round, so it merely ends the block rather than denying one to the pairs
// before it, and keeps a name and an enumerator of its own.
// CHECK-NEXT: T_64_SEQ = {{[0-9]+}}, // 3 registers.
// CHECK-NEXT: T7_64 = {{[0-9]+}},

// The blocks themselves, through which their registers are reached. The step
// is the member-index distance between the members that adjacent registers
// begin at: every second member begins a pair, every fourth one a quad.
// CHECK:      constexpr MCRegisterSequenceBlock R_64 = {R_64_SEQ, 4, 2};
// CHECK-NEXT: constexpr MCRegisterSequenceBlock R_128 = {R_128_SEQ, 2, 4};
// CHECK-NEXT: constexpr MCRegisterSequenceBlock T_64 = {T_64_SEQ, 3, 2};

// What each register of a block has where the first of them has a
// sub-register, said as how far along the one is from the other, so that the
// registers after the first need no sub-register lists of their own. A quad
// spans two pairs and four plain registers, and the pairs move half as far as
// the plain registers do, there being two of the latter to each of the former.
// CHECK:      extern const int16_t TestTargetRegDiffLists[]
// CHECK:      /* 46 */ 4, 4, 4, 4, 2, 2, 2, 0,

// The registers of a block still go by the shorter names, which is what a
// register of a block prints as. Their asm names, given as r[0:1] and so on
// above, are a separate matter and are emitted elsewhere.
// CHECK: TestTargetRegStrings[]
// CHECK: "R0_64\000"
// CHECK: "R0_128\000"

// Where the registers containing them are is said for the block as a whole, so
// no register of one lists them. Their SuperRegs fields, the third of each
// descriptor, point at the empty list at offset 7 rather than naming registers
// that have nothing to do with them.
//
// The first two pairs. The first lists its sub-registers, being the first
// register of its block; the second does not, being described by that one and
// the distances above.
// CHECK:      extern const MCRegisterDesc TestTargetRegDesc[]
// CHECK:      { {{[0-9]+}}, 20, 7, 0,
// CHECK-NEXT: { {{[0-9]+}}, 7, 7, 0,

// The series of register that the registers of a block are contained by, said
// as where one sits with respect to the register it contains: the register it
// begins at, how many members before that register it begins, how many members
// apart the registers of the series are, how many of them there are, and how
// far one is from the next.
//
// A quad holds two pairs: the pair beginning where the quad does, and the pair
// beginning two members later. So the pairs of the first sequence have two
// series, the same quads reached from two distances back, and a pair reads
// both and takes the one that names a register for it. The quads themselves
// are contained by nothing, and neither are the pairs of the second
// sequence.
// CHECK:      extern const MCSeqSuperRegSeries TestTargetSeqSuperRegSeries[]
// CHECK-NEXT: { R_128_SEQ, 0, 4, 2, 1 },
// CHECK-NEXT: { R_128_SEQ, 2, 4, 2, 1 },

// The blocks as the register info sees them: the first register, how many
// there are, how many members apart they begin, where the distances above
// start, and which of the series above are theirs.
// CHECK:      extern const MCSeqBlockDesc TestTargetSeqBlocks[]
// CHECK-NEXT: { R_64_SEQ, 4, 2, {{[0-9]+}}, 0, 2 },
// CHECK-NEXT: { R_128_SEQ, 2, 4, 46, 2, 0 },
// CHECK-NEXT: { T_64_SEQ, 3, 2, {{[0-9]+}}, 2, 0 },

// Having no enumerators of their own, registers of a block are referred to by
// the one that reserves the block and how far into it they sit. Here, the four
// pairs among the members of GPR64.
// CHECK: /* {{[0-9]+}} */ R_64_SEQ, R_64_SEQ + 1, R_64_SEQ + 2, R_64_SEQ + 3,
