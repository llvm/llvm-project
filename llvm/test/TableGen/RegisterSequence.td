// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common \
// RUN:   -DOVERLAP %s 2>&1 | FileCheck %s --check-prefix=OVERLAP

// Check that tuples made of consecutive members of a register sequence are
// named after the first of those members and their width, while tuples that
// hold registers that do not follow one another keep being named after every
// sub-register they contain.

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

def GPR64 : RegisterClass<"TestTarget", [v2i32], 64, (add R64)>;
def GPR128 : RegisterClass<"TestTarget", [v4i32], 128, (add R128)>;
def GPR64Spaced : RegisterClass<"TestTarget", [v2i32], 64, (add R64Spaced)>;
def OtherGPR64 : RegisterClass<"TestTarget", [v2i32], 64, (add RX64)>;

#ifdef OVERLAP
// A register is followed by at most one register, so it may only be stated
// once which one that is. Note that the last member of a sequence is followed
// by nothing, so sequences may still be chained at their ends.
def : RegisterSequence<[R2, R5]>;

// OVERLAP: error: Register 'R2' is already followed by 'R3'.
#endif

// Consecutive registers are named after the first of them and the width.
// CHECK:      R0_64 = {{[0-9]+}},
// CHECK-NEXT: R2_64 = {{[0-9]+}},
// CHECK-NEXT: R4_64 = {{[0-9]+}},
// CHECK-NEXT: R6_64 = {{[0-9]+}},

// Spaced registers do not follow one another, so they keep the longer names.
// CHECK-NEXT: R0_R2 = {{[0-9]+}},
// CHECK-NEXT: R2_R4 = {{[0-9]+}},
// CHECK-NEXT: R4_R6 = {{[0-9]+}},

// Quads of consecutive registers, told from the pairs by their width.
// CHECK-NEXT: R0_128 = {{[0-9]+}},
// CHECK-NEXT: R4_128 = {{[0-9]+}},

// Registers that belong to no sequence keep the longer names as well.
// CHECK-NEXT: RA_RB = {{[0-9]+}},
// CHECK-NEXT: RC_RD = {{[0-9]+}},

// The names the registers are known by are the shorter ones. Their asm names,
// given as r[0:1] and so on above, are a separate matter and are emitted
// elsewhere.
// CHECK: TestTargetRegStrings[]
// CHECK: "R0_64\000"
// CHECK: "R0_128\000"
