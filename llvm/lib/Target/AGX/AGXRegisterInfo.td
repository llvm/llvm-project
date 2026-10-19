//===-- AGXRegisterInfo.td - AGX register definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGX register file is addressed in 16-bit units ("half registers").
// A 32-bit register rN is the pair of halves rNl (low) and rNh (high), and a
// 64-bit register is a pair of 32-bit registers. The encoding value of every
// register is its first half register index; instructions carry the access
// size separately.
//
// General purpose registers (r) are per thread. Uniform registers (u) are
// shared by the SIMD-group, read-only for ALU instructions, and preloaded by
// the driver or written with uniform_store.
//
//===----------------------------------------------------------------------===//

class AGXReg<string n, bits<16> enc, list<Register> subregs = []>
    : Register<n> {
  let Namespace = "AGX";
  let HWEncoding = enc;
  let SubRegs = subregs;
}

let Namespace = "AGX" in {
def lo16 : SubRegIndex<16, 0>;
def hi16 : SubRegIndex<16, 16>;
def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;
def sub3 : SubRegIndex<32, 96>;
def sub0_sub1 : SubRegIndex<64, 0>;
def sub1_sub2 : SubRegIndex<64, 32>;
def sub2_sub3 : SubRegIndex<64, 64>;
def sub0_sub1_sub2 : SubRegIndex<96, 0>;
def sub1_sub2_sub3 : SubRegIndex<96, 32>;
}

//===----------------------------------------------------------------------===//
// General purpose registers
//===----------------------------------------------------------------------===//

foreach Index = 0...127 in {
  def R#Index#L : AGXReg<"r"#Index#"l", !mul(Index, 2)>;
  def R#Index#H : AGXReg<"r"#Index#"h", !add(!mul(Index, 2), 1)>;

  let SubRegIndices = [lo16, hi16], CoveredBySubRegs = 1 in
  def R#Index : AGXReg<"r"#Index, !mul(Index, 2),
                       [!cast<Register>("R"#Index#"L"),
                        !cast<Register>("R"#Index#"H")]>;
}

//===----------------------------------------------------------------------===//
// Uniform registers
//===----------------------------------------------------------------------===//

foreach Index = 0...255 in {
  def U#Index#L : AGXReg<"u"#Index#"l", !mul(Index, 2)>;
  def U#Index#H : AGXReg<"u"#Index#"h", !add(!mul(Index, 2), 1)>;

  let SubRegIndices = [lo16, hi16], CoveredBySubRegs = 1 in
  def U#Index : AGXReg<"u"#Index, !mul(Index, 2),
                       [!cast<Register>("U"#Index#"L"),
                        !cast<Register>("U"#Index#"H")]>;
}

//===----------------------------------------------------------------------===//
// Register classes
//===----------------------------------------------------------------------===//

def GPR16 : RegisterClass<"AGX", [i16, f16, bf16], 16,
                          (interleave (sequence "R%uL", 0, 127),
                                      (sequence "R%uH", 0, 127))>;

def GPR32 : RegisterClass<"AGX", [i32, f32, v2i16, v2f16, v2bf16], 32,
                          (sequence "R%u", 0, 127)>;

def UGPR16 : RegisterClass<"AGX", [i16, f16, bf16], 16,
                           (interleave (sequence "U%uL", 0, 255),
                                       (sequence "U%uH", 0, 255))> {
  let isAllocatable = 0;
}

def UGPR32 : RegisterClass<"AGX", [i32, f32, v2i16, v2f16, v2bf16], 32,
                           (sequence "U%u", 0, 255)> {
  let isAllocatable = 0;
}

// Consecutive register tuples. Vector memory operations accept any start
// register; 64-bit scalar ALU operands require an even first register.
def GPR32x2Tuples : RegisterTuples<[sub0, sub1],
                                   [(sequence "R%u", 0, 126),
                                    (sequence "R%u", 1, 127)]>;
def GPR32x3Tuples : RegisterTuples<[sub0, sub1, sub2],
                                   [(sequence "R%u", 0, 125),
                                    (sequence "R%u", 1, 126),
                                    (sequence "R%u", 2, 127)]>;
def GPR32x4Tuples : RegisterTuples<[sub0, sub1, sub2, sub3],
                                   [(sequence "R%u", 0, 124),
                                    (sequence "R%u", 1, 125),
                                    (sequence "R%u", 2, 126),
                                    (sequence "R%u", 3, 127)]>;
def UGPR32x2Tuples : RegisterTuples<[sub0, sub1],
                                    [(sequence "U%u", 0, 254),
                                     (sequence "U%u", 1, 255)]>;

def GPR32x2 : RegisterClass<"AGX", [i64, f64, v2i32, v2f32, v4i16, v4f16],
                            32, (add GPR32x2Tuples)>;
def GPR64 : RegisterClass<"AGX", [i64, f64, v2i32, v2f32, v4i16, v4f16],
                          64, (decimate GPR32x2Tuples, 2)>;
def GPR96 : RegisterClass<"AGX", [v3i32, v3f32], 32, (add GPR32x3Tuples)>;
def GPR128 : RegisterClass<"AGX", [v4i32, v4f32, v2i64, v2f64], 32,
                           (add GPR32x4Tuples)>;

def UGPR64 : RegisterClass<"AGX", [i64, f64, v2i32, v2f32], 32,
                           (add UGPR32x2Tuples)> {
  let isAllocatable = 0;
}

// ALU source classes: general purpose registers or uniform registers.
def SRC16 : RegisterClass<"AGX", [i16, f16, bf16], 16, (add GPR16, UGPR16)> {
  let isAllocatable = 0;
}
def SRC32 : RegisterClass<"AGX", [i32, f32, v2i16, v2f16, v2bf16], 32,
                          (add GPR32, UGPR32)> {
  let isAllocatable = 0;
}
// 64-bit memory base addresses: general purpose or uniform register pairs.
def BASE64 : RegisterClass<"AGX", [i64], 32, (add GPR32x2, UGPR64)> {
  let isAllocatable = 0;
}
