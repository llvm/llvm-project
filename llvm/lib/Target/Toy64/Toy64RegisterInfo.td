//===-- Toy64RegisterInfo.td - Toy64 Register Definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Toy64 register file.
//
// Toy64 has 32 64-bit integer registers:
//
//   x0  - hardwired zero
//   x1  - return address
//   x2  - stack pointer
//   x8  - frame pointer when needed
//   x10-x17 - argument registers
//
//===----------------------------------------------------------------------===//

// Base class for every Toy64 physical register.
//
// Enc is the 5-bit value that will eventually be placed in an instruction's
// register field.
//
// For example:
//
//     X10 -> encoding 10 -> binary 01010
//
class Toy64Reg<bits<16> Enc, string Name> : Register<Name> {
  let Namespace = "Toy64";
  let HWEncoding = Enc;
}

//===----------------------------------------------------------------------===//
// General-purpose registers
//===----------------------------------------------------------------------===//

def X0  : Toy64Reg<0,  "x0">;
def X1  : Toy64Reg<1,  "x1">;
def X2  : Toy64Reg<2,  "x2">;
def X3  : Toy64Reg<3,  "x3">;
def X4  : Toy64Reg<4,  "x4">;
def X5  : Toy64Reg<5,  "x5">;
def X6  : Toy64Reg<6,  "x6">;
def X7  : Toy64Reg<7,  "x7">;
def X8  : Toy64Reg<8,  "x8">;
def X9  : Toy64Reg<9,  "x9">;
def X10 : Toy64Reg<10, "x10">;
def X11 : Toy64Reg<11, "x11">;
def X12 : Toy64Reg<12, "x12">;
def X13 : Toy64Reg<13, "x13">;
def X14 : Toy64Reg<14, "x14">;
def X15 : Toy64Reg<15, "x15">;
def X16 : Toy64Reg<16, "x16">;
def X17 : Toy64Reg<17, "x17">;
def X18 : Toy64Reg<18, "x18">;
def X19 : Toy64Reg<19, "x19">;
def X20 : Toy64Reg<20, "x20">;
def X21 : Toy64Reg<21, "x21">;
def X22 : Toy64Reg<22, "x22">;
def X23 : Toy64Reg<23, "x23">;
def X24 : Toy64Reg<24, "x24">;
def X25 : Toy64Reg<25, "x25">;
def X26 : Toy64Reg<26, "x26">;
def X27 : Toy64Reg<27, "x27">;
def X28 : Toy64Reg<28, "x28">;
def X29 : Toy64Reg<29, "x29">;
def X30 : Toy64Reg<30, "x30">;
def X31 : Toy64Reg<31, "x31">;


// foreach i = 0-31 in {
//     def X#i : Toy64Reg<i, "x"#i>;
// }

//===----------------------------------------------------------------------===//
// Register classes
//===----------------------------------------------------------------------===//

// GPR is the general-purpose integer register class.
//
// Every member can hold an i64 value.
//
// Later, Toy64RegisterInfo.cpp will mark special registers such as X0 and X2
// as reserved so that LLVM's register allocator cannot assign arbitrary
// virtual registers to them.
def GPR : RegisterClass<"Toy64",[i64],64,
    (add
      X0,  X1,  X2,  X3,
      X4,  X5,  X6,  X7,
      X8,  X9,  X10, X11,
      X12, X13, X14, X15,
      X16, X17, X18, X19,
      X20, X21, X22, X23,
      X24, X25, X26, X27,
      X28, X29, X30, X31
    )
>;