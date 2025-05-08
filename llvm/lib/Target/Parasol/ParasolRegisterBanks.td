//===-- ParasolRegisterBanks.td - Describe the Parasol Banks -------*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Define the Parasol register banks used for GlobalISel.
///
//===----------------------------------------------------------------------===//

/// General Purpose Registers
def IRRegBank : RegisterBank<"IRRB", [IR]>;
def PRRegBank : RegisterBank<"PRRB", [PR]>;
def GPRRegBank : RegisterBank<"GPRB", [GPR]>;
