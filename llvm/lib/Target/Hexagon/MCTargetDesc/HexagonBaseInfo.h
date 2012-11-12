//===-- HexagonBaseInfo.h - Top level definitions for Hexagon --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Hexagon target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBASEINFO_H
#define HEXAGONBASEINFO_H

namespace llvm {

/// HexagonII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace HexagonII {
  // *** The code below must match HexagonInstrFormat*.td *** //

  // Insn types.
  // *** Must match HexagonInstrFormat*.td ***
  enum Type {
    TypePSEUDO = 0,
    TypeALU32  = 1,
    TypeCR     = 2,
    TypeJR     = 3,
    TypeJ      = 4,
    TypeLD     = 5,
    TypeST     = 6,
    TypeSYSTEM = 7,
    TypeXTYPE  = 8,
    TypeMEMOP  = 9,
    TypeNV     = 10,
    TypePREFIX = 30, // Such as extenders.
    TypeMARKER = 31  // Such as end of a HW loop.
  };



  // MCInstrDesc TSFlags
  // *** Must match HexagonInstrFormat*.td ***
  enum {
    // This 5-bit field describes the insn type.
    TypePos  = 0,
    TypeMask = 0x1f,

    // Solo instructions.
    SoloPos  = 5,
    SoloMask = 0x1,

    // Predicated instructions.
    PredicatedPos  = 6,
    PredicatedMask = 0x1
  };

  // *** The code above must match HexagonInstrFormat*.td *** //

} // End namespace HexagonII.

} // End namespace llvm.

#endif
