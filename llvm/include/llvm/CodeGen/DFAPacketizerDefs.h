//=- llvm/CodeGen/DFAPacketizerDefs.h - DFA Packetizer for VLIW ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Common definitions used by TableGen and the DFAPacketizer in CodeGen.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DFAPACKETIZERDEFS_H
#define LLVM_CODEGEN_DFAPACKETIZERDEFS_H

#include <vector>

namespace llvm {

// DFA_MAX_RESTERMS * DFA_MAX_RESOURCES must fit within sizeof DFAInput.
// This is verified in DFAPacketizer.cpp:DFAPacketizer::DFAPacketizer.
//
// e.g. terms x resource bit combinations that fit in uint32_t:
//      4 terms x 8  bits = 32 bits
//      3 terms x 10 bits = 30 bits
//      2 terms x 16 bits = 32 bits
//
// e.g. terms x resource bit combinations that fit in uint64_t:
//      8 terms x 8  bits = 64 bits
//      7 terms x 9  bits = 63 bits
//      6 terms x 10 bits = 60 bits
//      5 terms x 12 bits = 60 bits
//      4 terms x 16 bits = 64 bits <--- current
//      3 terms x 21 bits = 63 bits
//      2 terms x 32 bits = 64 bits
//
#define DFA_MAX_RESTERMS        4   // The max # of AND'ed resource terms.
#define DFA_MAX_RESOURCES       16  // The max # of resource bits in one term.

typedef uint64_t                DFAInput;
typedef int64_t                 DFAStateInput;
#define DFA_TBLTYPE             "int64_t" // For generating DFAStateInputTable.

namespace {
  DFAInput addDFAFuncUnits(DFAInput Inp, unsigned FuncUnits) {
    return (Inp << DFA_MAX_RESOURCES) | FuncUnits;
  }

  /// Return the DFAInput for an instruction class input vector.
  /// This function is used in both DFAPacketizer.cpp and in
  /// DFAPacketizerEmitter.cpp.
  DFAInput getDFAInsnInput(const std::vector<unsigned> &InsnClass) {
    DFAInput InsnInput = 0;
    assert ((InsnClass.size() <= DFA_MAX_RESTERMS) &&
            "Exceeded maximum number of DFA terms");
    for (auto U : InsnClass)
      InsnInput = addDFAFuncUnits(InsnInput, U);
    return InsnInput;
  }
}

}

#endif
