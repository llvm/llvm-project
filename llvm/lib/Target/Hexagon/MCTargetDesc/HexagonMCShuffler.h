//=-- HexagonMCShuffler.h ---------------------------------------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares the shuffling of insns inside a bundle according to the
// packet formation rules of the Hexagon ISA.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCSHUFFLER_H
#define HEXAGONMCSHUFFLER_H

#include "MCTargetDesc/HexagonShuffler.h"

namespace llvm {

class MCInst;

// Insn bundle shuffler.
class HexagonMCShuffler : public HexagonShuffler {
  bool immext_present;
  bool duplex_present;

public:
  HexagonMCShuffler(bool Fatal, MCInstrInfo const &MCII,
                    MCSubtargetInfo const &STI, MCInst &MCB)
      : HexagonShuffler(MCII, STI) {
    init(MCB);
  };
  HexagonMCShuffler(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                    MCInst &MCB, MCInst const &AddMI,
                    bool InsertAtFront)
      : HexagonShuffler(MCII, STI) {
    init(MCB, AddMI, InsertAtFront);
  };

  // Copy reordered bundle to another.
  void copyTo(MCInst &MCB);
  // Reorder and copy result to another.
  bool reshuffleTo(MCInst &MCB);

  bool immextPresent() const { return immext_present; };
  bool duplexPresent() const { return duplex_present; };

private:
  void init(MCInst &MCB);
  void init(MCInst &MCB, MCInst const &AddMI, bool InsertAtFront);
};

// Invocation of the shuffler.
bool HexagonMCShuffle(bool Fatal, MCInstrInfo const &MCII,
                      MCSubtargetInfo const &STI, MCInst &);
bool HexagonMCShuffle(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                      MCInst &, MCInst const &, int);
unsigned HexagonMCShuffle(MCInstrInfo const &MCII, MCSubtargetInfo const &STI,
                          MCContext &Context, MCInst &,
                          SmallVector<DuplexCandidate, 8>);
}

#endif // HEXAGONMCSHUFFLER_H
