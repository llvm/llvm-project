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
public:
  HexagonMCShuffler(MCContext &Context, bool Fatal, MCInstrInfo const &MCII,
                    MCSubtargetInfo const &STI, MCInst &MCB)
      : HexagonShuffler(Context, Fatal, MCII, STI) {
    init(MCB);
  };
  HexagonMCShuffler(MCContext &Context, bool Fatal, MCInstrInfo const &MCII,
                    MCSubtargetInfo const &STI, MCInst &MCB,
                    MCInst const &AddMI, bool InsertAtFront)
      : HexagonShuffler(Context, Fatal, MCII, STI) {
    init(MCB, AddMI, InsertAtFront);
  };

  // Copy reordered bundle to another.
  void copyTo(MCInst &MCB);
  // Reorder and copy result to another.
  bool reshuffleTo(MCInst &MCB);

private:
  void init(MCInst &MCB);
  void init(MCInst &MCB, MCInst const &AddMI, bool InsertAtFront);
};

// Invocation of the shuffler.
bool HexagonMCShuffle(MCContext &Context, bool Fatal, MCInstrInfo const &MCII,
                      MCSubtargetInfo const &STI, MCInst &);
bool HexagonMCShuffle(MCContext &Context, MCInstrInfo const &MCII,
                      MCSubtargetInfo const &STI, MCInst &, MCInst const &,
                      int);
bool HexagonMCShuffle(MCContext &Context, MCInstrInfo const &MCII,
                      MCSubtargetInfo const &STI, MCInst &,
                      SmallVector<DuplexCandidate, 8>);
} // namespace llvm

#endif // HEXAGONMCSHUFFLER_H
