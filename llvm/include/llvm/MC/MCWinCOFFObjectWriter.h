//===-- llvm/MC/MCWinCOFFObjectWriter.h - Win COFF Object Writer *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINCOFFOBJECTWRITER_H
#define LLVM_MC_MCWINCOFFOBJECTWRITER_H

namespace llvm {
  class MCWinCOFFObjectTargetWriter {
    const unsigned Machine;

  protected:
    MCWinCOFFObjectTargetWriter(unsigned Machine_);

  public:
    virtual ~MCWinCOFFObjectTargetWriter() {}

    unsigned getMachine() const { return Machine; }
    virtual unsigned getRelocType(unsigned FixupKind) const = 0;
  };

  /// \brief Construct a new Win COFF writer instance.
  ///
  /// \param MOTW - The target specific WinCOFF writer subclass.
  /// \param OS - The stream to write to.
  /// \returns The constructed object writer.
  MCObjectWriter *createWinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW,
                                            raw_ostream &OS);
} // End llvm namespace

#endif
