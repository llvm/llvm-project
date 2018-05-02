//===- llvm/MC/MCWinCOFFObjectWriter.h - Win COFF Object Writer -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINCOFFOBJECTWRITER_H
#define LLVM_MC_MCWINCOFFOBJECTWRITER_H

#include <memory>

namespace llvm {

class MCAsmBackend;
class MCContext;
class MCFixup;
class MCObjectWriter;
class MCValue;
class raw_pwrite_stream;

  class MCWinCOFFObjectTargetWriter {
    virtual void anchor();

    const unsigned Machine;

  protected:
    MCWinCOFFObjectTargetWriter(unsigned Machine_);

  public:
    virtual ~MCWinCOFFObjectTargetWriter() = default;

    unsigned getMachine() const { return Machine; }
    virtual unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                                  const MCFixup &Fixup, bool IsCrossSection,
                                  const MCAsmBackend &MAB) const = 0;
    virtual bool recordRelocation(const MCFixup &) const { return true; }
  };

  /// Construct a new Win COFF writer instance.
  ///
  /// \param MOTW - The target specific WinCOFF writer subclass.
  /// \param OS - The stream to write to.
  /// \returns The constructed object writer.
  std::unique_ptr<MCObjectWriter>
  createWinCOFFObjectWriter(std::unique_ptr<MCWinCOFFObjectTargetWriter> MOTW,
                            raw_pwrite_stream &OS);
} // end namespace llvm

#endif // LLVM_MC_MCWINCOFFOBJECTWRITER_H
