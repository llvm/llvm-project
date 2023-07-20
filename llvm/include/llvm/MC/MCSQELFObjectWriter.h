#ifndef LLVM_MC_MCSQELFOBJECTWRITER_H
#define LLVM_MC_MCSQELFOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {

class MCSQELFObjectTargetWriter : public MCObjectTargetWriter {

protected:
  explicit MCSQELFObjectTargetWriter();

public:
  virtual ~MCSQELFObjectTargetWriter();

  Triple::ObjectFormatType getFormat() const override { return Triple::SQELF; }

  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::SQELF;
  }
};

/// Construct a new SQELF writer instance.
///
/// \param MOTW - The target specific Wasm writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter>
createSQELFObjectWriter(std::unique_ptr<MCSQELFObjectTargetWriter> MOTW,
                        raw_pwrite_stream &OS);

} // namespace llvm

#endif // LLVM_MC_MCSQELFOBJECTWRITER_H
