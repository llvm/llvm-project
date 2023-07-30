#ifndef LLVM_MC_MCSQELFOBJECTWRITER_H
#define LLVM_MC_MCSQELFOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {

class MCSQELFObjectTargetWriter : public MCObjectTargetWriter {

public:
  explicit MCSQELFObjectTargetWriter(bool Is64Bit_, uint8_t OSABI_,
                                     uint16_t EMachine_,
                                     uint8_t ABIVersion_ = 0);
  virtual ~MCSQELFObjectTargetWriter();

  Triple::ObjectFormatType getFormat() const override { return Triple::SQELF; }

  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::SQELF;
  }

  /// \name Accessors
  /// @{
  uint8_t getOSABI() const { return OSABI; }
  uint8_t getABIVersion() const { return ABIVersion; }
  uint16_t getEMachine() const { return EMachine; }
  bool is64Bit() const { return Is64Bit; }
  /// @}
private:
  // TODO(fzakaria): for now we are copying very similar the fields
  // of MCELFObjectTargetWriter but they might be different later
  const uint8_t OSABI;
  const uint8_t ABIVersion;
  const uint16_t EMachine;
  const unsigned Is64Bit : 1;
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
