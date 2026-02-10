//===- llvm/MC/MCMachOCASWriter.h - Mach CAS Object Writer ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMACHOCASWRITER_H
#define LLVM_MC_MCMACHOCASWRITER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/MC/MCCASFormatSchemaBase.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>

namespace llvm {

class MachOCASWriter : public MachObjectWriter {
public:
  /// ObjectStore
  const Triple Target;
  cas::ObjectStore &CAS;
  CASBackendMode Mode;
  std::optional<MCTargetOptions::ResultCallBackTy> ResultCallBack;

  MachOCASWriter(std::unique_ptr<MCMachObjectTargetWriter> MOTW,
                 const Triple &TT, cas::ObjectStore &CAS, CASBackendMode Mode,
                 raw_pwrite_stream &OS, bool IsLittleEndian,
                 std::unique_ptr<mccasformats::MCCASSchema> Schema,
                 std::optional<MCTargetOptions::ResultCallBackTy> CallBack,
                 raw_pwrite_stream *CasIDOS = nullptr);

  uint8_t getAddressSize() { return Target.isArch32Bit() ? 4 : 8; }

  uint64_t writeObject() override;

  void resetBuffer() { OSOffset = InternalOS.tell(); }

  StringRef getContent() const { return InternalBuffer.substr(OSOffset); }

private:
  raw_pwrite_stream &OS;
  raw_pwrite_stream *CasIDOS;

  SmallString<512> InternalBuffer;
  raw_svector_ostream InternalOS;

  uint64_t OSOffset = 0;

  std::unique_ptr<mccasformats::MCCASSchema> Schema;
};

/// Construct a new Mach-O CAS writer instance.
///
/// This routine takes ownership of the target writer subclass.
///
/// \param MOTW - The target specific Mach-O writer subclass.
/// \param TT - The target triple.
/// \param CAS - The ObjectStore instance.
/// \param OS - The stream to write to.
/// \param Schema - The MCCAS schema to use.
/// \param CallBack - The callback to return the result CASID.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter> createMachOCASWriter(
    std::unique_ptr<MCMachObjectTargetWriter> MOTW, const Triple &TT,
    cas::ObjectStore &CAS, CASBackendMode Mode, raw_pwrite_stream &OS,
    bool IsLittleEndian, std::unique_ptr<mccasformats::MCCASSchema> Schema,
    std::optional<MCTargetOptions::ResultCallBackTy> CallBack = std::nullopt,
    raw_pwrite_stream *CasIDOS = nullptr);

} // end namespace llvm

#endif // LLVM_MC_MCMACHOCASWRITER_H
