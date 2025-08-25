//===- lib/MC/MachOCASWriter.cpp - Mach-O CAS File Writer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CASUtil/Utils.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCMachOCASWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolMachO.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MCCAS/MCCASFormatSchemaBase.h"
#include "llvm/MCCAS/MCCASObjectV1.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::mccasformats;

#define DEBUG_TYPE "mc"

MachOCASWriter::MachOCASWriter(
    std::unique_ptr<MCMachObjectTargetWriter> MOTW, const Triple &TT,
    cas::ObjectStore &CAS, CASBackendMode Mode, raw_pwrite_stream &OS,
    bool IsLittleEndian,
    std::function<const cas::ObjectProxy(llvm::MachOCASWriter &,
                                         llvm::MCAssembler &,
                                         cas::ObjectStore &, raw_ostream *)>
        CreateFromMcAssembler,
    std::function<Error(cas::ObjectProxy, cas::ObjectStore &, raw_ostream &)>
        SerializeObjectFile,
    std::optional<MCTargetOptions::ResultCallBackTy> ResultCallBack,
    raw_pwrite_stream *CasIDOS)
    : MachObjectWriter(std::move(MOTW), InternalOS, IsLittleEndian), Target(TT),
      CAS(CAS), Mode(Mode), ResultCallBack(ResultCallBack), OS(OS),
      CasIDOS(CasIDOS), InternalOS(InternalBuffer),
      CreateFromMcAssembler(CreateFromMcAssembler),
      SerializeObjectFile(SerializeObjectFile) {
  assert(TT.isLittleEndian() == IsLittleEndian && "Endianess should match");
}

uint64_t MachOCASWriter::writeObject() {
  auto &Asm = *this->Asm;
  uint64_t StartOffset = OS.tell();
  auto CASObj = CreateFromMcAssembler(*this, Asm, CAS, nullptr);

  auto VerifyObject = [&]() -> Error {
    SmallString<512> ObjectBuffer;
    raw_svector_ostream ObjectOS(ObjectBuffer);
    if (auto E = SerializeObjectFile(CASObj, CAS, ObjectOS))
      return E;

    if (!ObjectBuffer.equals(InternalBuffer))
      return createStringError(
          inconvertibleErrorCode(),
          "CASBackend output round-trip verification error");

    OS << ObjectBuffer;
    return Error::success();
  };

  if (CasIDOS)
    writeCASIDBuffer(CASObj.getID(), *CasIDOS);

  // If there is a callback, then just hand off the result through callback.
  if (ResultCallBack) {
    cantFail((*ResultCallBack)(CASObj.getID()));
  }

  switch (Mode) {
  case CASBackendMode::CASID:
    writeCASIDBuffer(CASObj.getID(), OS);
    break;
  case CASBackendMode::Native: {
    auto E = SerializeObjectFile(CASObj, CAS, OS);
    if (E)
      report_fatal_error(std::move(E));
    break;
  }
  case CASBackendMode::Verify: {
    if (auto E = VerifyObject())
      report_fatal_error(std::move(E));
  }
  }

  return OS.tell() - StartOffset;
}

std::unique_ptr<MCObjectWriter> llvm::createMachOCASWriter(
    std::unique_ptr<MCMachObjectTargetWriter> MOTW, const Triple &TT,
    cas::ObjectStore &CAS, CASBackendMode Mode, raw_pwrite_stream &OS,
    bool IsLittleEndian,
    std::function<const cas::ObjectProxy(llvm::MachOCASWriter &,
                                         llvm::MCAssembler &,
                                         cas::ObjectStore &, raw_ostream *)>
        CreateFromMcAssembler,
    std::function<Error(cas::ObjectProxy, cas::ObjectStore &, raw_ostream &)>
        SerializeObjectFile,
    std::optional<MCTargetOptions::ResultCallBackTy> ResultCallBack,
    raw_pwrite_stream *CasIDOS) {
  return std::make_unique<MachOCASWriter>(
      std::move(MOTW), TT, CAS, Mode, OS, IsLittleEndian, CreateFromMcAssembler,
      SerializeObjectFile, ResultCallBack, CasIDOS);
}
