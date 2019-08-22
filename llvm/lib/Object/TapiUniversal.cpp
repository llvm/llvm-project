//===- TapiUniversal.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Text-based Dynamic Library Stub format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/TapiUniversal.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"

using namespace llvm;
using namespace MachO;
using namespace object;

TapiUniversal::TapiUniversal(MemoryBufferRef Source, Error &Err)
    : Binary(ID_TapiUniversal, Source) {
  auto Result = TextAPIReader::get(Source);
  ErrorAsOutParameter ErrAsOuParam(&Err);
  if (!Result) {
    Err = Result.takeError();
    return;
  }
  ParsedFile = std::move(Result.get());

  auto Archs = ParsedFile->getArchitectures();
  for (auto Arch : Archs)
    Architectures.emplace_back(Arch);
}

TapiUniversal::~TapiUniversal() = default;

Expected<std::unique_ptr<TapiFile>>
TapiUniversal::ObjectForArch::getAsObjectFile() const {
  return std::unique_ptr<TapiFile>(new TapiFile(Parent->getMemoryBufferRef(),
                                                *Parent->ParsedFile.get(),
                                                Parent->Architectures[Index]));
}

Expected<std::unique_ptr<TapiUniversal>>
TapiUniversal::create(MemoryBufferRef Source) {
  Error Err = Error::success();
  std::unique_ptr<TapiUniversal> Ret(new TapiUniversal(Source, Err));
  if (Err)
    return std::move(Err);
  return std::move(Ret);
}
