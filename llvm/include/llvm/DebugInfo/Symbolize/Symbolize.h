//===- Symbolize.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Header for LLVM symbolization library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_SYMBOLIZE_SYMBOLIZE_H
#define LLVM_DEBUGINFO_SYMBOLIZE_SYMBOLIZE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/DebugInfo/Symbolize/DIFetcher.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace symbolize {

using namespace object;

using FunctionNameKind = DILineInfoSpecifier::FunctionNameKind;
using FileLineInfoKind = DILineInfoSpecifier::FileLineInfoKind;

class LLVMSymbolizer {
public:
  struct Options {
    FunctionNameKind PrintFunctions = FunctionNameKind::LinkageName;
    FileLineInfoKind PathStyle = FileLineInfoKind::AbsoluteFilePath;
    bool UseSymbolTable = true;
    bool Demangle = true;
    bool RelativeAddresses = false;
    bool UntagAddresses = false;
    bool UseDIA = false;
    std::string DefaultArch;
    std::vector<std::string> DsymHints;
    std::string FallbackDebugPath;
    std::string DWPName;
    std::vector<std::string> DebugFileDirectory;
  };

  LLVMSymbolizer() = default;
  LLVMSymbolizer(const Options &Opts) : Opts(Opts) {}

  ~LLVMSymbolizer() { flush(); }

  // Overloads accepting ObjectFile does not support COFF currently
  Expected<DILineInfo> symbolizeCode(const ObjectFile &Obj,
                                     object::SectionedAddress ModuleOffset);
  Expected<DILineInfo> symbolizeCode(const std::string &ModuleName,
                                     object::SectionedAddress ModuleOffset);
  Expected<DILineInfo> symbolizeCode(ArrayRef<uint8_t> BuildID,
                                     object::SectionedAddress ModuleOffset);
  Expected<DIInliningInfo>
  symbolizeInlinedCode(const ObjectFile &Obj,
                       object::SectionedAddress ModuleOffset);
  Expected<DIInliningInfo>
  symbolizeInlinedCode(const std::string &ModuleName,
                       object::SectionedAddress ModuleOffset);
  Expected<DIInliningInfo>
  symbolizeInlinedCode(ArrayRef<uint8_t> BuildID,
                       object::SectionedAddress ModuleOffset);

  Expected<DIGlobal> symbolizeData(const ObjectFile &Obj,
                                   object::SectionedAddress ModuleOffset);
  Expected<DIGlobal> symbolizeData(const std::string &ModuleName,
                                   object::SectionedAddress ModuleOffset);
  Expected<DIGlobal> symbolizeData(ArrayRef<uint8_t> BuildID,
                                   object::SectionedAddress ModuleOffset);
  Expected<std::vector<DILocal>>
  symbolizeFrame(const ObjectFile &Obj, object::SectionedAddress ModuleOffset);
  Expected<std::vector<DILocal>>
  symbolizeFrame(const std::string &ModuleName,
                 object::SectionedAddress ModuleOffset);
  Expected<std::vector<DILocal>>
  symbolizeFrame(ArrayRef<uint8_t> BuildID,
                 object::SectionedAddress ModuleOffset);
  void flush();

  static std::string
  DemangleName(const std::string &Name,
               const SymbolizableModule *DbiModuleDescriptor);

  void addDIFetcher(std::unique_ptr<DIFetcher> Fetcher) {
    DIFetchers.push_back(std::move(Fetcher));
  }

private:
  // Bundles together object file with code/data and object file with
  // corresponding debug info. These objects can be the same.
  using ObjectPair = std::pair<const ObjectFile *, const ObjectFile *>;

  template <typename T>
  Expected<DILineInfo>
  symbolizeCodeCommon(const T &ModuleSpecifier,
                      object::SectionedAddress ModuleOffset);
  template <typename T>
  Expected<DIInliningInfo>
  symbolizeInlinedCodeCommon(const T &ModuleSpecifier,
                             object::SectionedAddress ModuleOffset);
  template <typename T>
  Expected<DIGlobal> symbolizeDataCommon(const T &ModuleSpecifier,
                                         object::SectionedAddress ModuleOffset);
  template <typename T>
  Expected<std::vector<DILocal>>
  symbolizeFrameCommon(const T &ModuleSpecifier,
                       object::SectionedAddress ModuleOffset);

  /// Returns a SymbolizableModule or an error if loading debug info failed.
  /// Only one attempt is made to load a module, and errors during loading are
  /// only reported once. Subsequent calls to get module info for a module that
  /// failed to load will return nullptr.
  Expected<SymbolizableModule *>
  getOrCreateModuleInfo(const std::string &ModuleName);
  Expected<SymbolizableModule *> getOrCreateModuleInfo(const ObjectFile &Obj);

  /// Returns a SymbolizableModule or an error if loading debug info failed.
  /// Unlike the above, errors are reported each time, since they are more
  /// likely to be transient.
  Expected<SymbolizableModule *>
  getOrCreateModuleInfo(ArrayRef<uint8_t> BuildID);

  Expected<SymbolizableModule *>
  createModuleInfo(const ObjectFile *Obj, std::unique_ptr<DIContext> Context,
                   StringRef ModuleName);

  ObjectFile *lookUpDsymFile(const std::string &Path,
                             const MachOObjectFile *ExeObj,
                             const std::string &ArchName);
  ObjectFile *lookUpDebuglinkObject(const std::string &Path,
                                    const ObjectFile *Obj,
                                    const std::string &ArchName);
  ObjectFile *lookUpBuildIDObject(const std::string &Path,
                                  const ELFObjectFileBase *Obj,
                                  const std::string &ArchName);

  bool findDebugBinary(const std::string &OrigPath,
                       const std::string &DebuglinkName, uint32_t CRCHash,
                       std::string &Result);

  bool getOrFindDebugBinary(const ArrayRef<uint8_t> BuildID,
                            std::string &Result);

  /// Returns pair of pointers to object and debug object.
  Expected<ObjectPair> getOrCreateObjectPair(const std::string &Path,
                                             const std::string &ArchName);

  /// Return a pointer to object file at specified path, for a specified
  /// architecture (e.g. if path refers to a Mach-O universal binary, only one
  /// object file from it will be returned).
  Expected<ObjectFile *> getOrCreateObject(const std::string &Path,
                                           const std::string &ArchName);

  std::map<std::string, std::unique_ptr<SymbolizableModule>, std::less<>>
      Modules;
  StringMap<std::string> BuildIDPaths;

  /// Contains cached results of getOrCreateObjectPair().
  std::map<std::pair<std::string, std::string>, ObjectPair>
      ObjectPairForPathArch;

  /// Contains parsed binary for each path, or parsing error.
  std::map<std::string, OwningBinary<Binary>> BinaryForPath;

  /// Parsed object file for path/architecture pair, where "path" refers
  /// to Mach-O universal binary.
  std::map<std::pair<std::string, std::string>, std::unique_ptr<ObjectFile>>
      ObjectForUBPathAndArch;

  Options Opts;

  SmallVector<std::unique_ptr<DIFetcher>> DIFetchers;
};

} // end namespace symbolize
} // end namespace llvm

#endif // LLVM_DEBUGINFO_SYMBOLIZE_SYMBOLIZE_H
