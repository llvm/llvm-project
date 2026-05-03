//===--- CGDataAnalyzer.cpp - LLVM Advisor -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/CGDataAnalyzer.h"
#include "llvm/CGData/CodeGenDataReader.h"
#include "llvm/CGData/OutlinedHashTree.h"
#include "llvm/CGData/StableFunctionMap.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace llvm::advisor {

// Look for a .cgdata file adjacent to the object path or in the working dir.
static std::string findCGDataPath(const CapabilityContext &Ctx) {
  auto TryPath = [](StringRef Base) -> std::string {
    SmallString<256> P(Base);
    sys::path::replace_extension(P, "cgdata");
    if (sys::fs::exists(P))
      return P.str().str();
    return {};
  };
  if (!Ctx.ObjectPath.empty()) {
    std::string P = TryPath(Ctx.ObjectPath);
    if (!P.empty())
      return P;
  }
  if (!Ctx.WorkingDirectory.empty()) {
    SmallString<256> WD(Ctx.WorkingDirectory);
    sys::path::append(WD, "default.cgdata");
    if (sys::fs::exists(WD))
      return WD.str().str();
  }
  return {};
}

static json::Object buildResult(bool HasHashTree, bool HasFuncMap,
                                const OutlinedHashTree *HT,
                                const StableFunctionMap *FM) {
  json::Object R;
  R["has_outlined_hash_tree"] = HasHashTree;
  R["has_stable_function_map"] = HasFuncMap;
  if (HasHashTree && HT) {
    R["outlined_nodes"] = static_cast<int64_t>(HT->size());
    R["outlined_terminals"] =
        static_cast<int64_t>(HT->size(/*GetTerminalCountOnly=*/true));
    R["outlined_tree_depth"] = static_cast<int64_t>(HT->depth());
  }
  if (HasFuncMap && FM) {
    R["stable_unique_hashes"] =
        static_cast<int64_t>(FM->size(StableFunctionMap::UniqueHashCount));
    R["stable_total_functions"] =
        static_cast<int64_t>(FM->size(StableFunctionMap::TotalFunctionCount));
  }
  return R;
}

Expected<std::unique_ptr<CapabilityResult>>
CGDataAnalyzer::run(const CapabilityContext &Context) {
  std::string CGDataPath = findCGDataPath(Context);
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;

  if (!CGDataPath.empty()) {
    // Read from a dedicated .cgdata file.
    Expected<std::unique_ptr<CodeGenDataReader>> ReaderOrErr =
        CodeGenDataReader::create(CGDataPath, *vfs::getRealFileSystem());
    if (!ReaderOrErr)
      return ReaderOrErr.takeError();
    CodeGenDataReader &Reader = **ReaderOrErr;

    if (Error Err = Reader.read())
      return std::move(Err);

    bool HasHT = Reader.hasOutlinedHashTree();
    bool HasFM = Reader.hasStableFunctionMap();
    std::unique_ptr<OutlinedHashTree> HT;
    std::unique_ptr<StableFunctionMap> FM;
    if (HasHT)
      HT = Reader.releaseOutlinedHashTree();
    if (HasFM)
      FM = Reader.releaseStableFunctionMap();

    json::Object Result = buildResult(HasHT, HasFM, HT.get(), FM.get());
    Result["source"] = "cgdata_file";
    Result["cgdata_path"] = CGDataPath;
    Result["version"] = static_cast<int64_t>(Reader.getVersion());
    return makeJSONResult(CapID, UnitID, std::move(Result));
  }

  // Fall back to extracting CGData embedded in the object file.
  if (Context.ObjectPath.empty() || !sys::fs::exists(Context.ObjectPath))
    return makeUnavailableResult(
        CapID, UnitID, "no .cgdata file or object file for CGData analysis");

  Expected<object::OwningBinary<object::Binary>> BinOrErr =
      object::createBinary(Context.ObjectPath);
  if (!BinOrErr)
    return BinOrErr.takeError();
  const auto *ObjFile = dyn_cast<object::ObjectFile>(BinOrErr->getBinary());
  if (!ObjFile)
    return createStringError(inconvertibleErrorCode(), "not an object file: %s",
                             Context.ObjectPath.c_str());

  OutlinedHashTreeRecord HashRecord;
  StableFunctionMapRecord FuncRecord;
  if (Error Err = CodeGenDataReader::mergeFromObjectFile(ObjFile, HashRecord,
                                                         FuncRecord))
    return std::move(Err);

  const OutlinedHashTree *HT = HashRecord.HashTree.get();
  const StableFunctionMap *FM = FuncRecord.FunctionMap.get();
  // A hash tree containing only the root node (size == 1) is considered
  // effectively empty because it carries no outlined-function data.
  bool HasHT = HT && HT->size() != 1;
  bool HasFM = FM && !FM->empty();

  json::Object Result = buildResult(HasHT, HasFM, HT, FM);
  Result["source"] = "object_embedded";
  Result["object_path"] = Context.ObjectPath;
  if (!HasHT && !HasFM)
    Result["note"] = "object file contains no embedded CodeGenData sections";
  return makeJSONResult(CapID, UnitID, std::move(Result));
}

} // namespace llvm::advisor
