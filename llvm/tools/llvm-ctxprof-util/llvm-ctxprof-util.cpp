//===--- PGOCtxProfJSONReader.h - JSON format  ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// JSON format for the contextual profile for testing.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/ProfileData/CtxInstrContextNode.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::SubCommand FromJSON("fromJSON", "Convert from json");

static cl::opt<std::string> InputFilename(
    "input", cl::value_desc("input"), cl::init("-"),
    cl::desc(
        "Input file. The format is an array of contexts.\n"
        "Each context is a dictionary with the following keys:\n"
        "'Guid', mandatory. The value is a 64-bit integer.\n"
        "'Counters', mandatory. An array of 32-bit ints. These are the "
        "counter values.\n"
        "'Contexts', optional. An array containing arrays of contexts. The "
        "context array at a position 'i' is the set of callees at that "
        "callsite index. Use an empty array to indicate no callees."),
    cl::sub(FromJSON));

static cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                           cl::init("-"),
                                           cl::desc("Output file"),
                                           cl::sub(FromJSON));

namespace {
// A structural representation of the JSON input.
struct DeserializableCtx {
  GlobalValue::GUID Guid = 0;
  std::vector<uint64_t> Counters;
  std::vector<std::vector<DeserializableCtx>> Callsites;
};

ctx_profile::ContextNode *
createNode(std::vector<std::unique_ptr<char[]>> &Nodes,
           const std::vector<DeserializableCtx> &DCList);

// Convert a DeserializableCtx into a ContextNode, potentially linking it to
// its sibling (e.g. callee at same callsite) "Next".
ctx_profile::ContextNode *
createNode(std::vector<std::unique_ptr<char[]>> &Nodes,
           const DeserializableCtx &DC,
           ctx_profile::ContextNode *Next = nullptr) {
  auto AllocSize = ctx_profile::ContextNode::getAllocSize(DC.Counters.size(),
                                                          DC.Callsites.size());
  auto *Mem = Nodes.emplace_back(std::make_unique<char[]>(AllocSize)).get();
  std::memset(Mem, 0, AllocSize);
  auto *Ret = new (Mem) ctx_profile::ContextNode(DC.Guid, DC.Counters.size(),
                                                 DC.Callsites.size(), Next);
  std::memcpy(Ret->counters(), DC.Counters.data(),
              sizeof(uint64_t) * DC.Counters.size());
  for (const auto &[I, DCList] : llvm::enumerate(DC.Callsites))
    Ret->subContexts()[I] = createNode(Nodes, DCList);
  return Ret;
}

// Convert a list of DeserializableCtx into a linked list of ContextNodes.
ctx_profile::ContextNode *
createNode(std::vector<std::unique_ptr<char[]>> &Nodes,
           const std::vector<DeserializableCtx> &DCList) {
  ctx_profile::ContextNode *List = nullptr;
  for (const auto &DC : DCList)
    List = createNode(Nodes, DC, List);
  return List;
}
} // namespace

namespace llvm {
namespace json {
// Hook into the JSON deserialization.
bool fromJSON(const Value &E, DeserializableCtx &R, Path P) {
  json::ObjectMapper Mapper(E, P);
  return Mapper && Mapper.map("Guid", R.Guid) &&
         Mapper.map("Counters", R.Counters) &&
         Mapper.mapOptional("Callsites", R.Callsites);
}
} // namespace json
} // namespace llvm

// Save the bitstream profile from the JSON representation.
Error convertFromJSON() {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!BufOrError)
    return createFileError(InputFilename, BufOrError.getError());
  auto P = json::parse(BufOrError.get()->getBuffer());
  if (!P)
    return P.takeError();

  std::vector<DeserializableCtx> DCList;
  json::Path::Root R("");
  if (!fromJSON(*P, DCList, R))
    return R.getError();
  // Nodes provides memory backing for the ContextualNodes.
  std::vector<std::unique_ptr<char[]>> Nodes;
  std::error_code EC;
  raw_fd_stream Out(OutputFilename, EC);
  if (EC)
    return createStringError(EC, "failed to open output");
  PGOCtxProfileWriter Writer(Out);
  for (const auto &DC : DCList) {
    auto *TopList = createNode(Nodes, DC);
    if (!TopList)
      return createStringError(
          "Unexpected error converting internal structure to ctx profile");
    Writer.write(*TopList);
  }
  if (EC)
    return createStringError(EC, "failed to write output");
  return Error::success();
}

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLVM Contextual Profile Utils\n");
  ExitOnError ExitOnErr("llvm-ctxprof-util: ");
  if (FromJSON) {
    if (auto E = convertFromJSON()) {
      handleAllErrors(std::move(E), [&](const ErrorInfoBase &E) {
        E.log(errs());
        errs() << "\n";
      });
      return 1;
    }
    return 0;
  }
  cl::PrintHelpMessage();
  return 1;
}
