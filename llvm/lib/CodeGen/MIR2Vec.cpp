//===- MIR2Vec.cpp - Implementation of MIR2Vec ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the MIR2Vec algorithm for Machine IR embeddings.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace mir2vec;

#define DEBUG_TYPE "mir2vec"

STATISTIC(MIRVocabMissCounter,
          "Number of lookups to MIR entities not present in the vocabulary");

namespace llvm {
namespace mir2vec {
cl::OptionCategory MIR2VecCategory("MIR2Vec Options");

// FIXME: Use a default vocab when not specified
static cl::opt<std::string>
    VocabFile("mir2vec-vocab-path", cl::Optional,
              cl::desc("Path to the vocabulary file for MIR2Vec"), cl::init(""),
              cl::cat(MIR2VecCategory));
cl::opt<float> OpcWeight("mir2vec-opc-weight", cl::Optional, cl::init(1.0),
                         cl::desc("Weight for machine opcode embeddings"),
                         cl::cat(MIR2VecCategory));
} // namespace mir2vec
} // namespace llvm

//===----------------------------------------------------------------------===//
// Vocabulary Implementation
//===----------------------------------------------------------------------===//

MIRVocabulary::MIRVocabulary(VocabMap &&OpcodeEntries,
                             const TargetInstrInfo *TII)
    : TII(*TII) {
  // Fixme: Use static factory methods for creating vocabularies instead of
  // public constructors
  // Early return for invalid inputs - creates empty/invalid vocabulary
  if (!TII || OpcodeEntries.empty())
    return;

  buildCanonicalOpcodeMapping();

  unsigned CanonicalOpcodeCount = UniqueBaseOpcodeNames.size();
  assert(CanonicalOpcodeCount > 0 &&
         "No canonical opcodes found for target - invalid vocabulary");
  Layout.OperandBase = CanonicalOpcodeCount;
  generateStorage(OpcodeEntries);
  Layout.TotalEntries = Storage.size();
}

std::string MIRVocabulary::extractBaseOpcodeName(StringRef InstrName) {
  // Extract base instruction name using regex to capture letters and
  // underscores Examples: "ADD32rr" -> "ADD", "ARITH_FENCE" -> "ARITH_FENCE"
  //
  // TODO: Consider more sophisticated extraction:
  // - Handle complex prefixes like "AVX1_SETALLONES" correctly (Currently, it
  // would naively map to "AVX")
  // - Extract width suffixes (8,16,32,64) as separate features
  // - Capture addressing mode suffixes (r,i,m,ri,etc.) for better analysis
  // (Currently, instances like "MOV32mi" map to "MOV", but "ADDPDrr" would map
  // to "ADDPDrr")

  assert(!InstrName.empty() && "Instruction name should not be empty");

  // Use regex to extract initial sequence of letters and underscores
  static const Regex BaseOpcodeRegex("([a-zA-Z_]+)");
  SmallVector<StringRef, 2> Matches;

  if (BaseOpcodeRegex.match(InstrName, &Matches) && Matches.size() > 1) {
    StringRef Match = Matches[1];
    // Trim trailing underscores
    while (!Match.empty() && Match.back() == '_')
      Match = Match.drop_back();
    return Match.str();
  }

  // Fallback to original name if no pattern matches
  return InstrName.str();
}

unsigned MIRVocabulary::getCanonicalIndexForBaseName(StringRef BaseName) const {
  assert(!UniqueBaseOpcodeNames.empty() && "Canonical mapping not built");
  auto It = std::find(UniqueBaseOpcodeNames.begin(),
                      UniqueBaseOpcodeNames.end(), BaseName.str());
  assert(It != UniqueBaseOpcodeNames.end() &&
         "Base name not found in unique opcodes");
  return std::distance(UniqueBaseOpcodeNames.begin(), It);
}

unsigned MIRVocabulary::getCanonicalOpcodeIndex(unsigned Opcode) const {
  assert(isValid() && "MIR2Vec Vocabulary is invalid");
  auto BaseOpcode = extractBaseOpcodeName(TII.getName(Opcode));
  return getCanonicalIndexForBaseName(BaseOpcode);
}

std::string MIRVocabulary::getStringKey(unsigned Pos) const {
  assert(isValid() && "MIR2Vec Vocabulary is invalid");
  assert(Pos < Layout.TotalEntries && "Position out of bounds in vocabulary");

  // For now, all entries are opcodes since we only have one section
  if (Pos < Layout.OperandBase && Pos < UniqueBaseOpcodeNames.size()) {
    // Convert canonical index back to base opcode name
    auto It = UniqueBaseOpcodeNames.begin();
    std::advance(It, Pos);
    return *It;
  }

  llvm_unreachable("Invalid position in vocabulary");
  return "";
}

void MIRVocabulary::generateStorage(const VocabMap &OpcodeMap) {

  // Helper for handling missing entities in the vocabulary.
  // Currently, we use a zero vector. In the future, we will throw an error to
  // ensure that *all* known entities are present in the vocabulary.
  auto handleMissingEntity = [](StringRef Key) {
    LLVM_DEBUG(errs() << "MIR2Vec: Missing vocabulary entry for " << Key
                      << "; using zero vector. This will result in an error "
                         "in the future.\n");
    ++MIRVocabMissCounter;
  };

  // Initialize opcode embeddings section
  unsigned EmbeddingDim = OpcodeMap.begin()->second.size();
  std::vector<Embedding> OpcodeEmbeddings(Layout.OperandBase,
                                          Embedding(EmbeddingDim));

  // Populate opcode embeddings using canonical mapping
  for (auto COpcodeName : UniqueBaseOpcodeNames) {
    if (auto It = OpcodeMap.find(COpcodeName); It != OpcodeMap.end()) {
      auto COpcodeIndex = getCanonicalIndexForBaseName(COpcodeName);
      assert(COpcodeIndex < Layout.OperandBase &&
             "Canonical index out of bounds");
      OpcodeEmbeddings[COpcodeIndex] = It->second;
    } else {
      handleMissingEntity(COpcodeName);
    }
  }

  // TODO: Add operand/argument embeddings as additional sections
  // This will require extending the vocabulary format and layout

  // Scale the vocabulary sections based on the provided weights
  auto scaleVocabSection = [](std::vector<Embedding> &Embeddings,
                              double Weight) {
    for (auto &Embedding : Embeddings)
      Embedding *= Weight;
  };
  scaleVocabSection(OpcodeEmbeddings, OpcWeight);

  std::vector<std::vector<Embedding>> Sections(1);
  Sections[0] = std::move(OpcodeEmbeddings);

  Storage = ir2vec::VocabStorage(std::move(Sections));
}

void MIRVocabulary::buildCanonicalOpcodeMapping() {
  // Check if already built
  if (!UniqueBaseOpcodeNames.empty())
    return;

  // Build mapping from opcodes to canonical base opcode indices
  for (unsigned Opcode = 0; Opcode < TII.getNumOpcodes(); ++Opcode) {
    std::string BaseOpcode = extractBaseOpcodeName(TII.getName(Opcode));
    UniqueBaseOpcodeNames.insert(BaseOpcode);
  }

  LLVM_DEBUG(dbgs() << "MIR2Vec: Built canonical mapping for target with "
                    << UniqueBaseOpcodeNames.size()
                    << " unique base opcodes\n");
}

//===----------------------------------------------------------------------===//
// MIR2VecVocabLegacyAnalysis Implementation
//===----------------------------------------------------------------------===//

char MIR2VecVocabLegacyAnalysis::ID = 0;
INITIALIZE_PASS_BEGIN(MIR2VecVocabLegacyAnalysis, "mir2vec-vocab-analysis",
                      "MIR2Vec Vocabulary Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_END(MIR2VecVocabLegacyAnalysis, "mir2vec-vocab-analysis",
                    "MIR2Vec Vocabulary Analysis", false, true)

StringRef MIR2VecVocabLegacyAnalysis::getPassName() const {
  return "MIR2Vec Vocabulary Analysis";
}

Error MIR2VecVocabLegacyAnalysis::readVocabulary() {
  // TODO: Extend vocabulary format to support multiple sections
  // (opcodes, operands, etc.) similar to IR2Vec structure
  if (VocabFile.empty())
    return createStringError(
        errc::invalid_argument,
        "MIR2Vec vocabulary file path not specified; set it "
        "using --mir2vec-vocab-path");

  auto BufOrError = MemoryBuffer::getFileOrSTDIN(VocabFile, /*IsText=*/true);
  if (!BufOrError)
    return createFileError(VocabFile, BufOrError.getError());

  auto Content = BufOrError.get()->getBuffer();

  Expected<json::Value> ParsedVocabValue = json::parse(Content);
  if (!ParsedVocabValue)
    return ParsedVocabValue.takeError();

  unsigned Dim = 0;
  if (auto Err = ir2vec::VocabStorage::parseVocabSection(
          "entities", *ParsedVocabValue, StrVocabMap, Dim))
    return Err;

  return Error::success();
}

void MIR2VecVocabLegacyAnalysis::emitError(Error Err, LLVMContext &Ctx) {
  Ctx.emitError(toString(std::move(Err)));
}

mir2vec::MIRVocabulary
MIR2VecVocabLegacyAnalysis::getMIR2VecVocabulary(const Module &M) {
  if (StrVocabMap.empty()) {
    if (Error Err = readVocabulary()) {
      emitError(std::move(Err), M.getContext());
      return mir2vec::MIRVocabulary(std::move(StrVocabMap), nullptr);
    }
  }

  // Get machine module info to access machine functions and target info
  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  // Find first available machine function to get target instruction info
  for (const auto &F : M) {
    if (F.isDeclaration())
      continue;

    if (auto *MF = MMI.getMachineFunction(F)) {
      const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
      return mir2vec::MIRVocabulary(std::move(StrVocabMap), TII);
    }
  }

  // No machine functions available - return invalid vocabulary
  emitError(make_error<StringError>("No machine functions found in module",
                                    inconvertibleErrorCode()),
            M.getContext());
  return mir2vec::MIRVocabulary(std::move(StrVocabMap), nullptr);
}

//===----------------------------------------------------------------------===//
// Printer Passes Implementation
//===----------------------------------------------------------------------===//

char MIR2VecVocabPrinterLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(MIR2VecVocabPrinterLegacyPass, "print-mir2vec-vocab",
                      "MIR2Vec Vocabulary Printer Pass", false, true)
INITIALIZE_PASS_DEPENDENCY(MIR2VecVocabLegacyAnalysis)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_END(MIR2VecVocabPrinterLegacyPass, "print-mir2vec-vocab",
                    "MIR2Vec Vocabulary Printer Pass", false, true)

bool MIR2VecVocabPrinterLegacyPass::runOnMachineFunction(MachineFunction &MF) {
  return false;
}

bool MIR2VecVocabPrinterLegacyPass::doFinalization(Module &M) {
  auto &Analysis = getAnalysis<MIR2VecVocabLegacyAnalysis>();
  auto MIR2VecVocab = Analysis.getMIR2VecVocabulary(M);

  if (!MIR2VecVocab.isValid()) {
    OS << "MIR2Vec Vocabulary Printer: Invalid vocabulary\n";
    return false;
  }

  unsigned Pos = 0;
  for (const auto &Entry : MIR2VecVocab) {
    OS << "Key: " << MIR2VecVocab.getStringKey(Pos++) << ": ";
    Entry.print(OS);
  }

  return false;
}

MachineFunctionPass *
llvm::createMIR2VecVocabPrinterLegacyPass(raw_ostream &OS) {
  return new MIR2VecVocabPrinterLegacyPass(OS);
}
