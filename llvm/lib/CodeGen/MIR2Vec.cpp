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
#include "llvm/ADT/DepthFirstIterator.h"
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
cl::opt<float> CommonOperandWeight(
    "mir2vec-common-operand-weight", cl::Optional, cl::init(1.0),
    cl::desc("Weight for common operand embeddings"), cl::cat(MIR2VecCategory));
cl::opt<float>
    RegOperandWeight("mir2vec-reg-operand-weight", cl::Optional, cl::init(1.0),
                     cl::desc("Weight for register operand embeddings"),
                     cl::cat(MIR2VecCategory));
cl::opt<MIR2VecKind> MIR2VecEmbeddingKind(
    "mir2vec-kind", cl::Optional,
    cl::values(clEnumValN(MIR2VecKind::Symbolic, "symbolic",
                          "Generate symbolic embeddings for MIR")),
    cl::init(MIR2VecKind::Symbolic), cl::desc("MIR2Vec embedding kind"),
    cl::cat(MIR2VecCategory));

} // namespace mir2vec
} // namespace llvm

//===----------------------------------------------------------------------===//
// Vocabulary
//===----------------------------------------------------------------------===//

MIRVocabulary::MIRVocabulary(VocabMap &&OpcodeMap, VocabMap &&CommonOperandMap,
                             VocabMap &&PhysicalRegisterMap,
                             VocabMap &&VirtualRegisterMap,
                             const TargetInstrInfo &TII,
                             const TargetRegisterInfo &TRI,
                             const MachineRegisterInfo &MRI)
    : TII(TII), TRI(TRI), MRI(MRI) {
  buildCanonicalOpcodeMapping();
  unsigned CanonicalOpcodeCount = UniqueBaseOpcodeNames.size();
  assert(CanonicalOpcodeCount > 0 &&
         "No canonical opcodes found for target - invalid vocabulary");

  buildRegisterOperandMapping();

  // Define layout of vocabulary sections
  Layout.OpcodeBase = 0;
  Layout.CommonOperandBase = CanonicalOpcodeCount;
  // We expect same classes for physical and virtual registers
  Layout.PhyRegBase = Layout.CommonOperandBase + std::size(CommonOperandNames);
  Layout.VirtRegBase = Layout.PhyRegBase + RegisterOperandNames.size();

  generateStorage(OpcodeMap, CommonOperandMap, PhysicalRegisterMap,
                  VirtualRegisterMap);
  Layout.TotalEntries = Storage.size();
}

Expected<MIRVocabulary>
MIRVocabulary::create(VocabMap &&OpcodeMap, VocabMap &&CommonOperandMap,
                      VocabMap &&PhyRegMap, VocabMap &&VirtRegMap,
                      const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
                      const MachineRegisterInfo &MRI) {
  if (OpcodeMap.empty() || CommonOperandMap.empty() || PhyRegMap.empty() ||
      VirtRegMap.empty())
    return createStringError(errc::invalid_argument,
                             "Empty vocabulary entries provided");

  MIRVocabulary Vocab(std::move(OpcodeMap), std::move(CommonOperandMap),
                      std::move(PhyRegMap), std::move(VirtRegMap), TII, TRI,
                      MRI);

  // Validate Storage after construction
  if (!Vocab.Storage.isValid())
    return createStringError(errc::invalid_argument,
                             "Failed to create valid vocabulary storage");
  Vocab.ZeroEmbedding = Embedding(Vocab.Storage.getDimension(), 0.0);
  return std::move(Vocab);
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
  auto BaseOpcode = extractBaseOpcodeName(TII.getName(Opcode));
  return getCanonicalIndexForBaseName(BaseOpcode);
}

unsigned
MIRVocabulary::getCanonicalIndexForOperandName(StringRef OperandName) const {
  auto It = std::find(std::begin(CommonOperandNames),
                      std::end(CommonOperandNames), OperandName);
  assert(It != std::end(CommonOperandNames) &&
         "Operand name not found in common operands");
  return Layout.CommonOperandBase +
         std::distance(std::begin(CommonOperandNames), It);
}

unsigned
MIRVocabulary::getCanonicalIndexForRegisterClass(StringRef RegName,
                                                 bool IsPhysical) const {
  auto It = std::find(RegisterOperandNames.begin(), RegisterOperandNames.end(),
                      RegName);
  assert(It != RegisterOperandNames.end() &&
         "Register name not found in register operands");
  unsigned LocalIndex = std::distance(RegisterOperandNames.begin(), It);
  return (IsPhysical ? Layout.PhyRegBase : Layout.VirtRegBase) + LocalIndex;
}

std::string MIRVocabulary::getStringKey(unsigned Pos) const {
  assert(Pos < Layout.TotalEntries && "Position out of bounds in vocabulary");

  // Handle opcodes section
  if (Pos < Layout.CommonOperandBase) {
    // Convert canonical index back to base opcode name
    auto It = UniqueBaseOpcodeNames.begin();
    std::advance(It, Pos);
    assert(It != UniqueBaseOpcodeNames.end() &&
           "Canonical index out of bounds in opcode section");
    return *It;
  }

  auto getLocalIndex = [](unsigned Pos, size_t BaseOffset, size_t Bound,
                          const char *Msg) {
    unsigned LocalIndex = Pos - BaseOffset;
    assert(LocalIndex < Bound && Msg);
    return LocalIndex;
  };

  // Handle common operands section
  if (Pos < Layout.PhyRegBase) {
    unsigned LocalIndex = getLocalIndex(
        Pos, Layout.CommonOperandBase, std::size(CommonOperandNames),
        "Local index out of bounds in common operands");
    return CommonOperandNames[LocalIndex].str();
  }

  // Handle physical registers section
  if (Pos < Layout.VirtRegBase) {
    unsigned LocalIndex =
        getLocalIndex(Pos, Layout.PhyRegBase, RegisterOperandNames.size(),
                      "Local index out of bounds in physical registers");
    return "PhyReg_" + RegisterOperandNames[LocalIndex];
  }

  // Handle virtual registers section
  unsigned LocalIndex =
      getLocalIndex(Pos, Layout.VirtRegBase, RegisterOperandNames.size(),
                    "Local index out of bounds in virtual registers");
  return "VirtReg_" + RegisterOperandNames[LocalIndex];
}

void MIRVocabulary::generateStorage(const VocabMap &OpcodeMap,
                                    const VocabMap &CommonOperandsMap,
                                    const VocabMap &PhyRegMap,
                                    const VocabMap &VirtRegMap) {

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
  std::vector<Embedding> OpcodeEmbeddings(Layout.CommonOperandBase,
                                          Embedding(EmbeddingDim));

  // Populate opcode embeddings using canonical mapping
  for (auto COpcodeName : UniqueBaseOpcodeNames) {
    if (auto It = OpcodeMap.find(COpcodeName); It != OpcodeMap.end()) {
      auto COpcodeIndex = getCanonicalIndexForBaseName(COpcodeName);
      assert(COpcodeIndex < Layout.CommonOperandBase &&
             "Canonical index out of bounds");
      OpcodeEmbeddings[COpcodeIndex] = It->second;
    } else {
      handleMissingEntity(COpcodeName);
    }
  }

  // Initialize common operand embeddings section
  std::vector<Embedding> CommonOperandEmbeddings(std::size(CommonOperandNames),
                                                 Embedding(EmbeddingDim));
  unsigned OperandIndex = 0;
  for (const auto &CommonOperandName : CommonOperandNames) {
    if (auto It = CommonOperandsMap.find(CommonOperandName.str());
        It != CommonOperandsMap.end()) {
      CommonOperandEmbeddings[OperandIndex] = It->second;
    } else {
      handleMissingEntity(CommonOperandName);
    }
    ++OperandIndex;
  }

  // Helper lambda for creating register operand embeddings
  auto createRegisterEmbeddings = [&](const VocabMap &RegMap) {
    std::vector<Embedding> RegEmbeddings(TRI.getNumRegClasses(),
                                         Embedding(EmbeddingDim));
    unsigned RegOperandIndex = 0;
    for (const auto &RegOperandName : RegisterOperandNames) {
      if (auto It = RegMap.find(RegOperandName); It != RegMap.end())
        RegEmbeddings[RegOperandIndex] = It->second;
      else
        handleMissingEntity(RegOperandName);
      ++RegOperandIndex;
    }
    return RegEmbeddings;
  };

  // Initialize register operand embeddings sections
  std::vector<Embedding> PhyRegEmbeddings = createRegisterEmbeddings(PhyRegMap);
  std::vector<Embedding> VirtRegEmbeddings =
      createRegisterEmbeddings(VirtRegMap);

  // Scale the vocabulary sections based on the provided weights
  auto scaleVocabSection = [](std::vector<Embedding> &Embeddings,
                              double Weight) {
    for (auto &Embedding : Embeddings)
      Embedding *= Weight;
  };
  scaleVocabSection(OpcodeEmbeddings, OpcWeight);
  scaleVocabSection(CommonOperandEmbeddings, CommonOperandWeight);
  scaleVocabSection(PhyRegEmbeddings, RegOperandWeight);
  scaleVocabSection(VirtRegEmbeddings, RegOperandWeight);

  std::vector<std::vector<Embedding>> Sections(
      static_cast<unsigned>(Section::MaxSections));
  Sections[static_cast<unsigned>(Section::Opcodes)] =
      std::move(OpcodeEmbeddings);
  Sections[static_cast<unsigned>(Section::CommonOperands)] =
      std::move(CommonOperandEmbeddings);
  Sections[static_cast<unsigned>(Section::PhyRegisters)] =
      std::move(PhyRegEmbeddings);
  Sections[static_cast<unsigned>(Section::VirtRegisters)] =
      std::move(VirtRegEmbeddings);

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

void MIRVocabulary::buildRegisterOperandMapping() {
  // Check if already built
  if (!RegisterOperandNames.empty())
    return;

  for (unsigned RC = 0; RC < TRI.getNumRegClasses(); ++RC) {
    const TargetRegisterClass *RegClass = TRI.getRegClass(RC);
    if (!RegClass)
      continue;

    // Get the register class name
    StringRef ClassName = TRI.getRegClassName(RegClass);
    RegisterOperandNames.push_back(ClassName.str());
  }
}

unsigned MIRVocabulary::getCommonOperandIndex(
    MachineOperand::MachineOperandType OperandType) const {
  assert(OperandType != MachineOperand::MO_Register &&
         "Expected non-register operand type");
  assert(OperandType > MachineOperand::MO_Register &&
         OperandType < MachineOperand::MO_Last && "Operand type out of bounds");
  return static_cast<unsigned>(OperandType) - 1;
}

unsigned MIRVocabulary::getRegisterOperandIndex(Register Reg) const {
  assert(!RegisterOperandNames.empty() && "Register operand mapping not built");
  assert(Reg.isValid() && "Invalid register; not expected here");
  assert((Reg.isPhysical() || Reg.isVirtual()) &&
         "Expected a physical or virtual register");

  const TargetRegisterClass *RegClass = nullptr;

  // For physical registers, use TRI to get minimal register class as a
  // physical register can belong to multiple classes. For virtual
  // registers, use MRI to uniquely identify the assigned register class.
  if (Reg.isPhysical())
    RegClass = TRI.getMinimalPhysRegClass(Reg);
  else
    RegClass = MRI.getRegClass(Reg);

  if (RegClass)
    return RegClass->getID();
  // Fallback for registers without a class (shouldn't happen)
  llvm_unreachable("Register operand without a valid register class");
  return 0;
}

Expected<MIRVocabulary> MIRVocabulary::createDummyVocabForTest(
    const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
    const MachineRegisterInfo &MRI, unsigned Dim) {
  assert(Dim > 0 && "Dimension must be greater than zero");

  float DummyVal = 0.1f;

  VocabMap DummyOpcMap, DummyOperandMap, DummyPhyRegMap, DummyVirtRegMap;

  // Process opcodes directly without creating temporary vocabulary
  for (unsigned Opcode = 0; Opcode < TII.getNumOpcodes(); ++Opcode) {
    std::string BaseOpcode = extractBaseOpcodeName(TII.getName(Opcode));
    if (DummyOpcMap.count(BaseOpcode) == 0) { // Only add if not already present
      DummyOpcMap[BaseOpcode] = Embedding(Dim, DummyVal);
      DummyVal += 0.1f;
    }
  }

  // Add common operands
  for (const auto &CommonOperandName : CommonOperandNames) {
    DummyOperandMap[CommonOperandName.str()] = Embedding(Dim, DummyVal);
    DummyVal += 0.1f;
  }

  // Process register classes directly
  for (unsigned RC = 0; RC < TRI.getNumRegClasses(); ++RC) {
    const TargetRegisterClass *RegClass = TRI.getRegClass(RC);
    if (!RegClass)
      continue;

    std::string ClassName = TRI.getRegClassName(RegClass);
    DummyPhyRegMap[ClassName] = Embedding(Dim, DummyVal);
    DummyVirtRegMap[ClassName] = Embedding(Dim, DummyVal);
    DummyVal += 0.1f;
  }

  // Create vocabulary directly without temporary instance
  return MIRVocabulary::create(
      std::move(DummyOpcMap), std::move(DummyOperandMap),
      std::move(DummyPhyRegMap), std::move(DummyVirtRegMap), TII, TRI, MRI);
}

//===----------------------------------------------------------------------===//
// MIR2VecVocabProvider and MIR2VecVocabLegacyAnalysis
//===----------------------------------------------------------------------===//

Expected<mir2vec::MIRVocabulary>
MIR2VecVocabProvider::getVocabulary(const Module &M) {
  VocabMap OpcVocab, CommonOperandVocab, PhyRegVocabMap, VirtRegVocabMap;

  if (Error Err = readVocabulary(OpcVocab, CommonOperandVocab, PhyRegVocabMap,
                                 VirtRegVocabMap))
    return std::move(Err);

  for (const auto &F : M) {
    if (F.isDeclaration())
      continue;

    if (auto *MF = MMI.getMachineFunction(F)) {
      auto &Subtarget = MF->getSubtarget();
      if (const auto *TII = Subtarget.getInstrInfo())
        if (const auto *TRI = Subtarget.getRegisterInfo())
          return mir2vec::MIRVocabulary::create(
              std::move(OpcVocab), std::move(CommonOperandVocab),
              std::move(PhyRegVocabMap), std::move(VirtRegVocabMap), *TII, *TRI,
              MF->getRegInfo());
    }
  }
  return createStringError(errc::invalid_argument,
                           "No machine functions found in module");
}

Error MIR2VecVocabProvider::readVocabulary(VocabMap &OpcodeVocab,
                                           VocabMap &CommonOperandVocab,
                                           VocabMap &PhyRegVocabMap,
                                           VocabMap &VirtRegVocabMap) {
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

  unsigned OpcodeDim = 0, CommonOperandDim = 0, PhyRegOperandDim = 0,
           VirtRegOperandDim = 0;
  if (auto Err = ir2vec::VocabStorage::parseVocabSection(
          "Opcodes", *ParsedVocabValue, OpcodeVocab, OpcodeDim))
    return Err;

  if (auto Err = ir2vec::VocabStorage::parseVocabSection(
          "CommonOperands", *ParsedVocabValue, CommonOperandVocab,
          CommonOperandDim))
    return Err;

  if (auto Err = ir2vec::VocabStorage::parseVocabSection(
          "PhysicalRegisters", *ParsedVocabValue, PhyRegVocabMap,
          PhyRegOperandDim))
    return Err;

  if (auto Err = ir2vec::VocabStorage::parseVocabSection(
          "VirtualRegisters", *ParsedVocabValue, VirtRegVocabMap,
          VirtRegOperandDim))
    return Err;

  // All sections must have the same embedding dimension
  if (!(OpcodeDim == CommonOperandDim && CommonOperandDim == PhyRegOperandDim &&
        PhyRegOperandDim == VirtRegOperandDim)) {
    return createStringError(
        errc::illegal_byte_sequence,
        "MIR2Vec vocabulary sections have different dimensions");
  }

  return Error::success();
}

char MIR2VecVocabLegacyAnalysis::ID = 0;
INITIALIZE_PASS_BEGIN(MIR2VecVocabLegacyAnalysis, "mir2vec-vocab-analysis",
                      "MIR2Vec Vocabulary Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_END(MIR2VecVocabLegacyAnalysis, "mir2vec-vocab-analysis",
                    "MIR2Vec Vocabulary Analysis", false, true)

StringRef MIR2VecVocabLegacyAnalysis::getPassName() const {
  return "MIR2Vec Vocabulary Analysis";
}

//===----------------------------------------------------------------------===//
// MIREmbedder and its subclasses
//===----------------------------------------------------------------------===//

std::unique_ptr<MIREmbedder> MIREmbedder::create(MIR2VecKind Mode,
                                                 const MachineFunction &MF,
                                                 const MIRVocabulary &Vocab) {
  switch (Mode) {
  case MIR2VecKind::Symbolic:
    return std::make_unique<SymbolicMIREmbedder>(MF, Vocab);
  }
  return nullptr;
}

Embedding MIREmbedder::computeEmbeddings(const MachineBasicBlock &MBB) const {
  Embedding MBBVector(Dimension, 0);

  // Get instruction info for opcode name resolution
  const auto &Subtarget = MF.getSubtarget();
  const auto *TII = Subtarget.getInstrInfo();
  if (!TII) {
    MF.getFunction().getContext().emitError(
        "MIR2Vec: No TargetInstrInfo available; cannot compute embeddings");
    return MBBVector;
  }

  // Process each machine instruction in the basic block
  for (const auto &MI : MBB) {
    // Skip debug instructions and other metadata
    if (MI.isDebugInstr())
      continue;
    MBBVector += computeEmbeddings(MI);
  }

  return MBBVector;
}

Embedding MIREmbedder::computeEmbeddings() const {
  Embedding MFuncVector(Dimension, 0);

  // Consider all reachable machine basic blocks in the function
  for (const auto *MBB : depth_first(&MF))
    MFuncVector += computeEmbeddings(*MBB);
  return MFuncVector;
}

SymbolicMIREmbedder::SymbolicMIREmbedder(const MachineFunction &MF,
                                         const MIRVocabulary &Vocab)
    : MIREmbedder(MF, Vocab) {}

std::unique_ptr<SymbolicMIREmbedder>
SymbolicMIREmbedder::create(const MachineFunction &MF,
                            const MIRVocabulary &Vocab) {
  return std::make_unique<SymbolicMIREmbedder>(MF, Vocab);
}

Embedding SymbolicMIREmbedder::computeEmbeddings(const MachineInstr &MI) const {
  // Skip debug instructions and other metadata
  if (MI.isDebugInstr())
    return Embedding(Dimension, 0);

  // Opcode embedding
  Embedding InstructionEmbedding = Vocab[MI.getOpcode()];

  // Add operand contributions
  for (const MachineOperand &MO : MI.operands())
    InstructionEmbedding += Vocab[MO];

  return InstructionEmbedding;
}

//===----------------------------------------------------------------------===//
// Printer Passes
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
  auto MIR2VecVocabOrErr = Analysis.getMIR2VecVocabulary(M);

  if (!MIR2VecVocabOrErr) {
    OS << "MIR2Vec Vocabulary Printer: Failed to get vocabulary - "
       << toString(MIR2VecVocabOrErr.takeError()) << "\n";
    return false;
  }

  auto &MIR2VecVocab = *MIR2VecVocabOrErr;
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

char MIR2VecPrinterLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(MIR2VecPrinterLegacyPass, "print-mir2vec",
                      "MIR2Vec Embedder Printer Pass", false, true)
INITIALIZE_PASS_DEPENDENCY(MIR2VecVocabLegacyAnalysis)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_END(MIR2VecPrinterLegacyPass, "print-mir2vec",
                    "MIR2Vec Embedder Printer Pass", false, true)

bool MIR2VecPrinterLegacyPass::runOnMachineFunction(MachineFunction &MF) {
  auto &Analysis = getAnalysis<MIR2VecVocabLegacyAnalysis>();
  auto VocabOrErr =
      Analysis.getMIR2VecVocabulary(*MF.getFunction().getParent());
  assert(VocabOrErr && "Failed to get MIR2Vec vocabulary");
  auto &MIRVocab = *VocabOrErr;

  auto Emb = mir2vec::MIREmbedder::create(MIR2VecEmbeddingKind, MF, MIRVocab);
  if (!Emb) {
    OS << "Error creating MIR2Vec embeddings for function " << MF.getName()
       << "\n";
    return false;
  }

  OS << "MIR2Vec embeddings for machine function " << MF.getName() << ":\n";
  OS << "Machine Function vector: ";
  Emb->getMFunctionVector().print(OS);

  OS << "Machine basic block vectors:\n";
  for (const MachineBasicBlock &MBB : MF) {
    OS << "Machine basic block: " << MBB.getFullName() << ":\n";
    Emb->getMBBVector(MBB).print(OS);
  }

  OS << "Machine instruction vectors:\n";
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // Skip debug instructions as they are not
      // embedded
      if (MI.isDebugInstr())
        continue;

      OS << "Machine instruction: ";
      MI.print(OS);
      Emb->getMInstVector(MI).print(OS);
    }
  }

  return false;
}

MachineFunctionPass *llvm::createMIR2VecPrinterLegacyPass(raw_ostream &OS) {
  return new MIR2VecPrinterLegacyPass(OS);
}
