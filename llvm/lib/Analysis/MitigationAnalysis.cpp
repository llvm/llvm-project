//===--- MitigationAnalysis.cpp - Emit LLVM Code from ASTs for a Module ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This processes mitigation metadata to create a report on enablement
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MitigationAnalysis.h"
#include "llvm/Support/MitigationMarker.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

static cl::opt<MitigationAnalysisSummaryType> MitigationAnalysisSummary(
    "mitigation-summary", cl::Hidden,
    cl::init(MitigationAnalysisSummaryType::NONE),
    cl::desc("Enable exporting mitigation analysis information"),
    cl::values(
        clEnumValN(MitigationAnalysisSummaryType::NONE, "none",
                   "Do not export mitigation analysis information"),
        clEnumValN(MitigationAnalysisSummaryType::JSON, "json",
                   "Export mitigation analysis information in JSON format"),
        clEnumValN(MitigationAnalysisSummaryType::EMBED, "embed",
                   "Embed mitigation analysis information in the binary")));

static cl::opt<std::string> MitigationAnalysisOutputRoot(
    "mitigation-analysis-output-root",
    cl::desc("Folder to write mitigation analysis outputs"),
    cl::value_desc("MitigationOutputDirectory"), cl::init("/tmp"));

static cl::opt<std::string> OutputUnitName("output-unit-name", cl::init(""),
                                           cl::ZeroOrMore,
                                           cl::value_desc("OutputUnitName"),
                                           cl::desc("Output unit name"),
                                           cl::Hidden);

static cl::opt<std::string>
    LibCXXPrefix("mitigation-libcxx-prefix", cl::init("std::"),
                 cl::value_desc("LibCXXPrefix"),
                 cl::desc("Namespace prefix for LibC++ hardening functions"),
                 cl::Hidden);

MitigationAnalysisOptions llvm::getMitigationAnalysisOptions() {
  return MitigationAnalysisOptions(MitigationAnalysisSummary.getValue(),
                                   StringRef(MitigationAnalysisOutputRoot),
                                   StringRef(OutputUnitName.empty()
                                                 ? std::string("unknown")
                                                 : OutputUnitName),
                                   StringRef(LibCXXPrefix));
}

/**
 *
 * Mitigation metadata is stored in 2 sections: __mitigation and __mitigationstr
 *
 * Mitigation Section contains arrays of Packed Mitigation Data
 *   [0xFFFF][MitigationBits:uint16]
 *   [FUNC_HASH:20 bytes]
 *   [SRC_HASH:20 bytes]
 *
 * Mitigation String Section contains arrays of Packed Mitigation String
 *   [HASH:20 bytes]
 *   [NULL-TERMINATED STRING:var-length]
 *
 * Since this pass is expected to run for multiple modules during LTO,
 * each array is prefixed with a header. The BufferSize is the length
 * of the header (8-bytes) + length of array in bytes
 *   [0xDEADBEEF][BufferSize:uint32]
 *
 * The metadata leads to the following size regressions.
 *   44-bytes for the function's data
 *   20-bytes for the function hash + X-bytes for string
 *   20-bytes for the source path + Y-bytes for string
 */
static constexpr uint32_t kMitigationTag = 0xDEADBEEF;
static constexpr size_t kMitigationHashBytes = 20;
static_assert(kMitigationHashBytes <= 20,
              "kMitigationHashBytes must be <= 20 since it holds SHA1 hash");

AnalysisKey MitigationAnalysisPass::Key;

enum class MitigationState { NotAvailable, Disabled, Enabled };

static const std::unordered_map<MitigationState, std::string> mapStateToString =
    {
        {MitigationState::NotAvailable, "N/A"},
        {MitigationState::Disabled, "Disabled"},
        {MitigationState::Enabled, "Enabled"},
};

class MitigationInfo {
public:
  llvm::DenseMap<MitigationKey, MitigationState> Enablement;

  // This is not included in the MitigationKey list since determined during this
  // pass
  MitigationState LibCppHardeningMode = MitigationState::NotAvailable;

  std::string SourceMapping;
  std::string TypeSignature;
  uint64_t TypeId = 0;
  std::string Function;
  std::string Module;

  MitigationInfo() {
    // Initialize all mitigation states to NotAvailable
    for (size_t Mitigation = 0;
         Mitigation <
         static_cast<size_t>(llvm::MitigationKey::MITIGATION_KEY_MAX);
         ++Mitigation) {
      Enablement[MitigationKey(Mitigation)] = MitigationState::NotAvailable;
    }
  }

  void Pack(std::vector<uint8_t> &MitigationPacked,
            std::map<std::string, std::array<uint8_t, kMitigationHashBytes>>
                &Hashes) const noexcept {
    // Bit 0-1: Auto Var Init
    std::uint16_t Packed =
        MitigationStateToPacked(llvm::MitigationKey::AUTO_VAR_INIT, 0);
    // Bit 2-3: Stack Clash
    Packed |=
        MitigationStateToPacked(llvm::MitigationKey::STACK_CLASH_PROTECTION, 2);
    // Bit 4-5: Stack Protector
    Packed |= MitigationStateToPacked(llvm::MitigationKey::STACK_PROTECTOR, 4);
    // Bit 6-7: Stack Protector Strong
    Packed |=
        MitigationStateToPacked(llvm::MitigationKey::STACK_PROTECTOR_STRONG, 6);
    // Bit 8-9: LibCXX Hardening
    Packed |= MitigationStateToPacked(LibCppHardeningMode, 8);
    // Bit 10-11: CFI ICall
    Packed |= MitigationStateToPacked(llvm::MitigationKey::CFI_ICALL, 10);
    // Bit 12-13: CFI VCall
    Packed |= MitigationStateToPacked(llvm::MitigationKey::CFI_VCALL, 12);
    // Bit 14-15: Reserved

    auto MitigationPackedOldSize = MitigationPacked.size();
    MitigationPacked.resize(MitigationPackedOldSize + sizeof(uint16_t) * 2);
    // 0xFFFF represents the reserved bits for future use
    support::endian::write16(MitigationPacked.data() + MitigationPackedOldSize,
                             0xFFFF, llvm::endianness::little);
    support::endian::write16(MitigationPacked.data() + MitigationPackedOldSize +
                                 sizeof(uint16_t),
                             Packed, llvm::endianness::little);

    SHA1 Hasher;
    Hasher.update(Function);
    auto FuncDigest = Hasher.final();
    for (size_t i = 0; i < kMitigationHashBytes; ++i)
      MitigationPacked.push_back(FuncDigest[i]);

    Hasher.init();
    Hasher.update(SourceMapping);
    auto SrcDigest = Hasher.final();
    for (size_t i = 0; i < kMitigationHashBytes; ++i)
      MitigationPacked.push_back(SrcDigest[i]);

    // Export string -> hash mappings
    Hashes[Function] = FuncDigest;
    Hashes[SourceMapping] = SrcDigest;
  }

private:
  std::uint16_t MitigationStateToPacked(llvm::MitigationKey key,
                                        size_t offset) const {
    auto it = Enablement.find(key);
    return it != Enablement.end()
               ? MitigationStateToPacked(it->second, offset)
               : MitigationStateToPacked(MitigationState::NotAvailable, offset);
  }

  std::uint16_t MitigationStateToPacked(MitigationState state,
                                        size_t offset) const {
    switch (state) {
    case MitigationState::NotAvailable:
      return uint16_t(0b00) << offset;
    case MitigationState::Disabled:
      return uint16_t(0b01) << offset;
    case MitigationState::Enabled:
      return uint16_t(0b10) << offset;
    }
  }
};

class ModuleMitigationInfo {
public:
  struct MitigationEligibility {
    std::size_t Eligible;
    std::size_t Enabled;
  };

  llvm::DenseMap<MitigationKey, MitigationEligibility> Eligibility;
  MitigationEligibility LibCppHardeningMode;
  std::size_t TotalFunctions;

  ModuleMitigationInfo() : TotalFunctions(0) {
    for (size_t Mitigation = 0;
         Mitigation <
         static_cast<size_t>(llvm::MitigationKey::MITIGATION_KEY_MAX);
         ++Mitigation) {
      Eligibility[MitigationKey(Mitigation)] = {0, 0};
    }
    LibCppHardeningMode = {0, 0};
  }

  /// Convert a ModuleMitigationInfo struct to a JSON object.
  json::Object ModuleInfoToJson(StringRef ModuleName) const noexcept {
    json::Object Object;

    WriteMitigationToJSON(Object, "eligible_auto_var_init",
                          "enabled_auto_var_init",
                          llvm::MitigationKey::AUTO_VAR_INIT);
    WriteMitigationToJSON(Object, "eligible_stack_clash_protection",
                          "enabled_stack_clash_protection",
                          llvm::MitigationKey::STACK_CLASH_PROTECTION);
    WriteMitigationToJSON(Object, "eligible_stack_protector",
                          "enabled_stack_protector",
                          llvm::MitigationKey::STACK_PROTECTOR);
    WriteMitigationToJSON(Object, "eligible_stack_protector_strong",
                          "enabled_stack_protector_strong",
                          llvm::MitigationKey::STACK_PROTECTOR_STRONG);
    WriteMitigationToJSON(Object, "eligible_stack_protector_all",
                          "enabled_stack_protector_all",
                          llvm::MitigationKey::STACK_PROTECTOR_ALL);
    WriteMitigationToJSON(Object, "eligible_cfi_icall", "enabled_cfi_icall",
                          llvm::MitigationKey::CFI_ICALL);
    WriteMitigationToJSON(Object, "eligible_cfi_vcall", "enabled_cfi_vcall",
                          llvm::MitigationKey::CFI_VCALL);
    WriteMitigationToJSON(Object, "eligible_cfi_nvcall", "enabled_cfi_nvcall",
                          llvm::MitigationKey::CFI_NVCALL);

    Object["eligible_libcpp_hardening"] = LibCppHardeningMode.Eligible;
    Object["enabled_libcpp_hardening"] = LibCppHardeningMode.Enabled;

    Object["total_functions"] = TotalFunctions;
    Object["module"] = ModuleName;
    return Object;
  }

  void UpdateModuleInfo(const MitigationInfo &Info) noexcept {
    TotalFunctions++;

    UpdateModuleInfoEligibility(Info, MitigationKey::AUTO_VAR_INIT);
    UpdateModuleInfoEligibility(Info, MitigationKey::STACK_CLASH_PROTECTION);
    UpdateModuleInfoEligibility(Info, MitigationKey::STACK_PROTECTOR);
    UpdateModuleInfoEligibility(Info, MitigationKey::STACK_PROTECTOR_STRONG);
    UpdateModuleInfoEligibility(Info, MitigationKey::STACK_PROTECTOR_ALL);
    UpdateModuleInfoEligibility(Info, MitigationKey::CFI_ICALL);
    UpdateModuleInfoEligibility(Info, MitigationKey::CFI_VCALL);
    UpdateModuleInfoEligibility(Info, MitigationKey::CFI_NVCALL);

    LibCppHardeningMode.Eligible +=
        (Info.LibCppHardeningMode != MitigationState::NotAvailable);
    LibCppHardeningMode.Enabled +=
        (Info.LibCppHardeningMode == MitigationState::Enabled);
  }

private:
  void WriteMitigationToJSON(json::Object &Object,
                             const llvm::StringRef Eligible,
                             const llvm::StringRef Enabled,
                             llvm::MitigationKey Key) const noexcept {
    auto it = Eligibility.find(Key);
    if (it != Eligibility.end()) {
      Object[Eligible] = it->second.Eligible;
      Object[Enabled] = it->second.Enabled;
    } else {
      Object[Eligible] = 0;
      Object[Enabled] = 0;
    }
  }

  void UpdateModuleInfoEligibility(const MitigationInfo &Info,
                                   llvm::MitigationKey Key) noexcept {
    auto it = Info.Enablement.find(Key);
    if (it != Info.Enablement.end()) {
      Eligibility[Key].Eligible +=
          (it->second != MitigationState::NotAvailable);
      Eligibility[Key].Enabled += (it->second == MitigationState::Enabled);
    }
  }
};

/// Retrieve the first valid source path for the given function.
inline static std::string getFunctionSourcePath(const Function &F) {
  if (const DISubprogram *SP = F.getSubprogram()) {
    StringRef Dir = SP->getDirectory();
    StringRef File = SP->getFilename();
    if (!Dir.empty() && !File.empty())
      return (Twine(Dir) + "/" + File).str();
  }
  return "(unknown)";
}

/// Extract the first function type signature (that doesn't end with
/// .generalized) from metadata in Function F.
static std::string getFirstFunctionTypeSignature(Function &F) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  F.getAllMetadata(MDs);

  for (const auto &MD : MDs) {
    if (MD.first != LLVMContext::MD_type)
      continue;

    if (MDNode *Node = MD.second) {
      if (Node->getNumOperands() <= 1)
        continue;

      auto *Str = dyn_cast<MDString>(Node->getOperand(1));
      if (!Str)
        continue;

      StringRef Signature = Str->getString();
      if (!Signature.ends_with(".generalized"))
        return Signature.str();
    }
  }
  return "";
}

/// Extract a type ID from MD_type metadata in Function F (0 if not found).
static uint64_t getFunctionTypeId(Function &F) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  F.getAllMetadata(MDs);

  for (const auto &MD : MDs) {
    if (MD.first != LLVMContext::MD_type)
      continue;

    MDNode *Node = MD.second;
    if (!Node || Node->getNumOperands() <= 1)
      continue;

    auto *MDInt = dyn_cast<ConstantAsMetadata>(Node->getOperand(1));
    if (!MDInt)
      continue;

    if (auto *CI = dyn_cast<ConstantInt>(MDInt->getValue()))
      return CI->getZExtValue();
  }
  return 0;
}

/// Detect the libcpp hardening mode from calls in the given function.
static MitigationState
detectCXXHardeningMode(Function &F, StringMap<bool> HardenedCXXFunctions) {
  for (Instruction &I : instructions(F)) {
    auto *CallInstruct = dyn_cast<CallInst>(&I);
    if (!CallInstruct)
      continue;

    Function *CalledFunction = CallInstruct->getCalledFunction();
    if (!CalledFunction)
      continue;

    auto Iter = HardenedCXXFunctions.find(CalledFunction->getName());
    if (Iter != HardenedCXXFunctions.end())
      return Iter->second ? MitigationState::Enabled
                          : MitigationState::Disabled;
  }
  return MitigationState::NotAvailable;
}

static inline StringRef getMitigationSectionName(const Module &M) noexcept {
  Triple T(M.getTargetTriple());
  return T.isOSBinFormatMachO() ? "__DATA,__mitigation" : ".mitigation";
}

static inline StringRef getMitigationStrsSectionName(const Module &M) noexcept {
  Triple T(M.getTargetTriple());
  return T.isOSBinFormatMachO() ? "__DATA,__mitigationstr" : ".mitigationstr";
}

static void writeMitigationMetadataSection(
    Module &M, std::vector<uint8_t> &PackedMitigationData,
    std::map<std::string, std::array<uint8_t, kMitigationHashBytes>>
        &MitigationHashes) noexcept {
  LLVMContext &Context = M.getContext();

  // Update PackedMitigationData with header
  support::endian::write32(PackedMitigationData.data(), kMitigationTag,
                           llvm::endianness::little);
  uint32_t MitigationDataSize = PackedMitigationData.size();
  support::endian::write32(PackedMitigationData.data() + sizeof(kMitigationTag),
                           MitigationDataSize, llvm::endianness::little);

  auto *Int8Ty = Type::getInt8Ty(Context);
  auto *MitigationArrTy = ArrayType::get(Int8Ty, PackedMitigationData.size());
  auto *MitigationArrConst =
      ConstantDataArray::get(Context, PackedMitigationData);
  auto *GV =
      new GlobalVariable(M, MitigationArrTy, true, GlobalValue::PrivateLinkage,
                         MitigationArrConst, "mitigationdata");
  GV->setSection(getMitigationSectionName(M));
  GV->setAlignment(Align(1));
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  appendToUsed(M, {GV});

  // Add string map
  std::vector<uint8_t> MitigationHashArr;

  // Include length of header
  uint32_t MitHashArrSize = sizeof(kMitigationTag) + sizeof(MitHashArrSize);
  for (const auto &[Str, Hash] : MitigationHashes) {
    MitHashArrSize += kMitigationHashBytes + Str.size() + 1;
  }
  MitigationHashArr.reserve(sizeof(kMitigationTag) + sizeof(MitHashArrSize) +
                            MitHashArrSize);

  // Marker + Length
  MitigationHashArr.resize(sizeof(kMitigationTag) + sizeof(MitHashArrSize));
  support::endian::write32(MitigationHashArr.data(), kMitigationTag,
                           llvm::endianness::little);
  support::endian::write32(MitigationHashArr.data() + sizeof(kMitigationTag),
                           MitHashArrSize, llvm::endianness::little);

  for (const auto &[Str, Hash] : MitigationHashes) {
    for (size_t i = 0; i < kMitigationHashBytes; ++i)
      MitigationHashArr.push_back(Hash[i]);

    for (size_t i = 0; i < Str.size(); i++)
      MitigationHashArr.push_back(Str[i]);
    MitigationHashArr.push_back('\0'); // Ensure NULL terminated
  }
  auto *StrMapArrTy = ArrayType::get(Int8Ty, MitigationHashArr.size());
  auto *StrMapArrConst = ConstantDataArray::get(Context, MitigationHashArr);
  GV = new GlobalVariable(M, StrMapArrTy, true, GlobalValue::PrivateLinkage,
                          StrMapArrConst, "mitigationdata.strs");
  GV->setSection(getMitigationStrsSectionName(M));
  GV->setAlignment(MaybeAlign(1));
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  appendToUsed(M, {GV});
}

PreservedAnalyses MitigationAnalysisPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  if (Options.SummaryType == MitigationAnalysisSummaryType::NONE)
    return PreservedAnalyses::all();

  ModuleMitigationInfo ModuleInfo;
  std::vector<uint8_t> PackedMitigationData(8);
  std::map<std::string, std::array<uint8_t, kMitigationHashBytes>>
      PackedMitigationStrings;

  StringMap<bool> HardenedCXXFunctions;
  getHardenedAccessFunctions(M, HardenedCXXFunctions);

  const auto &MitigationToString = GetMitigationMetadataMapping();

  LLVMContext &Context = M.getContext();
  for (Function &F : M) {
    // If not defined in this module or part of LLVM, skip
    if (F.isDeclaration() || F.isIntrinsic())
      continue;

    MitigationInfo Info;
    Info.Module = Options.OutputUnitName;
    Info.Function = F.getName();

    for (const auto &[MitigationKey, MitigationValue] : MitigationToString) {
      unsigned KindID = Context.getMDKindID(MitigationValue);

      MDNode *MD = F.getMetadata(KindID);
      if (!MD)
        continue;

      if (MD->getNumOperands() != 1)
        continue;

      auto *ConstAsMeta = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
      if (!ConstAsMeta)
        continue;

      auto *Value = ConstAsMeta->getValue();
      auto ValueToState = Value->isOneValue() == 1 ? MitigationState::Enabled
                                                   : MitigationState::Disabled;
      Info.Enablement[MitigationKey] = ValueToState;
    }

    Info.LibCppHardeningMode = detectCXXHardeningMode(F, HardenedCXXFunctions);

    Info.SourceMapping = getFunctionSourcePath(F);
    Info.TypeSignature = getFirstFunctionTypeSignature(F);
    Info.TypeId = getFunctionTypeId(F);

    switch (MitigationAnalysisSummary) {
    case MitigationAnalysisSummaryType::EMBED:
      Info.Pack(PackedMitigationData, PackedMitigationStrings);
      break;
    case MitigationAnalysisSummaryType::JSON:
      ModuleInfo.UpdateModuleInfo(Info);
      break;
    case MitigationAnalysisSummaryType::NONE:
      // No-Op
      break;
    }
  }

  switch (MitigationAnalysisSummary) {
  case MitigationAnalysisSummaryType::EMBED:
    writeMitigationMetadataSection(M, PackedMitigationData,
                                   PackedMitigationStrings);
    break;
  case MitigationAnalysisSummaryType::JSON:
    writeJsonToFile(
        json::Value(ModuleInfo.ModuleInfoToJson(Options.OutputUnitName)));
    break;
  case MitigationAnalysisSummaryType::NONE:
    llvm_unreachable("Pass should early exit if summary type is None");
  }
  return PreservedAnalyses::all();
}

/// Write the given JSON string to file with a lock. On error, prints to stderr.
void MitigationAnalysisPass::writeJsonToFile(
    const llvm::json::Value &JsonValue) {
  std::string JsonString = formatv("{0:2}\n", JsonValue);
  if (JsonString.size() == 1)
    return;

  // Create output directory if it doesn't exist.
  sys::fs::create_directories(Options.OutputRoot, true);

  std::string FileName =
      Options.OutputUnitName == "unknown"
          ? std::string("mitigation_info.json")
          : formatv("mitigation_info-{0}.json", Options.OutputUnitName);

  std::string FilePath = Options.OutputRoot + "/" + FileName;

  std::error_code ErrCode;
  raw_fd_ostream OutputStream(FilePath, ErrCode, sys::fs::CD_OpenAlways,
                              sys::fs::FA_Read | sys::fs::FA_Write,
                              sys::fs::OF_Text | sys::fs::OF_UpdateAtime |
                                  sys::fs::OF_Append);
  if (ErrCode) {
    errs() << formatv("Couldn't write to {0}: {1}\n", FilePath,
                      ErrCode.message());
    return;
  }

  if (auto Lock = OutputStream.lock()) {
    OutputStream << JsonString;
    if (OutputStream.has_error())
      errs() << formatv("Couldn't write to {0}: Failed writing JSON\n",
                        FilePath);

  } else {
    errs() << formatv("Couldn't write to {0}: Couldn't acquire lock\n",
                      FilePath);
  }
}

void MitigationAnalysisPass::getHardenedAccessFunctions(
    Module &M, StringMap<bool> &HardenedCXXFunctions) {
  static StringRef OperatorFuncName = "operator[]";
  static StringRef SizeFuncName = "size";
  static StringRef TrapFuncName = "llvm.trap";

  for (Function &F : M) {
    StringRef Name = F.getName();

    // Screen for functions with parameters: `(ptr, size_t)`
    auto *FType = F.getFunctionType();
    if (FType->getNumParams() != 2)
      continue;
    auto *ParamType = FType->getParamType(1);
    if (!ParamType->isIntegerTy(sizeof(size_t) * 8))
      continue;
    auto *ThisType = dyn_cast<PointerType>(FType->getParamType(0));
    if (!ThisType)
      continue;

    // Demangle the base name of the function
    if (!compareDemangledFunctionName(Name, OperatorFuncName))
      continue;

    // Be pessimistic about checking size
    HardenedCXXFunctions[Name.str()] = false;

    for (Instruction &I : instructions(F)) {
      auto *CallInstruct = dyn_cast<CallInst>(&I);
      if (!CallInstruct)
        continue;

      Function *CalledFunction = CallInstruct->getCalledFunction();
      if (!CalledFunction)
        continue;

      bool IsTrapBuiltin =
          strncmp(TrapFuncName.data(), CalledFunction->getName().data(),
                  TrapFuncName.size()) == 0;

      // Check if `size()` or `llvm.trap` called
      if (IsTrapBuiltin || compareDemangledFunctionName(
                               CalledFunction->getName(), SizeFuncName)) {
        HardenedCXXFunctions[Name.str()] = true;
        break;
      }
    }
  }
}

bool MitigationAnalysisPass::compareDemangledFunctionName(
    StringRef MangledName, StringRef CompareName) {
  llvm::ItaniumPartialDemangler Demangler;
  if (Demangler.partialDemangle(MangledName.data()))
    return false;
  if (!Demangler.isFunction())
    return false;

  size_t DemangleBufLen = 0;
  char *DemangleBuf = Demangler.getFunctionBaseName(nullptr, &DemangleBufLen);
  if (DemangleBuf == nullptr || DemangleBufLen == 0)
    return false;

  bool CompareResult =
      strncmp(CompareName.data(), DemangleBuf, CompareName.size()) == 0;

  free(DemangleBuf);

  if (CompareResult && !Options.LibCXXPrefix.empty()) {
    char *DemangleBuf = Demangler.getFunctionName(nullptr, &DemangleBufLen);
    if (DemangleBuf == nullptr || DemangleBufLen == 0)
      return false;

    // Check if the function name starts with LibCXXPrefix
    CompareResult = strncmp(Options.LibCXXPrefix.data(), DemangleBuf,
                            Options.LibCXXPrefix.size()) == 0;

    free(DemangleBuf);
  }

  return CompareResult;
}
