#include "llvm/Analysis/MitigationAnalysis.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

AnalysisKey MitigationAnalysis::Key;

// Add a command line flag for the module name
static cl::opt<std::string>
    ClOutputModuleName("mitigation-analysis-dso-name", cl::Optional,
                       cl::desc("DSO name for the module"), cl::init(""));

enum class MitigationState { Ineligible, EligibleDisabled, EligibleEnabled };

static const std::unordered_map<MitigationState, std::string> mapStateToString =
    {
        {MitigationState::Ineligible, "N/A"},
        {MitigationState::EligibleDisabled, "Disabled"},
        {MitigationState::EligibleEnabled, "Enabled"},
};

struct ModuleMitigationInfo {
  std::size_t eligable_auto_var_init = 0;
  std::size_t enabled_auto_var_init = 0;

  std::size_t eligable_cfi_icall = 0;
  std::size_t enabled_cfi_icall = 0;

  std::size_t eligable_cfi_vcall = 0;
  std::size_t enabled_cfi_vcall = 0;

  std::size_t eligable_cfi_nvcall = 0;
  std::size_t enabled_cfi_nvcall = 0;

  std::size_t eligable_stack_clash_protection = 0;
  std::size_t enabled_stack_clash_protection = 0;

  std::size_t eligable_stack_protector = 0;
  std::size_t enabled_stack_protector = 0;

  std::size_t eligable_stack_protector_strong = 0;
  std::size_t enabled_stack_protector_strong = 0;

  std::size_t eligable_stack_protector_all = 0;
  std::size_t enabled_stack_protector_all = 0;

  std::size_t eligable_libcpp_hardening_mode = 0;
  std::size_t enabled_libcpp_hardening_mode = 0;

  std::size_t total_functions = 0;
};

struct MitigationInfo {
  MitigationState auto_var_init = MitigationState::Ineligible;
  MitigationState cfi_icall = MitigationState::Ineligible;
  MitigationState cfi_vcall = MitigationState::Ineligible;
  MitigationState cfi_nvcall = MitigationState::Ineligible;
  MitigationState stack_clash_protection = MitigationState::Ineligible;
  MitigationState stack_protector = MitigationState::Ineligible;
  MitigationState stack_protector_strong = MitigationState::Ineligible;
  MitigationState stack_protector_all = MitigationState::Ineligible;
  MitigationState libcpp_hardening_mode = MitigationState::Ineligible;
  std::string source_mapping = "(unknown)";
  std::string type_signature = "??";
  uint64_t type_id = 0;
  std::string function;
  std::string gmodule;
};

/// Convert an integer value (0 or 1) to the appropriate MitigationState.
static inline MitigationState valToState(int value) {
  switch (value) {
  case 0:
    return MitigationState::EligibleDisabled;
  case 1:
    return MitigationState::EligibleEnabled;
  default:
    return MitigationState::Ineligible;
  }
}

static void updateModuleInfo(ModuleMitigationInfo &moduleInfo,
                             const MitigationInfo &info) {
  moduleInfo.total_functions++;

  moduleInfo.eligable_auto_var_init +=
      (info.auto_var_init != MitigationState::Ineligible);
  moduleInfo.enabled_auto_var_init +=
      (info.auto_var_init == MitigationState::EligibleEnabled);

  moduleInfo.eligable_cfi_icall +=
      (info.cfi_icall != MitigationState::Ineligible);
  moduleInfo.enabled_cfi_icall +=
      (info.cfi_icall == MitigationState::EligibleEnabled);

  moduleInfo.eligable_cfi_vcall +=
      (info.cfi_vcall != MitigationState::Ineligible);
  moduleInfo.enabled_cfi_vcall +=
      (info.cfi_vcall == MitigationState::EligibleEnabled);

  moduleInfo.eligable_cfi_nvcall +=
      (info.cfi_nvcall != MitigationState::Ineligible);
  moduleInfo.enabled_cfi_nvcall +=
      (info.cfi_nvcall == MitigationState::EligibleEnabled);

  moduleInfo.eligable_stack_clash_protection +=
      (info.stack_clash_protection != MitigationState::Ineligible);
  moduleInfo.enabled_stack_clash_protection +=
      (info.stack_clash_protection == MitigationState::EligibleEnabled);

  moduleInfo.eligable_stack_protector +=
      (info.stack_protector != MitigationState::Ineligible);
  moduleInfo.enabled_stack_protector +=
      (info.stack_protector == MitigationState::EligibleEnabled);

  moduleInfo.eligable_stack_protector_strong +=
      (info.stack_protector_strong != MitigationState::Ineligible);
  moduleInfo.enabled_stack_protector_strong +=
      (info.stack_protector_strong == MitigationState::EligibleEnabled);

  moduleInfo.eligable_stack_protector_all +=
      (info.stack_protector_all != MitigationState::Ineligible);
  moduleInfo.enabled_stack_protector_all +=
      (info.stack_protector_all == MitigationState::EligibleEnabled);

  moduleInfo.eligable_libcpp_hardening_mode +=
      (info.libcpp_hardening_mode != MitigationState::Ineligible);
  moduleInfo.enabled_libcpp_hardening_mode +=
      (info.libcpp_hardening_mode == MitigationState::EligibleEnabled);
}

/// Print out fields in MitigationInfo for debugging/verification purposes.
#ifndef NDEBUG
static void printInfo(const MitigationInfo &info) {
  dbgs() << "module: " << info.gmodule << "\n";
  dbgs() << "function: " << info.function << "\n";
  dbgs() << "source_location: " << info.source_mapping << "\n";
  dbgs() << "auto-var-init: " << mapStateToString.at(info.auto_var_init)
         << "\n";
  dbgs() << "cfi-icall: " << mapStateToString.at(info.cfi_icall) << "\n";
  dbgs() << "cfi-vcall: " << mapStateToString.at(info.cfi_vcall) << "\n";
  dbgs() << "cfi-nvcall: " << mapStateToString.at(info.cfi_nvcall) << "\n";
  dbgs() << "stack-clash-protection: "
         << mapStateToString.at(info.stack_clash_protection) << "\n";
  dbgs() << "stack-protector: " << mapStateToString.at(info.stack_protector)
         << "\n";
  dbgs() << "stack-protector-strong: "
         << mapStateToString.at(info.stack_protector_strong) << "\n";
  dbgs() << "stack-protector-all: "
         << mapStateToString.at(info.stack_protector_all) << "\n";
  dbgs() << "libcpp-hardening-mode: "
         << mapStateToString.at(info.libcpp_hardening_mode) << "\n";
  dbgs() << "type_signature: " << info.type_signature << "\n";
  dbgs() << "type_id: " << info.type_id << "\n\n";
}
#endif

/// Convert a mitigation key + integer value into the appropriate field
/// of MitigationInfo. This replaces a long chain of if/else statements.
static void keyAndValueToInfo(MitigationInfo &info, StringRef key, int value) {
  static constexpr struct {
    const StringRef Key;
    MitigationState MitigationInfo::*Field;
  } Mappings[] = {
      {StringRef("auto-var-init"), &MitigationInfo::auto_var_init},
      {StringRef("cfi-icall"), &MitigationInfo::cfi_icall},
      {StringRef("cfi-vcall"), &MitigationInfo::cfi_vcall},
      {StringRef("cfi-nvcall"), &MitigationInfo::cfi_nvcall},
      {StringRef("stack-clash-protection"),
       &MitigationInfo::stack_clash_protection},
      {StringRef("stack-protector"), &MitigationInfo::stack_protector},
      {StringRef("stack-protector-strong"),
       &MitigationInfo::stack_protector_strong},
      {StringRef("stack-protector-all"), &MitigationInfo::stack_protector_all},
      {StringRef("libcpp-hardening-mode"),
       &MitigationInfo::libcpp_hardening_mode},
  };

  for (const auto &Mapping : Mappings) {
    if (key == Mapping.Key) {
      info.*(Mapping.Field) = valToState(value);
      break;
    }
  }
}

/// Retrieve the first valid source path for the given function.
static std::string getFunctionSourcePath(const Function &F) {
  if (const DISubprogram *SP = F.getSubprogram()) {
    std::string Dir = SP->getDirectory().str();
    std::string File = SP->getFilename().str();
    unsigned Line = SP->getLine();
    if (!Dir.empty() && !File.empty())
      return Dir + "/" + File + ":" + std::to_string(Line);
  }
  return "(unknown)";
}

/// Write the given JSON string to file with a lock. On error, prints to stderr.
static void writeJsonToFile(const std::string &jsonString,
                            const std::string &fileName,
                            const std::string &errorMsg) {
  std::error_code errCode;
  raw_fd_ostream OutputStream(fileName, errCode, sys::fs::CD_OpenAlways,
                              sys::fs::FA_Read | sys::fs::FA_Write,
                              sys::fs::OF_Text | sys::fs::OF_UpdateAtime |
                                  sys::fs::OF_Append);
  if (errCode) {
    errs() << errorMsg << "\n";
    errs() << errCode.message() << "\n";
    return;
  }

  if (auto lock = OutputStream.lock()) {
    OutputStream << jsonString << "\n";
    if (OutputStream.has_error()) {
      errs() << errorMsg << "\n";
      errs() << jsonString << "\n";
    }
  } else {
    errs() << errorMsg << "\n";
    errs() << "Couldn't acquire lock for " << fileName << "\n";
  }
}

/// Convert a MitigationInfo struct to a JSON object.
static json::Object infoToJson(const MitigationInfo &info) {
  json::Object object;
  object["auto_var_init"] = mapStateToString.at(info.auto_var_init);
  object["cfi_icall"] = mapStateToString.at(info.cfi_icall);
  object["cfi_vcall"] = mapStateToString.at(info.cfi_vcall);
  object["cfi_nvcall"] = mapStateToString.at(info.cfi_nvcall);
  object["stack_clash_protection"] =
      mapStateToString.at(info.stack_clash_protection);
  object["stack_protector"] = mapStateToString.at(info.stack_protector);
  object["stack_protector_strong"] =
      mapStateToString.at(info.stack_protector_strong);
  object["stack_protector_all"] = mapStateToString.at(info.stack_protector_all);
  object["libcpp_hardening_mode"] =
      mapStateToString.at(info.libcpp_hardening_mode);
  object["source_mapping"] = info.source_mapping;
  object["function"] = info.function;
  object["type_signature"] = info.type_signature;
  object["type_id"] = (uint64_t)info.type_id;
  object["module"] = info.gmodule;
  return object;
}

/// Convert a ModuleMitigationInfo struct to a JSON object.
static json::Object moduleInfoToJson(const ModuleMitigationInfo &moduleInfo) {
  json::Object object;

  object["enabled_auto_var_init"] = moduleInfo.enabled_auto_var_init;
  object["eligable_cfi_icall"] = moduleInfo.eligable_cfi_icall;
  object["enabled_cfi_icall"] = moduleInfo.enabled_cfi_icall;
  object["eligable_cfi_vcall"] = moduleInfo.eligable_cfi_vcall;
  object["enabled_cfi_vcall"] = moduleInfo.enabled_cfi_vcall;
  object["eligable_cfi_nvcall"] = moduleInfo.eligable_cfi_nvcall;
  object["enabled_cfi_nvcall"] = moduleInfo.enabled_cfi_nvcall;
  object["eligable_stack_clash_protection"] =
      moduleInfo.eligable_stack_clash_protection;
  object["enabled_stack_clash_protection"] =
      moduleInfo.enabled_stack_clash_protection;
  object["eligable_stack_protector"] = moduleInfo.eligable_stack_protector;
  object["enabled_stack_protector"] = moduleInfo.enabled_stack_protector;
  object["eligable_stack_protector_strong"] =
      moduleInfo.eligable_stack_protector_strong;
  object["enabled_stack_protector_strong"] =
      moduleInfo.enabled_stack_protector_strong;
  object["eligable_stack_protector_all"] =
      moduleInfo.eligable_stack_protector_all;
  object["enabled_stack_protector_all"] =
      moduleInfo.enabled_stack_protector_all;
  object["eligable_libcpp_hardening_mode"] =
      moduleInfo.eligable_libcpp_hardening_mode;
  object["enabled_libcpp_hardening_mode"] =
      moduleInfo.enabled_libcpp_hardening_mode;

  object["total_functions"] = moduleInfo.total_functions;
  object["module"] =
      ClOutputModuleName.empty() ? std::string("unknown") : ClOutputModuleName;
  return object;
}

/// Return true if function F calls a function whose name contains
/// targetFunctionName.
static bool functionCallsFunctionWithName(Function &F,
                                          StringRef targetFunctionName) {
  for (Instruction &I : instructions(F)) {
    auto *callInst = dyn_cast<CallInst>(&I);
    if (!callInst)
      continue;

    Function *calledFunction = callInst->getCalledFunction();
    if (calledFunction &&
        calledFunction->getName().contains(targetFunctionName))
      return true;
  }
  return false;
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
      auto *str = dyn_cast<MDString>(Node->getOperand(1));
      if (!str)
        continue;

      std::string signature = str->getString().str();
      if (!StringRef(signature).ends_with(".generalized"))
        return signature;
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

    auto *CI = dyn_cast<ConstantInt>(MDInt->getValue());
    if (CI) {
      return CI->getZExtValue();
    }
  }
  return 0;
}

/// Detect the libcpp hardening mode from calls in the given function.
static MitigationState detectLibcppHardeningMode(Function &F) {
  if (functionCallsFunctionWithName(F, "_libcpp_hardening_mode_enabled"))
    return MitigationState::EligibleEnabled;
  if (functionCallsFunctionWithName(F, "_libcpp_hardening_mode_disabled"))
    return MitigationState::EligibleDisabled;
  return MitigationState::Ineligible;
}

MitigationAnalysis::MitigationAnalysis(MitigationAnalysisSummary Summary)
    : summary_(std::move(Summary)) {}

PreservedAnalyses MitigationAnalysis::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  json::Array jsonArray;
  ModuleMitigationInfo moduleInfo;

  for (Function &F : M) {
    LLVMContext &Context = F.getContext();
    unsigned kindID = Context.getMDKindID("security_mitigations");
    MDNode *ExistingMD = F.getMetadata(kindID);
    if (!ExistingMD)
      continue;

    MitigationInfo info;
    info.gmodule = ClOutputModuleName.empty() ? std::string("unknown")
                                              : ClOutputModuleName;
    info.function = F.getName();

    for (unsigned i = 0; i < ExistingMD->getNumOperands(); ++i) {
      auto *node = dyn_cast<MDNode>(ExistingMD->getOperand(i));
      if (!node || node->getNumOperands() != 2)
        continue;

      auto *mds = dyn_cast<MDString>(node->getOperand(0));
      auto *cam = dyn_cast<ConstantAsMetadata>(node->getOperand(1));
      if (!mds || !cam)
        continue;

      if (auto *ci = cam->getValue()) {
        int value = ci->isOneValue() ? 1 : 0;
        keyAndValueToInfo(info, mds->getString(), value);
      }
    }

    info.libcpp_hardening_mode = detectLibcppHardeningMode(F);

    if (summary_ == MitigationAnalysisSummary::FUNCTION) {
      info.source_mapping = getFunctionSourcePath(F);
      info.type_signature = getFirstFunctionTypeSignature(F);
      info.type_id = getFunctionTypeId(F);

      DEBUG_WITH_TYPE(kMitigationAnalysisDebugType, printInfo(info));
      jsonArray.push_back(infoToJson(info));
    } else {
      // Start aggregating mitigations for entire module
      updateModuleInfo(moduleInfo, info);
    }
  }

  std::string fileName =
      ClOutputModuleName.empty()
          ? std::string("mitigation_info.json")
          : formatv("mitigation_info-{0}.json", ClOutputModuleName);
  if (!jsonArray.empty()) {
    std::string jsonString =
        formatv("{0}", json::Value(std::move(jsonArray))).str();
    if (!jsonString.empty()) {
      writeJsonToFile(jsonString, fileName,
                      formatv("Couldn't write to {0}!", fileName));
    }
  } else if (moduleInfo.total_functions > 0) {
    std::string jsonString =
        formatv("{0}", json::Value(moduleInfoToJson(moduleInfo))).str();
    if (!jsonString.empty()) {
      writeJsonToFile(jsonString, fileName,
                      formatv("Couldn't write to {0}!", fileName));
    }
  }

  return PreservedAnalyses::all();
}
