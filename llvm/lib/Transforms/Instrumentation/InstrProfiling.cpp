//===-- InstrProfiling.cpp - Frontend instrumentation based profiling -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers instrprof_* intrinsics emitted by a frontend for profiling.
// It also builds the data structures and initialization code needed for
// updating execution counts and emitting the profile at runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "instrprof"

namespace {

class InstrProfiling : public ModulePass {
public:
  static char ID;

  InstrProfiling() : ModulePass(ID) {}

  InstrProfiling(const InstrProfOptions &Options)
      : ModulePass(ID), Options(Options) {}

  const char *getPassName() const override {
    return "Frontend instrumentation-based coverage lowering";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

private:
  InstrProfOptions Options;
  Module *M;
  typedef struct PerFunctionProfileData {
    uint32_t NumValueSites[IPVK_Last+1];
    GlobalVariable* RegionCounters;
    GlobalVariable* DataVar;
    PerFunctionProfileData() : RegionCounters(nullptr), DataVar(nullptr) {
      memset(NumValueSites, 0, sizeof(uint32_t) * (IPVK_Last+1));
    }
  } PerFunctionProfileData;
  DenseMap<GlobalVariable *, PerFunctionProfileData> ProfileDataMap;
  std::vector<Value *> UsedVars;

  bool isMachO() const {
    return Triple(M->getTargetTriple()).isOSBinFormatMachO();
  }

  /// Get the section name for the counter variables.
  StringRef getCountersSection() const {
    return getInstrProfCountersSectionName(isMachO());
  }

  /// Get the section name for the name variables.
  StringRef getNameSection() const {
    return getInstrProfNameSectionName(isMachO());
  }

  /// Get the section name for the profile data variables.
  StringRef getDataSection() const {
    return getInstrProfDataSectionName(isMachO());
  }

  /// Get the section name for the coverage mapping data.
  StringRef getCoverageSection() const {
    return getInstrProfCoverageSectionName(isMachO());
  }

  /// Count the number of instrumented value sites for the function.
  void computeNumValueSiteCounts(InstrProfValueProfileInst *Ins);

  /// Replace instrprof_value_profile with a call to runtime library.
  void lowerValueProfileInst(InstrProfValueProfileInst *Ins);

  /// Replace instrprof_increment with an increment of the appropriate value.
  void lowerIncrement(InstrProfIncrementInst *Inc);

  /// Set up the section and uses for coverage data and its references.
  void lowerCoverageData(GlobalVariable *CoverageData);

  /// Get the region counters for an increment, creating them if necessary.
  ///
  /// If the counter array doesn't yet exist, the profile data variables
  /// referring to them will also be created.
  GlobalVariable *getOrCreateRegionCounters(InstrProfIncrementInst *Inc);

  /// Emit runtime registration functions for each profile data variable.
  void emitRegistration();

  /// Emit the necessary plumbing to pull in the runtime initialization.
  void emitRuntimeHook();

  /// Add uses of our data variables and runtime hook.
  void emitUses();

  /// Create a static initializer for our data, on platforms that need it,
  /// and for any profile output file that was specified.
  void emitInitialization();
};

} // anonymous namespace

char InstrProfiling::ID = 0;
INITIALIZE_PASS(InstrProfiling, "instrprof",
                "Frontend instrumentation-based coverage lowering.", false,
                false)

ModulePass *llvm::createInstrProfilingPass(const InstrProfOptions &Options) {
  return new InstrProfiling(Options);
}

bool InstrProfiling::runOnModule(Module &M) {
  bool MadeChange = false;

  this->M = &M;
  ProfileDataMap.clear();
  UsedVars.clear();

  // We did not know how many value sites there would be inside
  // the instrumented function. This is counting the number of instrumented
  // target value sites to enter it as field in the profile data variable.
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (auto I = BB.begin(), E = BB.end(); I != E;)
        if (auto *Ind = dyn_cast<InstrProfValueProfileInst>(I++))
          computeNumValueSiteCounts(Ind);

  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (auto I = BB.begin(), E = BB.end(); I != E;) {
        auto Instr = I++;
        if (auto *Inc = dyn_cast<InstrProfIncrementInst>(Instr)) {
          lowerIncrement(Inc);
          MadeChange = true;
        } else if (auto *Ind = dyn_cast<InstrProfValueProfileInst>(Instr)) {
          lowerValueProfileInst(Ind);
          MadeChange = true;
        }
      }

  if (GlobalVariable *Coverage =
          M.getNamedGlobal(getCoverageMappingVarName())) {
    lowerCoverageData(Coverage);
    MadeChange = true;
  }

  if (!MadeChange)
    return false;

  emitRegistration();
  emitRuntimeHook();
  emitUses();
  emitInitialization();
  return true;
}

static Constant *getOrInsertValueProfilingCall(Module &M) {
  LLVMContext &Ctx = M.getContext();
  auto *ReturnTy = Type::getVoidTy(M.getContext());
  Type *ParamTypes[] = {
#define VALUE_PROF_FUNC_PARAM(ParamType, ParamName, ParamLLVMType) ParamLLVMType
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *ValueProfilingCallTy =
      FunctionType::get(ReturnTy, makeArrayRef(ParamTypes), false);
  return M.getOrInsertFunction(getInstrProfValueProfFuncName(),
                               ValueProfilingCallTy);
}

void InstrProfiling::computeNumValueSiteCounts(InstrProfValueProfileInst *Ind) {

  GlobalVariable *Name = Ind->getName();
  uint64_t ValueKind = Ind->getValueKind()->getZExtValue();
  uint64_t Index = Ind->getIndex()->getZExtValue();
  auto It = ProfileDataMap.find(Name);
  if (It == ProfileDataMap.end()) {
    PerFunctionProfileData PD;
    PD.NumValueSites[ValueKind] = Index + 1;
    ProfileDataMap[Name] = PD;
  } else if (It->second.NumValueSites[ValueKind] <= Index)
    It->second.NumValueSites[ValueKind] = Index + 1;
}

void InstrProfiling::lowerValueProfileInst(InstrProfValueProfileInst *Ind) {

  GlobalVariable *Name = Ind->getName();
  auto It = ProfileDataMap.find(Name);
  assert(It != ProfileDataMap.end() && It->second.DataVar &&
    "value profiling detected in function with no counter incerement");

  GlobalVariable *DataVar = It->second.DataVar;
  uint64_t ValueKind = Ind->getValueKind()->getZExtValue();
  uint64_t Index = Ind->getIndex()->getZExtValue();
  for (uint32_t Kind = IPVK_First; Kind < ValueKind; ++Kind)
    Index += It->second.NumValueSites[Kind];

  IRBuilder<> Builder(Ind);
  Value* Args[3] = {Ind->getTargetValue(),
      Builder.CreateBitCast(DataVar, Builder.getInt8PtrTy()),
      Builder.getInt32(Index)};
  Ind->replaceAllUsesWith(
      Builder.CreateCall(getOrInsertValueProfilingCall(*M), Args));
  Ind->eraseFromParent();
}

void InstrProfiling::lowerIncrement(InstrProfIncrementInst *Inc) {
  GlobalVariable *Counters = getOrCreateRegionCounters(Inc);

  IRBuilder<> Builder(Inc);
  uint64_t Index = Inc->getIndex()->getZExtValue();
  Value *Addr = Builder.CreateConstInBoundsGEP2_64(Counters, 0, Index);
  Value *Count = Builder.CreateLoad(Addr, "pgocount");
  Count = Builder.CreateAdd(Count, Builder.getInt64(1));
  Inc->replaceAllUsesWith(Builder.CreateStore(Count, Addr));
  Inc->eraseFromParent();
}

void InstrProfiling::lowerCoverageData(GlobalVariable *CoverageData) {
  CoverageData->setSection(getCoverageSection());
  CoverageData->setAlignment(8);

  Constant *Init = CoverageData->getInitializer();
  // We're expecting { i32, i32, i32, i32, [n x { i8*, i32, i32 }], [m x i8] }
  // for some C. If not, the frontend's given us something broken.
  assert(Init->getNumOperands() == 6 && "bad number of fields in coverage map");
  assert(isa<ConstantArray>(Init->getAggregateElement(4)) &&
         "invalid function list in coverage map");
  ConstantArray *Records = cast<ConstantArray>(Init->getAggregateElement(4));
  for (unsigned I = 0, E = Records->getNumOperands(); I < E; ++I) {
    Constant *Record = Records->getOperand(I);
    Value *V = const_cast<Value *>(Record->getOperand(0))->stripPointerCasts();

    assert(isa<GlobalVariable>(V) && "Missing reference to function name");
    GlobalVariable *Name = cast<GlobalVariable>(V);

    // If we have region counters for this name, we've already handled it.
    auto It = ProfileDataMap.find(Name);
    if (It != ProfileDataMap.end())
      if (It->second.RegionCounters)
        continue;

    // Move the name variable to the right section.
    Name->setSection(getNameSection());
    Name->setAlignment(1);
  }
}

/// Get the name of a profiling variable for a particular function.
static std::string getVarName(InstrProfIncrementInst *Inc, StringRef Prefix) {
  StringRef NamePrefix = getInstrProfNameVarPrefix();
  StringRef Name = Inc->getName()->getName().substr(NamePrefix.size());
  return (Prefix + Name).str();
}

static inline bool shouldRecordFunctionAddr(Function *F) {
  // Check the linkage
  if (!F->hasLinkOnceLinkage() && !F->hasLocalLinkage() &&
      !F->hasAvailableExternallyLinkage())
    return true;
  // Check uses of this function for other than direct calls or invokes to it.
  return F->hasAddressTaken();
}

static inline Comdat *getOrCreateProfileComdat(Module &M,
                                               InstrProfIncrementInst *Inc) {
  // COFF format requires a COMDAT section to have a key symbol with the same
  // name.
  StringRef ComdatPrefix = (Triple(M.getTargetTriple()).isOSBinFormatCOFF()
                                ? getInstrProfDataVarPrefix()
                                : getInstrProfComdatPrefix());
  return M.getOrInsertComdat(StringRef(getVarName(Inc, ComdatPrefix)));
}

GlobalVariable *
InstrProfiling::getOrCreateRegionCounters(InstrProfIncrementInst *Inc) {
  GlobalVariable *NamePtr = Inc->getName();
  auto It = ProfileDataMap.find(NamePtr);
  PerFunctionProfileData PD;
  if (It != ProfileDataMap.end()) {
    if (It->second.RegionCounters)
      return It->second.RegionCounters;
    PD = It->second;
  }

  // Move the name variable to the right section. Place them in a COMDAT group
  // if the associated function is a COMDAT. This will make sure that
  // only one copy of counters of the COMDAT function will be emitted after
  // linking.
  Function *Fn = Inc->getParent()->getParent();
  Comdat *ProfileVarsComdat = nullptr;
  if (Fn->hasComdat())
    ProfileVarsComdat = getOrCreateProfileComdat(*M, Inc);
  NamePtr->setSection(getNameSection());
  NamePtr->setAlignment(1);
  NamePtr->setComdat(ProfileVarsComdat);

  uint64_t NumCounters = Inc->getNumCounters()->getZExtValue();
  LLVMContext &Ctx = M->getContext();
  ArrayType *CounterTy = ArrayType::get(Type::getInt64Ty(Ctx), NumCounters);

  // Create the counters variable.
  auto *CounterPtr =
      new GlobalVariable(*M, CounterTy, false, NamePtr->getLinkage(),
                         Constant::getNullValue(CounterTy),
                         getVarName(Inc, getInstrProfCountersVarPrefix()));
  CounterPtr->setVisibility(NamePtr->getVisibility());
  CounterPtr->setSection(getCountersSection());
  CounterPtr->setAlignment(8);
  CounterPtr->setComdat(ProfileVarsComdat);

  // Create data variable.
  auto *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  auto *Int16Ty = Type::getInt16Ty(Ctx);
  auto *Int16ArrayTy = ArrayType::get(Int16Ty, IPVK_Last+1);
  Type *DataTypes[] = {
    #define INSTR_PROF_DATA(Type, LLVMType, Name, Init) LLVMType,
    #include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *DataTy = StructType::get(Ctx, makeArrayRef(DataTypes));

  Constant *FunctionAddr = shouldRecordFunctionAddr(Fn) ?
                           ConstantExpr::getBitCast(Fn, Int8PtrTy) :
                           ConstantPointerNull::get(Int8PtrTy);

  Constant *Int16ArrayVals[IPVK_Last+1];
  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
    Int16ArrayVals[Kind] = ConstantInt::get(Int16Ty, PD.NumValueSites[Kind]);

  Constant *DataVals[] = {
    #define INSTR_PROF_DATA(Type, LLVMType, Name, Init) Init,
    #include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *Data = new GlobalVariable(*M, DataTy, false, NamePtr->getLinkage(),
                                  ConstantStruct::get(DataTy, DataVals),
                                  getVarName(Inc, getInstrProfDataVarPrefix()));
  Data->setVisibility(NamePtr->getVisibility());
  Data->setSection(getDataSection());
  Data->setAlignment(INSTR_PROF_DATA_ALIGNMENT);
  Data->setComdat(ProfileVarsComdat);

  PD.RegionCounters = CounterPtr;
  PD.DataVar = Data;
  ProfileDataMap[NamePtr] = PD;

  // Mark the data variable as used so that it isn't stripped out.
  UsedVars.push_back(Data);

  return CounterPtr;
}

void InstrProfiling::emitRegistration() {
  // Don't do this for Darwin.  compiler-rt uses linker magic.
  if (Triple(M->getTargetTriple()).isOSDarwin())
    return;

  // Use linker script magic to get data/cnts/name start/end.
  if (Triple(M->getTargetTriple()).isOSLinux() ||
      Triple(M->getTargetTriple()).isOSFreeBSD())
    return;

  // Construct the function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *VoidPtrTy = Type::getInt8PtrTy(M->getContext());
  auto *RegisterFTy = FunctionType::get(VoidTy, false);
  auto *RegisterF = Function::Create(RegisterFTy, GlobalValue::InternalLinkage,
                                     getInstrProfRegFuncsName(), M);
  RegisterF->setUnnamedAddr(true);
  if (Options.NoRedZone) RegisterF->addFnAttr(Attribute::NoRedZone);

  auto *RuntimeRegisterTy = FunctionType::get(VoidTy, VoidPtrTy, false);
  auto *RuntimeRegisterF =
      Function::Create(RuntimeRegisterTy, GlobalVariable::ExternalLinkage,
                       getInstrProfRegFuncName(), M);

  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", RegisterF));
  for (Value *Data : UsedVars)
    IRB.CreateCall(RuntimeRegisterF, IRB.CreateBitCast(Data, VoidPtrTy));
  IRB.CreateRetVoid();
}

void InstrProfiling::emitRuntimeHook() {

  // We expect the linker to be invoked with -u<hook_var> flag for linux,
  // for which case there is no need to emit the user function.
  if (Triple(M->getTargetTriple()).isOSLinux())
    return;

  // If the module's provided its own runtime, we don't need to do anything.
  if (M->getGlobalVariable(getInstrProfRuntimeHookVarName())) return;

  // Declare an external variable that will pull in the runtime initialization.
  auto *Int32Ty = Type::getInt32Ty(M->getContext());
  auto *Var =
      new GlobalVariable(*M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         nullptr, getInstrProfRuntimeHookVarName());

  // Make a function that uses it.
  auto *User = Function::Create(FunctionType::get(Int32Ty, false),
                                GlobalValue::LinkOnceODRLinkage,
                                getInstrProfRuntimeHookVarUseFuncName(), M);
  User->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone) User->addFnAttr(Attribute::NoRedZone);
  User->setVisibility(GlobalValue::HiddenVisibility);

  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", User));
  auto *Load = IRB.CreateLoad(Var);
  IRB.CreateRet(Load);

  // Mark the user variable as used so that it isn't stripped out.
  UsedVars.push_back(User);
}

void InstrProfiling::emitUses() {
  if (UsedVars.empty())
    return;

  GlobalVariable *LLVMUsed = M->getGlobalVariable("llvm.used");
  std::vector<Constant *> MergedVars;
  if (LLVMUsed) {
    // Collect the existing members of llvm.used.
    ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
    for (unsigned I = 0, E = Inits->getNumOperands(); I != E; ++I)
      MergedVars.push_back(Inits->getOperand(I));
    LLVMUsed->eraseFromParent();
  }

  Type *i8PTy = Type::getInt8PtrTy(M->getContext());
  // Add uses for our data.
  for (auto *Value : UsedVars)
    MergedVars.push_back(
        ConstantExpr::getBitCast(cast<Constant>(Value), i8PTy));

  // Recreate llvm.used.
  ArrayType *ATy = ArrayType::get(i8PTy, MergedVars.size());
  LLVMUsed =
      new GlobalVariable(*M, ATy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(ATy, MergedVars), "llvm.used");
  LLVMUsed->setSection("llvm.metadata");
}

void InstrProfiling::emitInitialization() {
  std::string InstrProfileOutput = Options.InstrProfileOutput;

  Constant *RegisterF = M->getFunction(getInstrProfRegFuncsName());
  if (!RegisterF && InstrProfileOutput.empty()) return;

  // Create the initialization function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *F = Function::Create(FunctionType::get(VoidTy, false),
                             GlobalValue::InternalLinkage,
                             getInstrProfInitFuncName(), M);
  F->setUnnamedAddr(true);
  F->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone) F->addFnAttr(Attribute::NoRedZone);

  // Add the basic block and the necessary calls.
  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", F));
  if (RegisterF)
    IRB.CreateCall(RegisterF, {});
  if (!InstrProfileOutput.empty()) {
    auto *Int8PtrTy = Type::getInt8PtrTy(M->getContext());
    auto *SetNameTy = FunctionType::get(VoidTy, Int8PtrTy, false);
    auto *SetNameF = Function::Create(SetNameTy, GlobalValue::ExternalLinkage,
                                      getInstrProfFileOverriderFuncName(), M);

    // Create variable for profile name.
    Constant *ProfileNameConst =
        ConstantDataArray::getString(M->getContext(), InstrProfileOutput, true);
    GlobalVariable *ProfileName =
        new GlobalVariable(*M, ProfileNameConst->getType(), true,
                           GlobalValue::PrivateLinkage, ProfileNameConst);

    IRB.CreateCall(SetNameF, IRB.CreatePointerCast(ProfileName, Int8PtrTy));
  }
  IRB.CreateRetVoid();

  appendToGlobalCtors(*M, F, 0);
}
