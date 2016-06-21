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

#include "llvm/Transforms/InstrProfiling.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "instrprof"

namespace {

cl::opt<bool> DoNameCompression("enable-name-compression",
                                cl::desc("Enable name string compression"),
                                cl::init(true));

cl::opt<bool> ValueProfileStaticAlloc(
    "vp-static-alloc",
    cl::desc("Do static counter allocation for value profiler"),
    cl::init(true));
cl::opt<double> NumCountersPerValueSite(
    "vp-counters-per-site",
    cl::desc("The average number of profile counters allocated "
             "per value profiling site."),
    // This is set to a very small value because in real programs, only
    // a very small percentage of value sites have non-zero targets, e.g, 1/30.
    // For those sites with non-zero profile, the average number of targets
    // is usually smaller than 2.
    cl::init(1.0));

class InstrProfilingLegacyPass : public ModulePass {
  InstrProfiling InstrProf;

public:
  static char ID;
  InstrProfilingLegacyPass() : ModulePass(ID), InstrProf() {}
  InstrProfilingLegacyPass(const InstrProfOptions &Options)
      : ModulePass(ID), InstrProf(Options) {}
  const char *getPassName() const override {
    return "Frontend instrumentation-based coverage lowering";
  }

  bool runOnModule(Module &M) override { return InstrProf.run(M); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // anonymous namespace

PreservedAnalyses InstrProfiling::run(Module &M, AnalysisManager<Module> &AM) {
  if (!run(M))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

char InstrProfilingLegacyPass::ID = 0;
INITIALIZE_PASS(InstrProfilingLegacyPass, "instrprof",
                "Frontend instrumentation-based coverage lowering.", false,
                false)

ModulePass *
llvm::createInstrProfilingLegacyPass(const InstrProfOptions &Options) {
  return new InstrProfilingLegacyPass(Options);
}

bool InstrProfiling::isMachO() const {
  return Triple(M->getTargetTriple()).isOSBinFormatMachO();
}

/// Get the section name for the counter variables.
StringRef InstrProfiling::getCountersSection() const {
  return getInstrProfCountersSectionName(isMachO());
}

/// Get the section name for the name variables.
StringRef InstrProfiling::getNameSection() const {
  return getInstrProfNameSectionName(isMachO());
}

/// Get the section name for the profile data variables.
StringRef InstrProfiling::getDataSection() const {
  return getInstrProfDataSectionName(isMachO());
}

/// Get the section name for the coverage mapping data.
StringRef InstrProfiling::getCoverageSection() const {
  return getInstrProfCoverageSectionName(isMachO());
}

bool InstrProfiling::run(Module &M) {
  bool MadeChange = false;

  this->M = &M;
  NamesVar = nullptr;
  NamesSize = 0;
  ProfileDataMap.clear();
  UsedVars.clear();

  // We did not know how many value sites there would be inside
  // the instrumented function. This is counting the number of instrumented
  // target value sites to enter it as field in the profile data variable.
  for (Function &F : M) {
    InstrProfIncrementInst *FirstProfIncInst = nullptr;
    for (BasicBlock &BB : F)
      for (auto I = BB.begin(), E = BB.end(); I != E; I++)
        if (auto *Ind = dyn_cast<InstrProfValueProfileInst>(I))
          computeNumValueSiteCounts(Ind);
        else if (FirstProfIncInst == nullptr)
          FirstProfIncInst = dyn_cast<InstrProfIncrementInst>(I);

    // Value profiling intrinsic lowering requires per-function profile data
    // variable to be created first.
    if (FirstProfIncInst != nullptr)
      static_cast<void>(getOrCreateRegionCounters(FirstProfIncInst));
  }

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

  if (GlobalVariable *CoverageNamesVar =
          M.getNamedGlobal(getCoverageUnusedNamesVarName())) {
    lowerCoverageData(CoverageNamesVar);
    MadeChange = true;
  }

  if (!MadeChange)
    return false;

  emitVNodes();
  emitNameData();
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
  Value *Args[3] = {Ind->getTargetValue(),
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

void InstrProfiling::lowerCoverageData(GlobalVariable *CoverageNamesVar) {

  ConstantArray *Names =
      cast<ConstantArray>(CoverageNamesVar->getInitializer());
  for (unsigned I = 0, E = Names->getNumOperands(); I < E; ++I) {
    Constant *NC = Names->getOperand(I);
    Value *V = NC->stripPointerCasts();
    assert(isa<GlobalVariable>(V) && "Missing reference to function name");
    GlobalVariable *Name = cast<GlobalVariable>(V);

    Name->setLinkage(GlobalValue::PrivateLinkage);
    ReferencedNames.push_back(Name);
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
  // Prohibit function address recording if the function is both internal and
  // COMDAT. This avoids the profile data variable referencing internal symbols
  // in COMDAT.
  if (F->hasLocalLinkage() && F->hasComdat())
    return false;
  // Check uses of this function for other than direct calls or invokes to it.
  // Inline virtual functions have linkeOnceODR linkage. When a key method
  // exists, the vtable will only be emitted in the TU where the key method
  // is defined. In a TU where vtable is not available, the function won't
  // be 'addresstaken'. If its address is not recorded here, the profile data
  // with missing address may be picked by the linker leading  to missing
  // indirect call target info.
  return F->hasAddressTaken() || F->hasLinkOnceLinkage();
}

static inline bool needsComdatForCounter(Function &F, Module &M) {

  if (F.hasComdat())
    return true;

  Triple TT(M.getTargetTriple());
  if (!TT.isOSBinFormatELF())
    return false;

  // See createPGOFuncNameVar for more details. To avoid link errors, profile
  // counters for function with available_externally linkage needs to be changed
  // to linkonce linkage. On ELF based systems, this leads to weak symbols to be
  // created. Without using comdat, duplicate entries won't be removed by the
  // linker leading to increased data segement size and raw profile size. Even
  // worse, since the referenced counter from profile per-function data object
  // will be resolved to the common strong definition, the profile counts for
  // available_externally functions will end up being duplicated in raw profile
  // data. This can result in distorted profile as the counts of those dups
  // will be accumulated by the profile merger.
  GlobalValue::LinkageTypes Linkage = F.getLinkage();
  if (Linkage != GlobalValue::ExternalWeakLinkage &&
      Linkage != GlobalValue::AvailableExternallyLinkage)
    return false;

  return true;
}

static inline Comdat *getOrCreateProfileComdat(Module &M, Function &F,
                                               InstrProfIncrementInst *Inc) {
  if (!needsComdatForCounter(F, M))
    return nullptr;

  // COFF format requires a COMDAT section to have a key symbol with the same
  // name. The linker targeting COFF also requires that the COMDAT
  // a section is associated to must precede the associating section. For this
  // reason, we must choose the counter var's name as the name of the comdat.
  StringRef ComdatPrefix = (Triple(M.getTargetTriple()).isOSBinFormatCOFF()
                                ? getInstrProfCountersVarPrefix()
                                : getInstrProfComdatPrefix());
  return M.getOrInsertComdat(StringRef(getVarName(Inc, ComdatPrefix)));
}

static bool needsRuntimeRegistrationOfSectionRange(const Module &M) {
  // Don't do this for Darwin.  compiler-rt uses linker magic.
  if (Triple(M.getTargetTriple()).isOSDarwin())
    return false;

  // Use linker script magic to get data/cnts/name start/end.
  if (Triple(M.getTargetTriple()).isOSLinux() ||
      Triple(M.getTargetTriple()).isOSFreeBSD() ||
      Triple(M.getTargetTriple()).isPS4CPU())
    return false;

  return true;
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
  ProfileVarsComdat = getOrCreateProfileComdat(*M, *Fn, Inc);

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

  auto *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  // Allocate statically the array of pointers to value profile nodes for
  // the current function.
  Constant *ValuesPtrExpr = ConstantPointerNull::get(Int8PtrTy);
  if (ValueProfileStaticAlloc && !needsRuntimeRegistrationOfSectionRange(*M)) {

    uint64_t NS = 0;
    for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
      NS += PD.NumValueSites[Kind];
    if (NS) {
      ArrayType *ValuesTy = ArrayType::get(Type::getInt64Ty(Ctx), NS);

      auto *ValuesVar =
          new GlobalVariable(*M, ValuesTy, false, NamePtr->getLinkage(),
                             Constant::getNullValue(ValuesTy),
                             getVarName(Inc, getInstrProfValuesVarPrefix()));
      ValuesVar->setVisibility(NamePtr->getVisibility());
      ValuesVar->setSection(getInstrProfValuesSectionName(isMachO()));
      ValuesVar->setAlignment(8);
      ValuesVar->setComdat(ProfileVarsComdat);
      ValuesPtrExpr =
          ConstantExpr::getBitCast(ValuesVar, llvm::Type::getInt8PtrTy(Ctx));
    }
  }

  // Create data variable.
  auto *Int16Ty = Type::getInt16Ty(Ctx);
  auto *Int16ArrayTy = ArrayType::get(Int16Ty, IPVK_Last + 1);
  Type *DataTypes[] = {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *DataTy = StructType::get(Ctx, makeArrayRef(DataTypes));

  Constant *FunctionAddr = shouldRecordFunctionAddr(Fn)
                               ? ConstantExpr::getBitCast(Fn, Int8PtrTy)
                               : ConstantPointerNull::get(Int8PtrTy);

  Constant *Int16ArrayVals[IPVK_Last + 1];
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
  // Now that the linkage set by the FE has been passed to the data and counter
  // variables, reset Name variable's linkage and visibility to private so that
  // it can be removed later by the compiler.
  NamePtr->setLinkage(GlobalValue::PrivateLinkage);
  // Collect the referenced names to be used by emitNameData.
  ReferencedNames.push_back(NamePtr);

  return CounterPtr;
}

void InstrProfiling::emitVNodes() {
  if (!ValueProfileStaticAlloc)
    return;

  // For now only support this on platforms that do
  // not require runtime registration to discover
  // named section start/end.
  if (needsRuntimeRegistrationOfSectionRange(*M))
    return;

  size_t TotalNS = 0;
  for (auto &PD : ProfileDataMap) {
    for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
      TotalNS += PD.second.NumValueSites[Kind];
  }

  if (!TotalNS)
    return;

  uint64_t NumCounters = TotalNS * NumCountersPerValueSite;
// Heuristic for small programs with very few total value sites.
// The default value of vp-counters-per-site is chosen based on
// the observation that large apps usually have a low percentage
// of value sites that actually have any profile data, and thus
// the average number of counters per site is low. For small
// apps with very few sites, this may not be true. Bump up the
// number of counters in this case.
#define INSTR_PROF_MIN_VAL_COUNTS 10
  if (NumCounters < INSTR_PROF_MIN_VAL_COUNTS)
    NumCounters = std::max(INSTR_PROF_MIN_VAL_COUNTS, (int)NumCounters * 2);

  auto &Ctx = M->getContext();
  Type *VNodeTypes[] = {
#define INSTR_PROF_VALUE_NODE(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *VNodeTy = StructType::get(Ctx, makeArrayRef(VNodeTypes));

  ArrayType *VNodesTy = ArrayType::get(VNodeTy, NumCounters);
  auto *VNodesVar = new GlobalVariable(
      *M, VNodesTy, false, llvm::GlobalValue::PrivateLinkage,
      Constant::getNullValue(VNodesTy), getInstrProfVNodesVarName());
  VNodesVar->setSection(getInstrProfVNodesSectionName(isMachO()));
  UsedVars.push_back(VNodesVar);
}

void InstrProfiling::emitNameData() {
  std::string UncompressedData;

  if (ReferencedNames.empty())
    return;

  std::string CompressedNameStr;
  if (Error E = collectPGOFuncNameStrings(ReferencedNames, CompressedNameStr,
                                          DoNameCompression)) {
    llvm::report_fatal_error(toString(std::move(E)), false);
  }

  auto &Ctx = M->getContext();
  auto *NamesVal = llvm::ConstantDataArray::getString(
      Ctx, StringRef(CompressedNameStr), false);
  NamesVar = new llvm::GlobalVariable(*M, NamesVal->getType(), true,
                                      llvm::GlobalValue::PrivateLinkage,
                                      NamesVal, getInstrProfNamesVarName());
  NamesSize = CompressedNameStr.size();
  NamesVar->setSection(getNameSection());
  UsedVars.push_back(NamesVar);
}

void InstrProfiling::emitRegistration() {
  if (!needsRuntimeRegistrationOfSectionRange(*M))
    return;

  // Construct the function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *VoidPtrTy = Type::getInt8PtrTy(M->getContext());
  auto *Int64Ty = Type::getInt64Ty(M->getContext());
  auto *RegisterFTy = FunctionType::get(VoidTy, false);
  auto *RegisterF = Function::Create(RegisterFTy, GlobalValue::InternalLinkage,
                                     getInstrProfRegFuncsName(), M);
  RegisterF->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  if (Options.NoRedZone)
    RegisterF->addFnAttr(Attribute::NoRedZone);

  auto *RuntimeRegisterTy = FunctionType::get(VoidTy, VoidPtrTy, false);
  auto *RuntimeRegisterF =
      Function::Create(RuntimeRegisterTy, GlobalVariable::ExternalLinkage,
                       getInstrProfRegFuncName(), M);

  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", RegisterF));
  for (Value *Data : UsedVars)
    if (Data != NamesVar)
      IRB.CreateCall(RuntimeRegisterF, IRB.CreateBitCast(Data, VoidPtrTy));

  if (NamesVar) {
    Type *ParamTypes[] = {VoidPtrTy, Int64Ty};
    auto *NamesRegisterTy =
        FunctionType::get(VoidTy, makeArrayRef(ParamTypes), false);
    auto *NamesRegisterF =
        Function::Create(NamesRegisterTy, GlobalVariable::ExternalLinkage,
                         getInstrProfNamesRegFuncName(), M);
    IRB.CreateCall(NamesRegisterF, {IRB.CreateBitCast(NamesVar, VoidPtrTy),
                                    IRB.getInt64(NamesSize)});
  }

  IRB.CreateRetVoid();
}

void InstrProfiling::emitRuntimeHook() {

  // We expect the linker to be invoked with -u<hook_var> flag for linux,
  // for which case there is no need to emit the user function.
  if (Triple(M->getTargetTriple()).isOSLinux())
    return;

  // If the module's provided its own runtime, we don't need to do anything.
  if (M->getGlobalVariable(getInstrProfRuntimeHookVarName()))
    return;

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
  if (Options.NoRedZone)
    User->addFnAttr(Attribute::NoRedZone);
  User->setVisibility(GlobalValue::HiddenVisibility);
  if (Triple(M->getTargetTriple()).supportsCOMDAT())
    User->setComdat(M->getOrInsertComdat(User->getName()));

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
  if (!RegisterF && InstrProfileOutput.empty())
    return;

  // Create the initialization function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *F = Function::Create(FunctionType::get(VoidTy, false),
                             GlobalValue::InternalLinkage,
                             getInstrProfInitFuncName(), M);
  F->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  F->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    F->addFnAttr(Attribute::NoRedZone);

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
