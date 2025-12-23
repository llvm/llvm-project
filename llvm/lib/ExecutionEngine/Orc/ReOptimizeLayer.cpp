#include "llvm/ExecutionEngine/Orc/ReOptimizeLayer.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"

using namespace llvm;
using namespace orc;

bool ReOptimizeLayer::ReOptMaterializationUnitState::tryStartReoptimize() {
  std::unique_lock<std::mutex> Lock(Mutex);
  if (Reoptimizing)
    return false;

  Reoptimizing = true;
  return true;
}

void ReOptimizeLayer::ReOptMaterializationUnitState::reoptimizeSucceeded() {
  std::unique_lock<std::mutex> Lock(Mutex);
  assert(Reoptimizing && "Tried to mark unstarted reoptimization as done");
  Reoptimizing = false;
  CurVersion++;
}

void ReOptimizeLayer::ReOptMaterializationUnitState::reoptimizeFailed() {
  std::unique_lock<std::mutex> Lock(Mutex);
  assert(Reoptimizing && "Tried to mark unstarted reoptimization as done");
  Reoptimizing = false;
}

static void orc_rt_lite_reoptimize_helper(
    shared::CWrapperFunctionBuffer (*JITDispatch)(void *Ctx, void *Tag,
                                                  void *Data, size_t Size),
    void *JITDispatchCtx, void *Tag, uint64_t MUID, uint32_t CurVersion) {
  // Serialize the arguments into a WrapperFunctionBuffer and call dispatch.
  using SPSArgs = shared::SPSArgList<uint64_t, uint32_t>;
  auto ArgBytes =
      shared::WrapperFunctionBuffer::allocate(SPSArgs::size(MUID, CurVersion));
  shared::SPSOutputBuffer OB(ArgBytes.data(), ArgBytes.size());
  if (!SPSArgs::serialize(OB, MUID, CurVersion)) {
    errs()
        << "Reoptimization error: could not serialize reoptimization arguments";
    abort();
  }
  shared::WrapperFunctionBuffer Buf{
      JITDispatch(JITDispatchCtx, Tag, ArgBytes.data(), ArgBytes.size())};

  if (const char *ErrMsg = Buf.getOutOfBandError()) {
    errs() << "Reoptimization error: " << ErrMsg << "\naborting.\n";
    abort();
  }
}

Error ReOptimizeLayer::addOrcRTLiteSupport(JITDylib &PlatformJD,
                                           const DataLayout &DL) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto Mod = std::make_unique<Module>("orc-rt-lite-reoptimize.ll", *Ctx);
  Mod->setDataLayout(DL);

  IRBuilder<> Builder(*Ctx);

  // Create basic types portably
  Type *VoidTy = Type::getVoidTy(*Ctx);
  Type *Int8Ty = Type::getInt8Ty(*Ctx);
  Type *Int32Ty = Type::getInt32Ty(*Ctx);
  Type *Int64Ty = Type::getInt64Ty(*Ctx);
  Type *VoidPtrTy = PointerType::getUnqual(*Ctx);

  // Helper function type: void (void*, void*, void*, uint64_t, uint32_t)
  FunctionType *HelperFnTy = FunctionType::get(
      VoidTy, {VoidPtrTy, VoidPtrTy, VoidPtrTy, Int64Ty, Int32Ty}, false);

  // Define ReoptimizeTag with initializer = 0
  GlobalVariable *ReoptimizeTag = new GlobalVariable(
      *Mod, Int8Ty, false, GlobalValue::ExternalLinkage,
      ConstantInt::get(Int8Ty, 0), "__orc_rt_reoptimize_tag");

  // Define orc_rt_lite_reoptimize function: void (uint64_t, uint32_t)
  FunctionType *ReOptimizeFnTy =
      FunctionType::get(VoidTy, {Int64Ty, Int32Ty}, false);

  Function *ReOptimizeFn =
      Function::Create(ReOptimizeFnTy, Function::ExternalLinkage,
                       "__orc_rt_reoptimize", Mod.get());

  // Set parameter names
  auto ArgIt = ReOptimizeFn->arg_begin();
  Value *MUID = &*ArgIt++;
  MUID->setName("MUID");
  Value *CurVersion = &*ArgIt;
  CurVersion->setName("CurVersion");

  // Build function body
  BasicBlock *Entry = BasicBlock::Create(*Ctx, "entry", ReOptimizeFn);
  Builder.SetInsertPoint(Entry);

  // Create absolute address constants
  auto &JDI = PlatformJD.getExecutionSession()
                  .getExecutorProcessControl()
                  .getJITDispatchInfo();

  Type *IntPtrTy = DL.getIntPtrType(*Ctx);
  Constant *JITDispatchPtr = ConstantExpr::getIntToPtr(
      ConstantInt::get(IntPtrTy, JDI.JITDispatchFunction.getValue()),
      VoidPtrTy);
  Constant *JITDispatchCtxPtr = ConstantExpr::getIntToPtr(
      ConstantInt::get(IntPtrTy, JDI.JITDispatchContext.getValue()), VoidPtrTy);
  Constant *HelperFnAddr = ConstantExpr::getIntToPtr(
      ConstantInt::get(IntPtrTy, reinterpret_cast<uintptr_t>(
                                     &orc_rt_lite_reoptimize_helper)),
      PointerType::getUnqual(*Ctx));

  // Cast ReoptimizeTag to void*
  Value *ReoptimizeTagPtr = Builder.CreatePointerCast(ReoptimizeTag, VoidPtrTy);

  // Call the helper function
  Builder.CreateCall(
      HelperFnTy, HelperFnAddr,
      {JITDispatchPtr, JITDispatchCtxPtr, ReoptimizeTagPtr, MUID, CurVersion});

  // Return void
  Builder.CreateRetVoid();

  return BaseLayer.add(PlatformJD,
                       ThreadSafeModule(std::move(Mod), std::move(Ctx)));
}

Error ReOptimizeLayer::registerRuntimeFunctions(JITDylib &PlatformJD) {
  ExecutionSession::JITDispatchHandlerAssociationMap WFs;
  using ReoptimizeSPSSig = shared::SPSError(uint64_t, uint32_t);
  WFs[Mangle("__orc_rt_reoptimize_tag")] =
      ES.wrapAsyncWithSPS<ReoptimizeSPSSig>(this,
                                            &ReOptimizeLayer::rt_reoptimize);
  return ES.registerJITDispatchHandlers(PlatformJD, std::move(WFs));
}

void ReOptimizeLayer::emit(std::unique_ptr<MaterializationResponsibility> R,
                           ThreadSafeModule TSM) {
  auto &JD = R->getTargetJITDylib();

  bool HasNonCallable = false;
  for (auto &KV : R->getSymbols()) {
    auto &Flags = KV.second;
    if (!Flags.isCallable())
      HasNonCallable = true;
  }

  if (HasNonCallable) {
    BaseLayer.emit(std::move(R), std::move(TSM));
    return;
  }

  auto &MUState = createMaterializationUnitState(TSM);

  if (auto Err = R->withResourceKeyDo([&](ResourceKey Key) {
        registerMaterializationUnitResource(Key, MUState);
      })) {
    ES.reportError(std::move(Err));
    R->failMaterialization();
    return;
  }

  if (auto Err =
          ProfilerFunc(*this, MUState.getID(), MUState.getCurVersion(), TSM)) {
    ES.reportError(std::move(Err));
    R->failMaterialization();
    return;
  }

  auto InitialDests =
      emitMUImplSymbols(MUState, MUState.getCurVersion(), JD, std::move(TSM));
  if (!InitialDests) {
    ES.reportError(InitialDests.takeError());
    R->failMaterialization();
    return;
  }

  RSManager.emitRedirectableSymbols(std::move(R), std::move(*InitialDests));
}

Error ReOptimizeLayer::reoptimizeIfCallFrequent(ReOptimizeLayer &Parent,
                                                ReOptMaterializationUnitID MUID,
                                                unsigned CurVersion,
                                                ThreadSafeModule &TSM) {
  return TSM.withModuleDo([&](Module &M) -> Error {
    Type *I64Ty = Type::getInt64Ty(M.getContext());
    GlobalVariable *Counter = new GlobalVariable(
        M, I64Ty, false, GlobalValue::InternalLinkage,
        Constant::getNullValue(I64Ty), "__orc_reopt_counter");
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      auto &BB = F.getEntryBlock();
      auto *IP = &*BB.getFirstInsertionPt();
      IRBuilder<> IRB(IP);
      Value *Threshold = ConstantInt::get(I64Ty, CallCountThreshold, true);
      Value *Cnt = IRB.CreateLoad(I64Ty, Counter);
      // Use EQ to prevent further reoptimize calls.
      Value *Cmp = IRB.CreateICmpEQ(Cnt, Threshold);
      Value *Added = IRB.CreateAdd(Cnt, ConstantInt::get(I64Ty, 1));
      (void)IRB.CreateStore(Added, Counter);
      Instruction *SplitTerminator = SplitBlockAndInsertIfThen(Cmp, IP, false);
      createReoptimizeCall(M, *SplitTerminator, MUID, CurVersion);
    }
    return Error::success();
  });
}

Expected<SymbolMap>
ReOptimizeLayer::emitMUImplSymbols(ReOptMaterializationUnitState &MUState,
                                   uint32_t Version, JITDylib &JD,
                                   ThreadSafeModule TSM) {
  DenseMap<SymbolStringPtr, SymbolStringPtr> RenamedMap;
  cantFail(TSM.withModuleDo([&](Module &M) -> Error {
    MangleAndInterner Mangle(ES, M.getDataLayout());
    for (auto &F : M)
      if (!F.isDeclaration()) {
        std::string NewName =
            (F.getName() + ".__def__." + Twine(Version)).str();
        RenamedMap[Mangle(F.getName())] = Mangle(NewName);
        F.setName(NewName);
      }
    return Error::success();
  }));

  auto RT = JD.createResourceTracker();
  if (auto Err =
          JD.define(std::make_unique<BasicIRLayerMaterializationUnit>(
                        BaseLayer, *getManglingOptions(), std::move(TSM)),
                    RT))
    return Err;
  MUState.setResourceTracker(RT);

  SymbolLookupSet LookupSymbols;
  for (auto [K, V] : RenamedMap)
    LookupSymbols.add(V);

  auto ImplSymbols =
      ES.lookup({{&JD, JITDylibLookupFlags::MatchAllSymbols}}, LookupSymbols,
                LookupKind::Static, SymbolState::Resolved);
  if (auto Err = ImplSymbols.takeError())
    return Err;

  SymbolMap Result;
  for (auto [K, V] : RenamedMap)
    Result[K] = (*ImplSymbols)[V];

  return Result;
}

void ReOptimizeLayer::rt_reoptimize(SendErrorFn SendResult,
                                    ReOptMaterializationUnitID MUID,
                                    uint32_t CurVersion) {
  auto &MUState = getMaterializationUnitState(MUID);
  if (CurVersion < MUState.getCurVersion() || !MUState.tryStartReoptimize()) {
    SendResult(Error::success());
    return;
  }

  ThreadSafeModule TSM = cloneToNewContext(MUState.getThreadSafeModule());
  auto OldRT = MUState.getResourceTracker();
  auto &JD = OldRT->getJITDylib();

  if (auto Err = ReOptFunc(*this, MUID, CurVersion + 1, OldRT, TSM)) {
    ES.reportError(std::move(Err));
    MUState.reoptimizeFailed();
    SendResult(Error::success());
    return;
  }

  auto SymbolDests =
      emitMUImplSymbols(MUState, CurVersion + 1, JD, std::move(TSM));
  if (!SymbolDests) {
    ES.reportError(SymbolDests.takeError());
    MUState.reoptimizeFailed();
    SendResult(Error::success());
    return;
  }

  if (auto Err = RSManager.redirect(JD, std::move(*SymbolDests))) {
    ES.reportError(std::move(Err));
    MUState.reoptimizeFailed();
    SendResult(Error::success());
    return;
  }

  MUState.reoptimizeSucceeded();
  SendResult(Error::success());
}

void ReOptimizeLayer::createReoptimizeCall(Module &M, Instruction &IP,
                                           ReOptMaterializationUnitID MUID,
                                           uint32_t CurVersion) {
  Type *MUIDTy = IntegerType::get(M.getContext(), 64);
  Type *VersionTy = IntegerType::get(M.getContext(), 32);
  Function *ReoptimizeFunc = M.getFunction("__orc_rt_reoptimize");
  if (!ReoptimizeFunc) {
    std::vector<Type *> ArgTys = {MUIDTy, VersionTy};
    FunctionType *FuncTy =
        FunctionType::get(Type::getVoidTy(M.getContext()), ArgTys, false);
    ReoptimizeFunc = Function::Create(FuncTy, GlobalValue::ExternalLinkage,
                                      "__orc_rt_reoptimize", &M);
  }
  Constant *MUIDArg = ConstantInt::get(MUIDTy, MUID, false);
  Constant *CurVersionArg = ConstantInt::get(VersionTy, CurVersion, false);
  IRBuilder<> IRB(&IP);
  (void)IRB.CreateCall(ReoptimizeFunc, {MUIDArg, CurVersionArg});
}

ReOptimizeLayer::ReOptMaterializationUnitState &
ReOptimizeLayer::createMaterializationUnitState(const ThreadSafeModule &TSM) {
  std::unique_lock<std::mutex> Lock(Mutex);
  ReOptMaterializationUnitID MUID = NextID;
  MUStates.emplace(MUID,
                   ReOptMaterializationUnitState(MUID, cloneToNewContext(TSM)));
  ++NextID;
  return MUStates.at(MUID);
}

ReOptimizeLayer::ReOptMaterializationUnitState &
ReOptimizeLayer::getMaterializationUnitState(ReOptMaterializationUnitID MUID) {
  std::unique_lock<std::mutex> Lock(Mutex);
  return MUStates.at(MUID);
}

void ReOptimizeLayer::registerMaterializationUnitResource(
    ResourceKey Key, ReOptMaterializationUnitState &State) {
  std::unique_lock<std::mutex> Lock(Mutex);
  MUResources[Key].insert(State.getID());
}

Error ReOptimizeLayer::handleRemoveResources(JITDylib &JD, ResourceKey K) {
  std::unique_lock<std::mutex> Lock(Mutex);
  for (auto MUID : MUResources[K])
    MUStates.erase(MUID);

  MUResources.erase(K);
  return Error::success();
}

void ReOptimizeLayer::handleTransferResources(JITDylib &JD, ResourceKey DstK,
                                              ResourceKey SrcK) {
  std::unique_lock<std::mutex> Lock(Mutex);
  MUResources[DstK].insert_range(MUResources[SrcK]);
  MUResources.erase(SrcK);
}
