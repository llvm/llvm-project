#include "llvm/ExecutionEngine/Orc/ReOptimizeLayer.h"

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

Error ReOptimizeLayer::reigsterRuntimeFunctions(JITDylib &PlatformJD) {
  ExecutionSession::JITDispatchHandlerAssociationMap WFs;
  using ReoptimizeSPSSig = shared::SPSError(uint64_t, uint32_t, shared::SPSSequence<shared::SPSTuple<uint32_t, uint64_t>>);
  WFs[ES.intern("__orc_rt_reoptimize_tag")] =
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
    dbgs() << "Skipped" << "\n";
    TSM.withModuleDo([&](Module& M) {
      dbgs() << M << "\n";
    });
    BaseLayer.emit(std::move(R), std::move(TSM));
    return;
  }

  TSM.withModuleDo([&](Module& M){
      uint32_t ID = 1;
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
        for (auto &B : F) {
          for (auto &I : B) {
            if (auto* Call = dyn_cast<llvm::CallInst>(&I)) {
              if (Call->isIndirectCall()) {
                LLVMContext& C = Call->getContext();
                Type *I32Ty = Type::getInt32Ty(C);
                Constant *One = ConstantInt::get(I32Ty, ID);
                MDNode* N = MDNode::get(C, llvm::ValueAsMetadata::getConstant(One));
                Call->setMetadata("call_id", N);
              }
            }
          }
        }
      }
    });

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

  RSManager.emitRedirectableSymbols(std::move(R), *InitialDests);

  SymbolLookupSet LookupSymbols;
  for (auto [K, V] : *InitialDests)
    LookupSymbols.add(K);

  auto FinalSymbols =
      ES.lookup({{&JD, JITDylibLookupFlags::MatchAllSymbols}}, LookupSymbols,
                LookupKind::Static, SymbolState::Resolved);
  if (auto Err = FinalSymbols.takeError()) {
    ES.reportError(InitialDests.takeError());
    R->failMaterialization();
    return;
  }
  
  for (auto [K,V] : (*FinalSymbols)) {
    FuncAddrToMU[V.getAddress()] = {MUState.getID(), *K};
  }
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
    dbgs() << "Adding instrumentation" << "\n";
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

      for (auto &B : F) {
        for (auto& I : B) {
          if (auto* Call = dyn_cast<llvm::CallInst>(&I)) {
            if (Call->isIndirectCall()) {
                auto* VAM = cast<ValueAsMetadata>(dyn_cast<MDNode>(Call->getMetadata("call_id"))->getOperand(0));
                int CallID = dyn_cast<ConstantInt>(VAM->getValue())->getSExtValue();
                createFucnCountCall(M, I, MUID, CallID, Call->getCalledOperand());
            }
          }
        }
      }
    }
    dbgs() << M << "\n";
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
  for (auto [K, V] : RenamedMap) {
    Result[K] = (*ImplSymbols)[V];
  }


  return Result;
}

void ReOptimizeLayer::rt_reoptimize(SendErrorFn SendResult,
                                    ReOptMaterializationUnitID MUID,
                                    uint32_t CurVersion, const std::vector<std::pair<uint32_t,uint64_t>>& Profile) {

  auto &MUState = getMaterializationUnitState(MUID);
  if (CurVersion < MUState.getCurVersion() || !MUState.tryStartReoptimize()) {
    SendResult(Error::success());
    return;
  }

  ThreadSafeModule TSM = cloneToNewContext(MUState.getThreadSafeModule());
  TSM.withModuleDo([&](Module& M) {
    dbgs() << "Reoptimization requested for" << M.getName() << "\n";
  });

  auto OldRT = MUState.getResourceTracker();
  auto &JD = OldRT->getJITDylib();

  if (auto Err = ReOptFunc(*this, MUID, CurVersion + 1, OldRT, Profile, TSM)) {
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

Expected<Constant *> ReOptimizeLayer::createReoptimizeArgBuffer(
    Module &M, ReOptMaterializationUnitID MUID, uint32_t CurVersion) {
  size_t ArgBufferSize = SPSReoptimizeArgList::size(MUID, CurVersion);
  std::vector<char> ArgBuffer(ArgBufferSize);
  shared::SPSOutputBuffer OB(ArgBuffer.data(), ArgBuffer.size());
  if (!SPSReoptimizeArgList::serialize(OB, MUID, CurVersion))
    return make_error<StringError>("Could not serealize args list",
                                   inconvertibleErrorCode());
  return ConstantDataArray::get(M.getContext(), ArrayRef(ArgBuffer));
}

void ReOptimizeLayer::createReoptimizeCall(Module &M, Instruction &IP,
                                           ReOptMaterializationUnitID MUID, uint32_t CurVersion) {
  Function *DispatchFunc = M.getFunction("__orc_rt_reoptimize");
  if (!DispatchFunc) {
    std::vector<Type *> Args = {IntegerType::get(M.getContext(), 64), IntegerType::get(M.getContext(), 32)};
    FunctionType *FuncTy =
        FunctionType::get(Type::getVoidTy(M.getContext()), Args, false);
    DispatchFunc = Function::Create(FuncTy, GlobalValue::ExternalLinkage,
                                    "__orc_rt_reoptimize", &M);
  }
  Constant *MUIDConst = ConstantInt::get(
      IntegerType::get(M.getContext(), 64), MUID, false);
  Constant *VersionConst= ConstantInt::get(
      IntegerType::get(M.getContext(), 32), CurVersion, false);
  IRBuilder<> IRB(&IP);
  (void)IRB.CreateCall(DispatchFunc,
                       {MUIDConst, VersionConst});
}

// Create IR reoptimize request fucntion call.
void ReOptimizeLayer::createFucnCountCall(Module &M, Instruction &IP, ReOptMaterializationUnitID MUID, uint32_t CallID, Value* FuncPtr) {
  Function *DispatchFunc = M.getFunction("__orc_rt_increment_call_count");
  if (!DispatchFunc) {
    std::vector<Type *> Args = {IntegerType::get(M.getContext(), 64), IntegerType::get(M.getContext(), 32), Type::getInt8PtrTy(M.getContext())};
    FunctionType *FuncTy =
        FunctionType::get(Type::getVoidTy(M.getContext()), Args, false);
    DispatchFunc = Function::Create(FuncTy, GlobalValue::ExternalLinkage,
                                    "__orc_rt_increment_call_count", &M);
  }
  Constant *MUIDConst = ConstantInt::get(
      IntegerType::get(M.getContext(), 64), MUID, false);
  Constant *CallIDConst = ConstantInt::get(
      IntegerType::get(M.getContext(), 32), CallID, false);
  IRBuilder<> IRB(&IP);
  (void)IRB.CreateCall(DispatchFunc,
                       {MUIDConst, CallIDConst, FuncPtr});
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
  MUResources[DstK].insert(MUResources[SrcK].begin(), MUResources[SrcK].end());
  MUResources.erase(SrcK);
}
