//===- WebAssemblyGlobalsTagging.cpp - Global tagging in IR ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <utility>
#include <vector>

using namespace llvm;

static const Align kTagGranuleSize = Align(16);

static bool shouldTagGlobal(GlobalVariable &G) {
  if (!G.isTagged())
    return false;

  assert(G.hasSanitizerMetadata() &&
         "Missing sanitizer metadata, but symbol is apparently tagged.");
  GlobalValue::SanitizerMetadata Meta = G.getSanitizerMetadata();

  // For now, don't instrument constant data, as it'll be in .rodata anyway. It
  // may be worth instrumenting these in future to stop them from being used as
  // gadgets.
  if (G.getName().starts_with("llvm.") || G.isThreadLocal() || G.isConstant()) {
    Meta.Memtag = false;
    G.setSanitizerMetadata(Meta);
    return false;
  }

  return true;
}

// Technically, due to ELF symbol interposition semantics, we can't change the
// alignment or size of symbols. If we increase the alignment or size of a
// symbol, the compiler may make optimisations based on this new alignment or
// size. If the symbol is interposed, this optimisation could lead to
// alignment-related or OOB read/write crashes.
//
// This is handled in the linker. When the linker sees multiple declarations of
// a global variable, and some are tagged, and some are untagged, it resolves it
// to be an untagged definition - but preserves the tag-granule-rounded size and
// tag-granule-alignment. This should prevent these kind of crashes intra-DSO.
// For cross-DSO, it's been a reasonable contract that if you're interposing a
// sanitizer-instrumented global, then the interposer also needs to be
// sanitizer-instrumented.
//
// FIXME: In theory, this can be fixed by splitting the size/alignment of
// globals into two uses: an "output alignment" that's emitted to the ELF file,
// and an "optimisation alignment" that's used for optimisation. Thus, we could
// adjust the output alignment only, and still optimise based on the pessimistic
// pre-tagging size/alignment.
static GlobalVariable *tagGlobalDefinition(Module &M, GlobalVariable *G) {
  Constant *Initializer = G->getInitializer();
  uint64_t SizeInBytes =
      M.getDataLayout().getTypeAllocSize(Initializer->getType());

  uint64_t NewSize = alignTo(SizeInBytes, kTagGranuleSize);
  if (SizeInBytes != NewSize) {
    // Pad the initializer out to the next multiple of 16 bytes.
    llvm::SmallVector<uint8_t> Init(NewSize - SizeInBytes, 0);
    Constant *Padding = ConstantDataArray::get(M.getContext(), Init);
    Initializer = ConstantStruct::getAnon({Initializer, Padding});
    auto *NewGV = new GlobalVariable(
        M, Initializer->getType(), G->isConstant(), G->getLinkage(),
        Initializer, "", G, G->getThreadLocalMode(), G->getAddressSpace());
    NewGV->copyAttributesFrom(G);
    NewGV->setComdat(G->getComdat());
    NewGV->copyMetadata(G, 0);

    NewGV->takeName(G);
    G->replaceAllUsesWith(NewGV);
    G->eraseFromParent();
    G = NewGV;
  }

  G->setAlignment(std::max(G->getAlign().valueOrOne(), kTagGranuleSize));

  // Ensure that tagged globals don't get merged by ICF - as they should have
  // different tags at runtime.
  G->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  return G;
}

/// Returns true if V is (transitively) used to initialize a global variable or
/// an alias. Such uses hold a link-time constant and cannot be replaced with a
/// tagged pointer materialized at runtime.
static bool isUsedInGlobalInitializer(const Value &V) {
  for (const User *U : V.users()) {
    if (const auto *GV = dyn_cast<GlobalVariable>(U)) {
      // llvm.used and llvm.compiler.used only keep symbols alive; their
      // contents are never dereferenced.
      if (GV->getName() == "llvm.used" || GV->getName() == "llvm.compiler.used")
        continue;
      return true;
    }
    if (isa<GlobalAlias>(U))
      return true;
    if (const auto *C = dyn_cast<Constant>(U))
      if (!isa<GlobalValue>(C) && isUsedInGlobalInitializer(*C))
        return true;
  }
  return false;
}

static bool
containsTaggedGlobal(const Constant *C,
                     const DenseSet<const GlobalVariable *> &TaggedGlobals) {
  if (const auto *G = dyn_cast<GlobalVariable>(C))
    return TaggedGlobals.contains(G);
  for (const Value *Op : C->operands())
    if (const auto *OpC = dyn_cast<Constant>(Op))
      if (containsTaggedGlobal(OpC, TaggedGlobals))
        return true;
  return false;
}

namespace {

/// Replaces every use of a tagged global inside a function with a pointer
/// whose tag was materialized by the runtime. This is the wasm equivalent of
/// AArch64 forcing tagged globals through the GOT: the tag is only known once
/// the module has been instantiated (see the "memtag.globals" descriptor
/// section emitted by the linker), so it cannot be folded into the address.
class TaggedGlobalUseMaterializer {
public:
  TaggedGlobalUseMaterializer(
      Function &F, const DenseSet<const GlobalVariable *> &TaggedGlobals,
      Function *MemtagLoad)
      : F(F), TaggedGlobals(TaggedGlobals), MemtagLoad(MemtagLoad) {}

  void run() {
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        rewriteOperands(I);
  }

private:
  Value *getTaggedPtr(GlobalVariable *G) {
    auto It = TaggedPtrs.find(G);
    if (It != TaggedPtrs.end())
      return It->second;

    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> Builder(&*Entry.getFirstInsertionPt());
    CallInst *TaggedPtr =
        Builder.CreateCall(MemtagLoad, {Builder.getInt32(0), G});
    TaggedPtrs.insert({G, TaggedPtr});
    return TaggedPtr;
  }

  Value *materializeConstantExpr(ConstantExpr *CE, Instruction *InsertBefore) {
    Instruction *NewI = CE->getAsInstruction();
    NewI->insertBefore(InsertBefore->getIterator());
    for (Use &U : NewI->operands()) {
      Value *Op = U.get();
      if (auto *G = dyn_cast<GlobalVariable>(Op);
          G && TaggedGlobals.contains(G)) {
        U.set(getTaggedPtr(G));
      } else if (auto *Nested = dyn_cast<ConstantExpr>(Op);
                 Nested && containsTaggedGlobal(Nested, TaggedGlobals)) {
        U.set(materializeConstantExpr(Nested, InsertBefore));
      }
    }
    return NewI;
  }

  void rewriteOperands(Instruction &I) {
    if (auto *CB = dyn_cast<CallBase>(&I);
        CB && CB->getCalledFunction() == MemtagLoad)
      return;

    for (Use &U : I.operands()) {
      Value *Op = U.get();
      if (auto *G = dyn_cast<GlobalVariable>(Op);
          G && TaggedGlobals.contains(G)) {
        U.set(getTaggedPtr(G));
      } else if (auto *CE = dyn_cast<ConstantExpr>(Op);
                 CE && containsTaggedGlobal(CE, TaggedGlobals)) {
        U.set(materializeConstantExpr(CE, &I));
      }
    }
  }

  Function &F;
  const DenseSet<const GlobalVariable *> &TaggedGlobals;
  Function *MemtagLoad;
  DenseMap<GlobalVariable *, CallInst *> TaggedPtrs;
};

} // anonymous namespace

namespace {
class WebAssemblyGlobalsTagging : public ModulePass {
public:
  static char ID;

  explicit WebAssemblyGlobalsTagging() : ModulePass(ID) {
    initializeWebAssemblyGlobalsTaggingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "WebAssembly Globals Tagging";
  }
};
} // anonymous namespace

char WebAssemblyGlobalsTagging::ID = 0;

bool WebAssemblyGlobalsTagging::runOnModule(Module &M) {
  // No mutating the globals in-place, or iterator invalidation occurs.
  std::vector<GlobalVariable *> GlobalsToTag;
  for (GlobalVariable &G : M.globals()) {
    if (G.isDeclaration() || !shouldTagGlobal(G))
      continue;

    // A pointer stored in a global initializer is a link-time constant; it
    // cannot be replaced by a tagged pointer materialized at runtime. Leave
    // such globals untagged, as the runtime would otherwise trap on the raw
    // pointer.
    if (isUsedInGlobalInitializer(G)) {
      GlobalValue::SanitizerMetadata Meta = G.getSanitizerMetadata();
      Meta.Memtag = false;
      G.setSanitizerMetadata(Meta);
      continue;
    }

    GlobalsToTag.push_back(&G);
  }

  DenseSet<const GlobalVariable *> TaggedGlobals;
  for (GlobalVariable *G : GlobalsToTag)
    TaggedGlobals.insert(tagGlobalDefinition(M, G));

  if (TaggedGlobals.empty())
    return false;

  Function *MemtagLoad =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_memtag_load);
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    TaggedGlobalUseMaterializer(F, TaggedGlobals, MemtagLoad).run();
  }
  return true;
}

INITIALIZE_PASS_BEGIN(WebAssemblyGlobalsTagging, "WebAssembly-globals-tagging",
                      "WebAssembly Globals Tagging Pass", false, false)
INITIALIZE_PASS_END(WebAssemblyGlobalsTagging, "WebAssembly-globals-tagging",
                    "WebAssembly Globals Tagging Pass", false, false)

ModulePass *llvm::createWebAssemblyGlobalsTaggingPass() {
  return new WebAssemblyGlobalsTagging();
}
