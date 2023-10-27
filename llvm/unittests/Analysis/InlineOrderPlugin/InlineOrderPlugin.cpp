#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/InlineOrder.h"

using namespace llvm;

namespace {

class NoFooInlineOrder : public InlineOrder<std::pair<CallBase *, int>> {
public:
  NoFooInlineOrder(FunctionAnalysisManager &FAM, const InlineParams &Params,
                   ModuleAnalysisManager &MAM, Module &M) {
    DefaultInlineOrder = getDefaultInlineOrder(FAM, Params, MAM, M);
  }
  size_t size() override { return DefaultInlineOrder->size(); }
  void push(const std::pair<CallBase *, int> &Elt) override {
    // We ignore calles named "foo"
    if (Elt.first->getCalledFunction()->getName() == "foo") {
      DefaultInlineOrder->push(Elt);
    }
  }
  std::pair<CallBase *, int> pop() override {
    return DefaultInlineOrder->pop();
  }
  void erase_if(function_ref<bool(std::pair<CallBase *, int>)> Pred) override {
    DefaultInlineOrder->erase_if(Pred);
  }

private:
  std::unique_ptr<InlineOrder<std::pair<CallBase *, int>>> DefaultInlineOrder;
};

std::unique_ptr<InlineOrder<std::pair<CallBase *, int>>>
NoFooInlineOrderFactory(FunctionAnalysisManager &FAM,
                        const InlineParams &Params, ModuleAnalysisManager &MAM,
                        Module &M) {
  return std::make_unique<NoFooInlineOrder>(FAM, Params, MAM, M);
}

} // namespace

/* New PM Registration */
llvm::PassPluginLibraryInfo getDefaultDynamicInlineOrderPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DynamicDefaultInlineOrder",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            // We use the PassBuilder's callback mechanism
            // to register our Analysis: this will register
            // our PluginInlineOrderAnalysis instance with
            // the ModuleAnalysisManager
            PB.registerAnalysisRegistrationCallback(
                [](ModuleAnalysisManager &MAM) {
                  MAM.registerPass([] {
                    // defaultInlineOrderFactory will be
                    // used to create an InlineOrder
                    return PluginInlineOrderAnalysis(NoFooInlineOrderFactory);
                  });
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getDefaultDynamicInlineOrderPluginInfo();
}
