#include "Obfuscator/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <random>

using namespace mlir;

namespace {

/// Utility: generate random obfuscated names (hex-based)
static std::string generateObfuscatedName(std::mt19937 &rng) {
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
  uint32_t num = dist(rng);

  // format as hex string: f_a1b2c3d4
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "f_%08x", num);
  return std::string(buffer);
}

/// Symbol Obfuscation Pass
struct SymbolObfuscatePass
    : public PassWrapper<SymbolObfuscatePass, OperationPass<ModuleOp>> {

  SymbolObfuscatePass() = default;
  SymbolObfuscatePass(const std::string &k) : key(k) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    SymbolTable symbolTable(module);

    // initialize RNG with deterministic seed (from key)
    std::seed_seq seq(key.begin(), key.end());
    std::mt19937 rng(seq);

    // Mapping: oldName -> newName
    llvm::StringMap<std::string> renameMap;

    // Step 1: Rename symbol definitions (functions)
    module.walk([&](func::FuncOp func) {
      StringRef oldName = func.getName();
      std::string newName = generateObfuscatedName(rng);

      renameMap[oldName] = newName;
      symbolTable.setSymbolName(func, newName);
    });

    // Step 2: Update symbol references everywhere
    module.walk([&](Operation *op) {
      SmallVector<NamedAttribute> updatedAttrs;
      bool changed = false;

      for (auto &attr : op->getAttrs()) {
        if (auto symAttr = attr.getValue().dyn_cast<SymbolRefAttr>()) {
          StringRef old = symAttr.getRootReference();
          if (renameMap.count(old)) {
            auto newRef = SymbolRefAttr::get(ctx, renameMap[old]);
            updatedAttrs.emplace_back(attr.getName(), newRef);
            changed = true;
            continue;
          }
        }
        // no change -> keep original
        updatedAttrs.push_back(attr);
      }

      if (changed) {
        op->setAttrDictionary(DictionaryAttr::get(ctx, updatedAttrs));
      }
    });
  }

  std::string key = "seed";
};

} // namespace

/// Public factory
std::unique_ptr<Pass> mlir::createSymbolObfuscatePass(llvm::StringRef key) {
  return std::make_unique<SymbolObfuscatePass>(key.str());
}
