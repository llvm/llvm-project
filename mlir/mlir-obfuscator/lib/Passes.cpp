#include "Obfuscator/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

using namespace mlir;

namespace {

/// Simple XOR encryption for demonstration
static std::string xorEncrypt(const std::string &input, const std::string &key) {
  std::string out = input;
  for (size_t i = 0; i < input.size(); i++) {
    out[i] = input[i] ^ key[i % key.size()];
  }
  return out;
}

/// String Encryption Pass
struct StringEncryptPass 
    : public PassWrapper<StringEncryptPass, OperationPass<ModuleOp>> { // This lines makes a skeleton for the pass

  StringEncryptPass() = default;
  StringEncryptPass(const std::string &k) : key(k) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    module.walk([&](Operation *op) { // Walk visits every operation in the IR, including the nested ones
      bool changed = false;
      SmallVector<NamedAttribute> newAttrs;

      for (auto &attr : op->getAttrs()) {
        // Only encrypt string attributes
        if (auto strAttr = attr.getValue().dyn_cast<StringAttr>()) {
          std::string original = strAttr.getValue().str();
          std::string encrypted = xorEncrypt(original, key);

          auto newValue = StringAttr::get(ctx, encrypted);
          newAttrs.emplace_back(attr.getName(), newValue);
          changed = true;
        } else {
          newAttrs.push_back(attr);
        }
      }

      // Replace attribute dictionary if something changed
      if (changed) {
        op->setAttrDictionary(DictionaryAttr::get(ctx, newAttrs));
      }
    });
  }

  std::string key = "default_key";
};

} // namespace

/// Factory function exposed to the outside world
std::unique_ptr<Pass> mlir::createStringEncryptPass(llvm::StringRef key) {
  return std::make_unique<StringEncryptPass>(key.str());
}
