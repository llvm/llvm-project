#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "generate-pattern-catalog"

using namespace mlir;

void RewriterBase::CatalogingListener::notifyOperationInserted(
    Operation *op, InsertPoint previous) {
  LLVM_DEBUG(llvm::dbgs() << patternName << " | notifyOperationInserted"
                          << " | " << op->getName() << "\n");
  ForwardingListener::notifyOperationInserted(op, previous);
}

void RewriterBase::CatalogingListener::notifyOperationModified(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << patternName << " | notifyOperationModified"
                          << " | " << op->getName() << "\n");
  ForwardingListener::notifyOperationModified(op);
}

void RewriterBase::CatalogingListener::notifyOperationReplaced(
    Operation *op, Operation *newOp) {
  LLVM_DEBUG(llvm::dbgs() << patternName
                          << " | notifyOperationReplaced (with op)"
                          << " | " << op->getName() << " | " << newOp->getName()
                          << "\n");
  ForwardingListener::notifyOperationReplaced(op, newOp);
}

void RewriterBase::CatalogingListener::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  LLVM_DEBUG(llvm::dbgs() << patternName
                          << " | notifyOperationReplaced (with values)"
                          << " | " << op->getName() << "\n");
  ForwardingListener::notifyOperationReplaced(op, replacement);
}

void RewriterBase::CatalogingListener::notifyOperationErased(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << patternName << " | notifyOperationErased"
                          << " | " << op->getName() << "\n");
  ForwardingListener::notifyOperationErased(op);
}

void RewriterBase::CatalogingListener::notifyPatternBegin(
    const Pattern &pattern, Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << patternName << " | notifyPatternBegin"
                          << " | " << op->getName() << "\n");
  ForwardingListener::notifyPatternBegin(pattern, op);
}
