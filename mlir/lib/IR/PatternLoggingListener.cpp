#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "pattern-logging-listener"

using namespace mlir;

void RewriterBase::PatternLoggingListener::notifyOperationInserted(
    Operation *op, InsertPoint previous) {
  LDBG() << patternName << " | notifyOperationInserted"
         << " | " << op->getName();
  ForwardingListener::notifyOperationInserted(op, previous);
}

void RewriterBase::PatternLoggingListener::notifyOperationModified(
    Operation *op) {
  LDBG() << patternName << " | notifyOperationModified"
         << " | " << op->getName();
  ForwardingListener::notifyOperationModified(op);
}

void RewriterBase::PatternLoggingListener::notifyOperationReplaced(
    Operation *op, Operation *newOp) {
  LDBG() << patternName << " | notifyOperationReplaced (with op)"
         << " | " << op->getName() << " | " << newOp->getName();
  ForwardingListener::notifyOperationReplaced(op, newOp);
}

void RewriterBase::PatternLoggingListener::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  LDBG() << patternName << " | notifyOperationReplaced (with values)"
         << " | " << op->getName();
  ForwardingListener::notifyOperationReplaced(op, replacement);
}

void RewriterBase::PatternLoggingListener::notifyOperationErased(
    Operation *op) {
  LDBG() << patternName << " | notifyOperationErased"
         << " | " << op->getName();
  ForwardingListener::notifyOperationErased(op);
}

void RewriterBase::PatternLoggingListener::notifyPatternBegin(
    const Pattern &pattern, Operation *op) {
  LDBG() << patternName << " | notifyPatternBegin"
         << " | " << op->getName();
  ForwardingListener::notifyPatternBegin(pattern, op);
}
