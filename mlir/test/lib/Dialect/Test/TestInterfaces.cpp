#include "TestInterfaces.h"

using namespace mlir;

bool mlir::TestEffects::Effect::classof(
    const mlir::SideEffects::Effect *effect) {
  return isa<mlir::TestEffects::Concrete>(effect);
}
