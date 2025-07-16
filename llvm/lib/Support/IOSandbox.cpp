#include "llvm/Support/IOSandbox.h"

#include <cassert>

using namespace llvm;

thread_local bool IOSandboxEnabled = false;

SaveAndRestore<bool> sys::sandbox_scoped_enable() {
  return {IOSandboxEnabled, true};
}

SaveAndRestore<bool> sys::sandbox_scoped_disable() {
  return {IOSandboxEnabled, false};
}

void sys::sandbox_violation_if_enabled() {
  assert(!IOSandboxEnabled && "sandbox violation");
}
