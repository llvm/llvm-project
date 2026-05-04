//===-- EJitRegistration.cpp - AOT Registration Callbacks -----------------===//

#include "llvm/ExecutionEngine/EJIT/EJitRegistration.h"
#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"

using namespace llvm::ejit;

extern "C" {

void ejit_register_bitcode(const char *funcName,
                           const uint8_t *bitcodeData, uint64_t bitcodeSize) {
  EJitRegistrationStore::instance().registerBitcode(
      funcName, bitcodeData, static_cast<size_t>(bitcodeSize));
}

void ejit_register_period_array(const char *periodName,
                                const char *varName,
                                void *baseAddr, uint64_t arraySize) {
  EJitRegistrationStore::instance().registerPeriodArray(
      periodName, varName, baseAddr, arraySize);
}

void ejit_register_static_var(const char *varName, void *varAddr) {
  EJitRegistrationStore::instance().registerStaticVar(varName, varAddr);
}

} // extern "C"
