//===--- level_zero/src/OmpWrapper.cpp --------------------------- C++ -*-===//
//
// Implement wrapper for OpenMP compatibility through dlopen
//
//===----------------------------------------------------------------------===//

#include "DLWrap.h"
#include "Shared/Debug.h"
#include "llvm/Support/DynamicLibrary.h"

#include "L0Defs.h"

DLWRAP_INITIALIZE()

DLWRAP_INTERNAL(omp_get_max_teams, 0)
DLWRAP_INTERNAL(omp_get_teams_thread_limit, 0)

DLWRAP_FINALIZE()

#ifndef TARGET_NAME
#error "Missing TARGET_NAME macro"
#endif
#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "TARGET " GETNAME(TARGET_NAME) " RTL"
#endif

static bool loadOpenMP() {
  static bool Loaded{false};
  if (Loaded)
    return true;

  const char *OpenMPLibrary = "libomp.so";
  std::string ErrMsg;

  DP("Trying to load %s\n", OpenMPLibrary);
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(OpenMPLibrary, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    if (ErrMsg.empty())
      ErrMsg = "unknown error";
    DP("Unable to load library '%s': %s!\n", OpenMPLibrary, ErrMsg.c_str());
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P == nullptr) {
      DP("Unable to find '%s' in '%s'!\n", Sym, OpenMPLibrary);
      return false;
    }
    DP("Implementing %s with dlsym(%s) -> %p\n", Sym, Sym, P);

    *dlwrap::pointer(I) = P;
  }

  return true;
}

int omp_get_max_teams() {
  if (!loadOpenMP())
    return 0;
  return dlwrap_omp_get_max_teams();
}

int omp_get_teams_thread_limit() {
  if (!loadOpenMP())
    return 0;
  return dlwrap_omp_get_teams_thread_limit();
}
