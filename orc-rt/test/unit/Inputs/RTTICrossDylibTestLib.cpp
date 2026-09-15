// A minimal shared library for RTTICrossDylibTest: constructs a
// CrossDylibTestError using its own linked copy of orc-rt's RTTI state, so
// that the test binary can confirm that isA<>() and casting still work when
// the object's RTTIRoot::LibraryID differs from the caller's.

#include "RTTICrossDylibTestLib.h"
#include "RTTICrossDylibTestError.h"

#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif

using namespace orc_rt;
using orc_rt_test::CrossDylibTestError;

extern "C" TEST_EXPORT ErrorInfoBase *rttiCrossDylibTest_makeError(int Code) {
  return new CrossDylibTestError(Code);
}

extern "C" TEST_EXPORT void rttiCrossDylibTest_destroyError(ErrorInfoBase *E) {
  delete E;
}

// Exposes this library's own RTTIRoot::LibraryID, so the test binary can
// confirm the two libraries are genuinely using distinct identities (i.e.
// that the test below exercises the cross-library strcmp path, not the
// same-library pointer-equality fast path).
extern "C" TEST_EXPORT const void *rttiCrossDylibTest_libraryID() {
  CrossDylibTestError E(0);
  return E.libraryID();
}
