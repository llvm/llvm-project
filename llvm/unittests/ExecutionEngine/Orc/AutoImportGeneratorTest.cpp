//===- AutoImportGeneratorTest.cpp - AutoImportGenerator unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include "OrcTestCommon.h"

// AutoImportGenerator itself builds the __imp_ pointer slot and the jump-thunk
// (it defines them in its own stub graph) and resolves the imported address
// through the host's dynamic loader (GetProcAddress/dlsym). These tests cover
// the generator in isolation: its slot+thunk synthesis, its export-table
// authority, and its ResourceTracker lifecycle.
//
// The tests execute the generator's thunk in-process, so an x86_64 host is
// required (the thunk is native code we call); they import a no-argument,
// integer-returning function the host exports -- kernel32!GetCurrentProcessId
// on Windows, libc getpid on POSIX. Such functions are ABI-safe to call
// regardless of the stub graph's calling convention: no argument registers to
// mismatch, and the result comes back in EAX under both the Win64 and SysV
// ABIs.
#if defined(__x86_64__) || defined(_M_X64)

#if defined(_WIN32)
// Declared here (rather than via <windows.h>) to avoid the Windows macro soup
// in an LLVM unit test. kernel32 is auto-linked, so these resolve at link time.
extern "C" unsigned long GetCurrentProcessId(void);
extern "C" unsigned long GetCurrentThreadId(void);
#define AIG_IMPORT_LIB "kernel32.dll"
#define AIG_SYM1 "GetCurrentProcessId"
#define AIG_SYM2 "GetCurrentThreadId"
#else
#include <unistd.h>
#define AIG_IMPORT_LIB nullptr // resolve against the current process (libc)
#define AIG_SYM1 "getpid"
#define AIG_SYM2 "getppid"
#endif

using namespace llvm;
using namespace llvm::orc;

namespace {

// Call the real AIG_SYM1 directly, to compare against the thunk's result. Both
// the POSIX and Windows choices return a 32-bit value in EAX, so we read the
// thunk as a 32-bit-returning function.
static unsigned callRealSym1() {
#if defined(_WIN32)
  return static_cast<unsigned>(GetCurrentProcessId());
#else
  return static_cast<unsigned>(::getpid());
#endif
}

class AutoImportGeneratorTest : public testing::Test {
public:
  ~AutoImportGeneratorTest() override {
    if (auto Err = ES.endSession())
      ES.reportError(std::move(Err));
  }

protected:
  // Use SelfExecutorProcessControl so the generator has a real (in-process)
  // DylibManager to load the library and resolve its exports through. The
  // architecture (x86_64) is the only real constraint on synthesis; the
  // synthesized slot+thunk are linked and executed in-process below.
  ExecutionSession ES{cantFail(SelfExecutorProcessControl::Create())};
  std::unique_ptr<DylibManager> DylibMgr{
      cantFail(ES.getExecutorProcessControl().createDefaultDylibMgr())};
  JITDylib &JD = ES.createBareJITDylib("main");
  ObjectLinkingLayer ObjLinkingLayer{
      ES, std::make_unique<jitlink::InProcessMemoryManager>(4096)};

  // The address the generator itself will resolve a name to, fetched through
  // the very same loader path (DynamicLibrary::getAddressOfSymbol on the same
  // library) so the slot contents can be compared exactly.
  void *realAddr(const char *Name) {
    std::string Err;
    auto Lib = sys::DynamicLibrary::getPermanentLibrary(AIG_IMPORT_LIB, &Err);
    EXPECT_TRUE(Lib.isValid()) << Err;
    return Lib.getAddressOfSymbol(Name);
  }
};

// For a symbol the library exports, the generator must synthesize both an
// __imp_X IAT slot holding X's real address and an X thunk that jumps through
// it -- and both must be usable. This covers the dllimport (__imp_-mediated)
// path and the direct-call path in one shot.
TEST_F(AutoImportGeneratorTest, SynthesizesImpSlotAndThunk) {
  void *RealSym1 = realAddr(AIG_SYM1);
  ASSERT_NE(RealSym1, nullptr);

  auto AIGOrErr =
      AutoImportGenerator::Load(ES, ObjLinkingLayer, *DylibMgr, AIG_IMPORT_LIB);
  ASSERT_THAT_EXPECTED(AIGOrErr, Succeeded());
  JD.addGenerator(std::move(*AIGOrErr));

  // The __imp_ slot holds the symbol's real address in the library.
  auto ImpSym = ES.lookup(&JD, "__imp_" AIG_SYM1);
  ASSERT_THAT_EXPECTED(ImpSym, Succeeded());
  void **Slot = ImpSym->getAddress().toPtr<void **>();
  EXPECT_EQ(*Slot, RealSym1);

  // The thunk is a distinct, synthesized definition (so &X yields the thunk,
  // not the implementation in the library) ...
  auto ThunkSym = ES.lookup(&JD, AIG_SYM1);
  ASSERT_THAT_EXPECTED(ThunkSym, Succeeded());
  EXPECT_NE(ThunkSym->getAddress(), ImpSym->getAddress());
  EXPECT_NE(ThunkSym->getAddress().toPtr<void *>(), RealSym1);

  // ... and calling it jumps through the slot to the real implementation.
  auto Thunk = ThunkSym->getAddress().toPtr<unsigned (*)()>();
  EXPECT_EQ(Thunk(), callRealSym1());
}

// The library's export table is the authority: a name it does not export must
// be left unresolved, so the link fails exactly as a static link would.
TEST_F(AutoImportGeneratorTest, UnexportedSymbolFailsToLink) {
  // The failed lookup surfaces as an Expected error; swallow any asynchronous
  // report so it does not pollute the test log.
  ES.setErrorReporter(consumeError);

  auto AIGOrErr =
      AutoImportGenerator::Load(ES, ObjLinkingLayer, *DylibMgr, AIG_IMPORT_LIB);
  ASSERT_THAT_EXPECTED(AIGOrErr, Succeeded());
  JD.addGenerator(std::move(*AIGOrErr));

  EXPECT_THAT_EXPECTED(
      ES.lookup(&JD, "__imp_this_symbol_is_definitely_not_exported_zzz"),
      Failed());
}

// All synthesized stubs are owned by a single, generator-managed
// ResourceTracker that the client can reclaim in one step; a subsequent import
// transparently starts a fresh tracker.
TEST_F(AutoImportGeneratorTest, StubsResourceTrackerLifecycle) {
  auto AIGOrErr =
      AutoImportGenerator::Load(ES, ObjLinkingLayer, *DylibMgr, AIG_IMPORT_LIB);
  ASSERT_THAT_EXPECTED(AIGOrErr, Succeeded());
  AutoImportGenerator &AIG = **AIGOrErr;
  JD.addGenerator(std::move(*AIGOrErr));

  // No stubs synthesized yet.
  EXPECT_EQ(AIG.getImportStubsResourceTracker(), nullptr);

  ASSERT_THAT_EXPECTED(ES.lookup(&JD, "__imp_" AIG_SYM1), Succeeded());
  ResourceTrackerSP RT1 = AIG.getImportStubsResourceTracker();
  ASSERT_NE(RT1, nullptr);
  EXPECT_FALSE(RT1->isDefunct());

  // Reclaim every synthesized slot and thunk in one step, without tearing down
  // the JITDylib.
  EXPECT_THAT_ERROR(RT1->remove(), Succeeded());
  EXPECT_TRUE(RT1->isDefunct());

  // A later import transparently starts a fresh tracker.
  ASSERT_THAT_EXPECTED(ES.lookup(&JD, "__imp_" AIG_SYM2), Succeeded());
  ResourceTrackerSP RT2 = AIG.getImportStubsResourceTracker();
  ASSERT_NE(RT2, nullptr);
  EXPECT_NE(RT2, RT1);
  EXPECT_FALSE(RT2->isDefunct());
}

} // namespace

#endif // x86_64
