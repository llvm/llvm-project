//===- llvm/unittest/Support/DynamicLibrary/DynamicLibraryTest.cpp --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

#include "PipSqueak.h"
#include <string>

using namespace llvm;
using namespace llvm::sys;

extern "C" PIPSQUEAK_EXPORT const char *TestA() { return "ProcessCall"; }

std::string LibPath() {
  std::string Path =
      fs::getMainExecutable("DynamicLibraryTests", (void *)&TestA);
  llvm::SmallString<256> Buf(path::parent_path(Path));
  path::append(Buf, "PipSqueak.so");
  return Buf.str();
}

#if defined(_WIN32) || (defined(HAVE_DLFCN_H) && defined(HAVE_DLOPEN))

typedef void (*SetStrings)(std::string &GStr, std::string &LStr);
typedef const char *(*GetString)();

template <class T> static T FuncPtr(void *Ptr) {
  union {
    T F;
    void *P;
  } Tmp;
  Tmp.P = Ptr;
  return Tmp.F;
}
template <class T> static void* PtrFunc(T *Func) {
  union {
    T *F;
    void *P;
  } Tmp;
  Tmp.F = Func;
  return Tmp.P;
}

static const char *OverloadTestA() { return "OverloadCall"; }

std::string StdString(const char *Ptr) { return Ptr ? Ptr : ""; }

TEST(DynamicLibrary, Overload) {
  {
    std::string Err;
    llvm_shutdown_obj Shutdown;
    DynamicLibrary DL =
        DynamicLibrary::getPermanentLibrary(LibPath().c_str(), &Err);
    EXPECT_TRUE(DL.isValid());
    EXPECT_TRUE(Err.empty());

    GetString GS = FuncPtr<GetString>(DL.getAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS != &TestA);
    EXPECT_EQ(StdString(GS()), "LibCall");

    GS = FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS != &TestA);
    EXPECT_EQ(StdString(GS()), "LibCall");

    DL = DynamicLibrary::getPermanentLibrary(nullptr, &Err);
    EXPECT_TRUE(DL.isValid());
    EXPECT_TRUE(Err.empty());

    GS = FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS == &TestA);
    EXPECT_EQ(StdString(GS()), "ProcessCall");

    GS = FuncPtr<GetString>(DL.getAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS == &TestA);
    EXPECT_EQ(StdString(GS()), "ProcessCall");

    DynamicLibrary::AddSymbol("TestA", PtrFunc(&OverloadTestA));
    GS = FuncPtr<GetString>(DL.getAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS != &OverloadTestA);

    GS = FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS == &OverloadTestA);
    EXPECT_EQ(StdString(GS()), "OverloadCall");
  }
  EXPECT_TRUE(FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol(
                  "TestA")) == nullptr);
}

TEST(DynamicLibrary, Shutdown) {
  std::string A, B;
  {
    std::string Err;
    llvm_shutdown_obj Shutdown;
    DynamicLibrary DL =
        DynamicLibrary::getPermanentLibrary(LibPath().c_str(), &Err);
    EXPECT_TRUE(DL.isValid());
    EXPECT_TRUE(Err.empty());

    SetStrings SS = FuncPtr<SetStrings>(
        DynamicLibrary::SearchForAddressOfSymbol("SetStrings"));
    EXPECT_TRUE(SS != nullptr);

    SS(A, B);
    EXPECT_EQ(B, "Local::Local");
  }
  EXPECT_EQ(A, "Global::~Global");
  EXPECT_EQ(B, "Local::~Local");
  EXPECT_TRUE(FuncPtr<SetStrings>(DynamicLibrary::SearchForAddressOfSymbol(
                  "SetStrings")) == nullptr);
}

#else

TEST(DynamicLibrary, Unsupported) {
  std::string Err;
  DynamicLibrary DL =
      DynamicLibrary::getPermanentLibrary(LibPath().c_str(), &Err);
  EXPECT_FALSE(DL.isValid());
  EXPECT_EQ(Err, "dlopen() not supported on this platform");
}

#endif
