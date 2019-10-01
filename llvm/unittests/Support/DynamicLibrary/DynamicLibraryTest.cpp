//===- llvm/unittest/Support/DynamicLibrary/DynamicLibraryTest.cpp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

#include "PipSqueak.h"

using namespace llvm;
using namespace llvm::sys;

std::string LibPath(const std::string Name = "PipSqueak") {
  const std::vector<testing::internal::string> &Argvs =
      testing::internal::GetArgvs();
  const char *Argv0 =
      Argvs.size() > 0 ? Argvs[0].c_str() : "DynamicLibraryTests";
  void *Ptr = (void*)(intptr_t)TestA;
  std::string Path = fs::getMainExecutable(Argv0, Ptr);
  llvm::SmallString<256> Buf(path::parent_path(Path));
  path::append(Buf, (Name + LTDL_SHLIB_EXT).c_str());
  return Buf.str();
}

#if defined(_WIN32) || (defined(HAVE_DLFCN_H) && defined(HAVE_DLOPEN))

typedef void (*SetStrings)(std::string &GStr, std::string &LStr);
typedef void (*TestOrder)(std::vector<std::string> &V);
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

    // Test overloading local symbols does not occur by default
    GS = FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS == &TestA);
    EXPECT_EQ(StdString(GS()), "ProcessCall");

    GS = FuncPtr<GetString>(DL.getAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS == &TestA);
    EXPECT_EQ(StdString(GS()), "ProcessCall");

    // Test overloading by forcing library priority when searching for a symbol
    DynamicLibrary::SearchOrder = DynamicLibrary::SO_LoadedFirst;
    GS = FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS != &TestA);
    EXPECT_EQ(StdString(GS()), "LibCall");

    DynamicLibrary::AddSymbol("TestA", PtrFunc(&OverloadTestA));
    GS = FuncPtr<GetString>(DL.getAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS != &OverloadTestA);

    GS = FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol("TestA"));
    EXPECT_TRUE(GS != nullptr && GS == &OverloadTestA);
    EXPECT_EQ(StdString(GS()), "OverloadCall");
  }
  EXPECT_TRUE(FuncPtr<GetString>(DynamicLibrary::SearchForAddressOfSymbol(
                  "TestA")) == nullptr);

  // Check serach ordering is reset to default after call to llvm_shutdown
  EXPECT_TRUE(DynamicLibrary::SearchOrder == DynamicLibrary::SO_Linker);
}

TEST(DynamicLibrary, Shutdown) {
  std::string A("PipSqueak"), B, C("SecondLib");
  std::vector<std::string> Order;
  {
    std::string Err;
    llvm_shutdown_obj Shutdown;
    DynamicLibrary DL =
        DynamicLibrary::getPermanentLibrary(LibPath(A).c_str(), &Err);
    EXPECT_TRUE(DL.isValid());
    EXPECT_TRUE(Err.empty());

    SetStrings SS_0 = FuncPtr<SetStrings>(
        DynamicLibrary::SearchForAddressOfSymbol("SetStrings"));
    EXPECT_TRUE(SS_0 != nullptr);

    SS_0(A, B);
    EXPECT_EQ(B, "Local::Local(PipSqueak)");

    TestOrder TO_0 = FuncPtr<TestOrder>(
        DynamicLibrary::SearchForAddressOfSymbol("TestOrder"));
    EXPECT_TRUE(TO_0 != nullptr);
    
    DynamicLibrary DL2 =
        DynamicLibrary::getPermanentLibrary(LibPath(C).c_str(), &Err);
    EXPECT_TRUE(DL2.isValid());
    EXPECT_TRUE(Err.empty());

    // Should find latest version of symbols in SecondLib
    SetStrings SS_1 = FuncPtr<SetStrings>(
        DynamicLibrary::SearchForAddressOfSymbol("SetStrings"));
    EXPECT_TRUE(SS_1 != nullptr);
    EXPECT_TRUE(SS_0 != SS_1);

    TestOrder TO_1 = FuncPtr<TestOrder>(
        DynamicLibrary::SearchForAddressOfSymbol("TestOrder"));
    EXPECT_TRUE(TO_1 != nullptr);
    EXPECT_TRUE(TO_0 != TO_1);

    B.clear();
    SS_1(C, B);
    EXPECT_EQ(B, "Local::Local(SecondLib)");

    TO_0(Order);
    TO_1(Order);
  }
  EXPECT_EQ(A, "Global::~Global");
  EXPECT_EQ(B, "Local::~Local");
  EXPECT_TRUE(FuncPtr<SetStrings>(DynamicLibrary::SearchForAddressOfSymbol(
                  "SetStrings")) == nullptr);

  // Test unload/destruction ordering
  EXPECT_EQ(Order.size(), 2UL);
  EXPECT_EQ(Order.front(), "SecondLib");
  EXPECT_EQ(Order.back(), "PipSqueak");
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
