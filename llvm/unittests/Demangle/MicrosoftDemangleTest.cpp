//===-- MicrosoftDemangleTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <string>
#include <string_view>

using namespace llvm;

namespace {
std::string microsoftDemangleString(std::string_view MangledName,
                                    MSDemangleFlags Flags) {
  char *Demangled = microsoftDemangle(MangledName, nullptr, nullptr, Flags);
  if (!Demangled)
    return std::string(MangledName);
  std::string Result(Demangled);
  std::free(Demangled);
  return Result;
}
} // namespace

TEST(MicrosoftDemangle, demangleFunction) {
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXXZ", MSDF_None),
            "void __cdecl foo(void)");
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXXZ", MSDF_NoReturnType),
            "__cdecl foo(void)");
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXXZ", MSDF_NoCallingConvention),
            "void foo(void)");
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXXZ", MSDF_NoVoidParameter),
            "void __cdecl foo()");
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXXZ",
                                    MSDemangleFlags(MSDF_NoCallingConvention |
                                                    MSDF_NoReturnType |
                                                    MSDF_NoVoidParameter)),
            "foo()");
}

TEST(MicrosoftDemangle, demangleFunctionWithParameters) {
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXHH@Z", MSDF_None),
            "void __cdecl foo(int, int)");
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXHH@Z", MSDF_NoReturnType),
            "__cdecl foo(int, int)");
  EXPECT_EQ(microsoftDemangleString("?foo@@YAXHH@Z", MSDF_NoCallingConvention),
            "void foo(int, int)");
  EXPECT_EQ(microsoftDemangleString(
                "?foo@@YAXHH@Z",
                MSDemangleFlags(MSDF_NoReturnType | MSDF_NoCallingConvention)),
            "foo(int, int)");
}

TEST(MicrosoftDemangle, demangleMethod) {
  EXPECT_EQ(microsoftDemangleString("?method@Struct@@QEAAXXZ", MSDF_None),
            "public: void __cdecl Struct::method(void)");
  EXPECT_EQ(
      microsoftDemangleString("?method@Struct@@QEAAXXZ", MSDF_NoReturnType),
      "public: __cdecl Struct::method(void)");
  EXPECT_EQ(microsoftDemangleString("?method@Struct@@QEAAXXZ",
                                    MSDF_NoCallingConvention),
            "public: void Struct::method(void)");
  EXPECT_EQ(microsoftDemangleString("?method@Struct@@QEAAXXZ",
                                    MSDF_NoAccessSpecifier),
            "void __cdecl Struct::method(void)");
  EXPECT_EQ(
      microsoftDemangleString("?method@Struct@@QEAAXXZ", MSDF_NoVoidParameter),
      "public: void __cdecl Struct::method()");
  EXPECT_EQ(
      microsoftDemangleString(
          "?method@Struct@@QEAAXXZ",
          MSDemangleFlags(MSDF_NoCallingConvention | MSDF_NoAccessSpecifier |
                          MSDF_NoReturnType | MSDF_NoVoidParameter)),
      "Struct::method()");
  EXPECT_EQ(
      microsoftDemangleString("?privateMethod@Struct@@AEAAXXZ", MSDF_None),
      "private: void __cdecl Struct::privateMethod(void)");
  EXPECT_EQ(
      microsoftDemangleString("?protectedMethod@Struct@@IEAAXXZ", MSDF_None),
      "protected: void __cdecl Struct::protectedMethod(void)");
}

TEST(MicrosoftDemangle, demangleConstMethod) {
  EXPECT_EQ(microsoftDemangleString("?constMethod@Struct@@QEBAXXZ", MSDF_None),
            "public: void __cdecl Struct::constMethod(void) const");
  EXPECT_EQ(microsoftDemangleString("?constMethod@Struct@@QEBAXXZ",
                                    MSDF_NoReturnType),
            "public: __cdecl Struct::constMethod(void) const");
  EXPECT_EQ(microsoftDemangleString("?constMethod@Struct@@QEBAXXZ",
                                    MSDF_NoCallingConvention),
            "public: void Struct::constMethod(void) const");
  EXPECT_EQ(microsoftDemangleString("?constMethod@Struct@@QEBAXXZ",
                                    MSDF_NoVoidParameter),
            "public: void __cdecl Struct::constMethod() const");
  EXPECT_EQ(microsoftDemangleString("?constMethod@Struct@@QEBAXXZ",
                                    MSDemangleFlags(MSDF_NoReturnType |
                                                    MSDF_NoCallingConvention |
                                                    MSDF_NoVoidParameter)),
            "public: Struct::constMethod() const");
}

TEST(MicrosoftDemangle, demangleVirtualMethod) {
  EXPECT_EQ(microsoftDemangleString("?vfunc@Base@@UEAAXXZ", MSDF_None),
            "public: virtual void __cdecl Base::vfunc(void)");
  EXPECT_EQ(microsoftDemangleString("?vfunc@Base@@UEAAXXZ", MSDF_NoReturnType),
            "public: virtual __cdecl Base::vfunc(void)");
  EXPECT_EQ(
      microsoftDemangleString("?vfunc@Base@@UEAAXXZ", MSDF_NoAccessSpecifier),
      "virtual void __cdecl Base::vfunc(void)");
  EXPECT_EQ(microsoftDemangleString(
                "?vfunc@Base@@UEAAXXZ",
                MSDemangleFlags(MSDF_NoReturnType | MSDF_NoAccessSpecifier)),
            "virtual __cdecl Base::vfunc(void)");
}

TEST(MicrosoftDemangle, demangleStaticMethod) {
  EXPECT_EQ(microsoftDemangleString("?staticMethod@Struct@@SAXXZ", MSDF_None),
            "public: static void __cdecl Struct::staticMethod(void)");
  EXPECT_EQ(
      microsoftDemangleString("?staticMethod@Struct@@SAXXZ", MSDF_NoReturnType),
      "public: static __cdecl Struct::staticMethod(void)");
  EXPECT_EQ(microsoftDemangleString("?staticMethod@Struct@@SAXXZ",
                                    MSDF_NoAccessSpecifier),
            "static void __cdecl Struct::staticMethod(void)");
  EXPECT_EQ(microsoftDemangleString("?staticMethod@Struct@@SAXXZ",
                                    MSDF_NoCallingConvention),
            "public: static void Struct::staticMethod(void)");
  EXPECT_EQ(microsoftDemangleString("?staticMethod@Struct@@SAXXZ",
                                    MSDemangleFlags(MSDF_NoReturnType |
                                                    MSDF_NoAccessSpecifier |
                                                    MSDF_NoCallingConvention)),
            "static Struct::staticMethod(void)");
}

TEST(MicrosoftDemangle, demangleConstructor) {
  EXPECT_EQ(microsoftDemangleString("??0Animal@@QEAA@XZ", MSDF_None),
            "public: __cdecl Animal::Animal(void)");
  EXPECT_EQ(microsoftDemangleString("??0Animal@@QEAA@XZ", MSDF_NoReturnType),
            "public: __cdecl Animal::Animal(void)");
  EXPECT_EQ(
      microsoftDemangleString("??0Animal@@QEAA@XZ", MSDF_NoCallingConvention),
      "public: Animal::Animal(void)");
  EXPECT_EQ(
      microsoftDemangleString("??0Animal@@QEAA@XZ", MSDF_NoAccessSpecifier),
      "__cdecl Animal::Animal(void)");
  EXPECT_EQ(microsoftDemangleString("??0Animal@@QEAA@XZ", MSDF_NoVoidParameter),
            "public: __cdecl Animal::Animal()");
  EXPECT_EQ(
      microsoftDemangleString(
          "??0Animal@@QEAA@XZ",
          MSDemangleFlags(MSDF_NoCallingConvention | MSDF_NoAccessSpecifier |
                          MSDF_NoReturnType | MSDF_NoVoidParameter)),
      "Animal::Animal()");
}

TEST(MicrosoftDemangle, demangleDestructor) {
  EXPECT_EQ(microsoftDemangleString("??1Animal@@QEAA@XZ", MSDF_None),
            "public: __cdecl Animal::~Animal(void)");
  EXPECT_EQ(microsoftDemangleString("??1Animal@@QEAA@XZ", MSDF_NoReturnType),
            "public: __cdecl Animal::~Animal(void)");
  EXPECT_EQ(
      microsoftDemangleString("??1Animal@@QEAA@XZ", MSDF_NoCallingConvention),
      "public: Animal::~Animal(void)");
  EXPECT_EQ(
      microsoftDemangleString("??1Animal@@QEAA@XZ", MSDF_NoAccessSpecifier),
      "__cdecl Animal::~Animal(void)");
  EXPECT_EQ(microsoftDemangleString("??1Animal@@QEAA@XZ", MSDF_NoVoidParameter),
            "public: __cdecl Animal::~Animal()");
  EXPECT_EQ(
      microsoftDemangleString(
          "??1Animal@@QEAA@XZ",
          MSDemangleFlags(MSDF_NoCallingConvention | MSDF_NoAccessSpecifier |
                          MSDF_NoReturnType | MSDF_NoVoidParameter)),
      "Animal::~Animal()");
}

TEST(MicrosoftDemangle, demangleOperatorPlus) {
  EXPECT_EQ(microsoftDemangleString("??HInt@@QEAAHH@Z", MSDF_None),
            "public: int __cdecl Int::operator+(int)");
  EXPECT_EQ(microsoftDemangleString("??HInt@@QEAAHH@Z", MSDF_NoReturnType),
            "public: __cdecl Int::operator+(int)");
  EXPECT_EQ(
      microsoftDemangleString("??HInt@@QEAAHH@Z", MSDF_NoCallingConvention),
      "public: int Int::operator+(int)");
  EXPECT_EQ(microsoftDemangleString(
                "??HInt@@QEAAHH@Z",
                MSDemangleFlags(MSDF_NoReturnType | MSDF_NoCallingConvention)),
            "public: Int::operator+(int)");
}

TEST(MicrosoftDemangle, demangleCopyOperator) {
  EXPECT_EQ(microsoftDemangleString("??0Dog@@QEAA@AEBU0@@Z", MSDF_None),
            "public: __cdecl Dog::Dog(struct Dog const &)");
  EXPECT_EQ(microsoftDemangleString("??0Dog@@QEAA@AEBU0@@Z", MSDF_NoReturnType),
            "public: __cdecl Dog::Dog(struct Dog const &)");
  EXPECT_EQ(microsoftDemangleString("??0Dog@@QEAA@AEBU0@@Z",
                                    MSDF_NoCallingConvention),
            "public: Dog::Dog(struct Dog const &)");
  EXPECT_EQ(
      microsoftDemangleString("??0Dog@@QEAA@AEBU0@@Z", MSDF_NoAccessSpecifier),
      "__cdecl Dog::Dog(struct Dog const &)");
  EXPECT_EQ(
      microsoftDemangleString("??0Dog@@QEAA@AEBU0@@Z", MSDF_NoTagSpecifier),
      "public: __cdecl Dog::Dog(Dog const &)");
  EXPECT_EQ(microsoftDemangleString("??0Dog@@QEAA@AEBU0@@Z",
                                    MSDemangleFlags(MSDF_NoCallingConvention |
                                                    MSDF_NoAccessSpecifier |
                                                    MSDF_NoTagSpecifier)),
            "Dog::Dog(Dog const &)");
}

TEST(MicrosoftDemangle, demangleAssignmentOperator) {
  EXPECT_EQ(microsoftDemangleString("??4Dog@@QEAAAEAU0@AEBU0@@Z", MSDF_None),
            "public: struct Dog & __cdecl Dog::operator=(struct Dog const &)");
  EXPECT_EQ(
      microsoftDemangleString("??4Dog@@QEAAAEAU0@AEBU0@@Z", MSDF_NoReturnType),
      "public: __cdecl Dog::operator=(struct Dog const &)");
  EXPECT_EQ(microsoftDemangleString("??4Dog@@QEAAAEAU0@AEBU0@@Z",
                                    MSDF_NoCallingConvention),
            "public: struct Dog & Dog::operator=(struct Dog const &)");
  EXPECT_EQ(microsoftDemangleString("??4Dog@@QEAAAEAU0@AEBU0@@Z",
                                    MSDF_NoAccessSpecifier),
            "struct Dog & __cdecl Dog::operator=(struct Dog const &)");
  EXPECT_EQ(microsoftDemangleString("??4Dog@@QEAAAEAU0@AEBU0@@Z",
                                    MSDF_NoTagSpecifier),
            "public: Dog & __cdecl Dog::operator=(Dog const &)");
  EXPECT_EQ(microsoftDemangleString(
                "??4Dog@@QEAAAEAU0@AEBU0@@Z",
                MSDemangleFlags(MSDF_NoReturnType | MSDF_NoCallingConvention |
                                MSDF_NoAccessSpecifier | MSDF_NoTagSpecifier)),
            "Dog::operator=(Dog const &)");
}

TEST(MicrosoftDemangle, demangleType) {
  EXPECT_EQ(microsoftDemangleString(".?AUStruct@@", MSDF_None),
            "struct Struct `RTTI Type Descriptor Name'");
  EXPECT_EQ(microsoftDemangleString(".?AUStruct@@", MSDF_NoTagSpecifier),
            "Struct `RTTI Type Descriptor Name'");
  EXPECT_EQ(microsoftDemangleString(".?AUStruct@@",
                                    MSDF_NoDecorativeRTTITypeDescriptor),
            "struct Struct");
  EXPECT_EQ(
      microsoftDemangleString(
          ".?AUStruct@@", MSDemangleFlags(MSDF_NoTagSpecifier |
                                          MSDF_NoDecorativeRTTITypeDescriptor)),
      "Struct");
}

TEST(MicrosoftDemangle, demangleVariable) {
  EXPECT_EQ(microsoftDemangleString("?globalVar@@3HA", MSDF_None),
            "int globalVar");
  EXPECT_EQ(microsoftDemangleString("?globalVar@@3HA", MSDF_NoVariableType),
            "globalVar");
  // This flag should not impact variables
  EXPECT_EQ(microsoftDemangleString("?globalVar@@3HA",
                                    MSDF_NoDecorativeRTTITypeDescriptor),
            "int globalVar");
}

TEST(MicrosoftDemangle, demangleMemberVariable) {
  EXPECT_EQ(microsoftDemangleString("?var@Struct@@2HA", MSDF_None),
            "public: static int Struct::var");
  EXPECT_EQ(microsoftDemangleString("?var@Struct@@2HA", MSDF_NoVariableType),
            "public: static Struct::var");
  EXPECT_EQ(microsoftDemangleString("?var@Struct@@2HA", MSDF_NoAccessSpecifier),
            "static int Struct::var");
  EXPECT_EQ(microsoftDemangleString("?var@Struct@@2HA", MSDF_NoMemberType),
            "public: int Struct::var");
  EXPECT_EQ(microsoftDemangleString("?var@Struct@@2HA",
                                    MSDemangleFlags(MSDF_NoAccessSpecifier |
                                                    MSDF_NoVariableType |
                                                    MSDF_NoMemberType)),
            "Struct::var");
}

TEST(MicrosoftDemangle, demangleIndirectVariables) {
  EXPECT_EQ(microsoftDemangleString("?ptr@@3PEAHA", MSDF_None), "int *ptr");
  EXPECT_EQ(microsoftDemangleString("?ptr@@3PEAHA", MSDF_NoVariableType),
            "ptr");
  EXPECT_EQ(microsoftDemangleString("?ptrConst@@3PEBHA", MSDF_None),
            "int const *ptrConst");
  EXPECT_EQ(microsoftDemangleString("?ptrPtr@@3PEAPEAHA", MSDF_None),
            "int **ptrPtr");
  EXPECT_EQ(microsoftDemangleString("?ref@@3AEAHA", MSDF_None), "int &ref");
}

TEST(MicrosoftDemangle, demangleFunctionPointers) {
  EXPECT_EQ(microsoftDemangleString("?funcPtr@@3P6AXXZA", MSDF_None),
            "void (__cdecl *funcPtr)(void)");
  EXPECT_EQ(microsoftDemangleString("?funcPtr@@3P6AXXZA", MSDF_NoVariableType),
            "funcPtr");
  EXPECT_EQ(microsoftDemangleString("?funcPtr@@3P6AHXZA", MSDF_NoVoidParameter),
            "int (__cdecl *funcPtr)()");
}

TEST(MicrosoftDemangle, demangleNestedClass) {
  EXPECT_EQ(microsoftDemangleString("?inner@Outer@@QEAAXXZ", MSDF_None),
            "public: void __cdecl Outer::inner(void)");
  EXPECT_EQ(microsoftDemangleString("?inner@Outer@@QEAAXXZ", MSDF_NoReturnType),
            "public: __cdecl Outer::inner(void)");
  EXPECT_EQ(
      microsoftDemangleString("?inner@Outer@@QEAAXXZ", MSDF_NoAccessSpecifier),
      "void __cdecl Outer::inner(void)");
  EXPECT_EQ(microsoftDemangleString(
                "?inner@Outer@@QEAAXXZ",
                MSDemangleFlags(MSDF_NoReturnType | MSDF_NoAccessSpecifier)),
            "__cdecl Outer::inner(void)");
}

TEST(MicrosoftDemangle, demangleStdcallFunction) {
  EXPECT_EQ(microsoftDemangleString("?stdcallFunc@@YGXXZ", MSDF_None),
            "void __stdcall stdcallFunc(void)");
  EXPECT_EQ(
      microsoftDemangleString("?stdcallFunc@@YGXXZ", MSDF_NoCallingConvention),
      "void stdcallFunc(void)");
}
