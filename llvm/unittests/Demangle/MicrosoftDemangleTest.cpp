//===-- MicrosoftDemangleTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/MicrosoftDemangle.h"
#include "llvm/Demangle/MicrosoftDemangleNodes.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::ms_demangle;

// After demangling a thunk function, the Signature node should retain
// NodeKind::ThunkSignature. A base-class slice assignment in
// demangleFunctionEncoding() currently overwrites the Kind field,
// causing it to become NodeKind::FunctionSignature instead.
TEST(MicrosoftDemangle, ThunkSignaturePreservesNodeKind) {
  // ?f@C@@WBA@EAAHXZ demangles to:
  //   [thunk]: public: virtual int __cdecl C::f`adjustor{16}'(void)
  // 'W' = FC_Public | FC_Virtual | FC_StaticThisAdjust
  Demangler D;
  std::string_view MangledName = "?f@C@@WBA@EAAHXZ";
  SymbolNode *S = D.parse(MangledName);
  ASSERT_FALSE(D.Error);
  ASSERT_NE(S, nullptr);
  ASSERT_EQ(S->kind(), NodeKind::FunctionSymbol);

  auto *FSN = static_cast<FunctionSymbolNode *>(S);
  ASSERT_NE(FSN->Signature, nullptr);

  // This is the key assertion: the signature must identify as a
  // ThunkSignature, not a plain FunctionSignature.
  EXPECT_EQ(FSN->Signature->kind(), NodeKind::ThunkSignature);
}
