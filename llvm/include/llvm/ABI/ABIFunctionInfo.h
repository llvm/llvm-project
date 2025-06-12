//===----- ABIFunctionInfo.h - ABI Function Information ----- C++ ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines ABIFunctionInfo and associated types used in representing the
// ABI-coerced types for function arguments and return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_ABIFUNCTIONINFO_H
#define LLVM_ABI_ABIFUNCTIONINFO_H

#include "ABIInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/TrailingObjects.h"

namespace llvm {
namespace abi {

struct FunctionABIInfo {
  llvm::CallingConv::ID CC = llvm::CallingConv::C;
  llvm::CallingConv::ID EffectiveCC = llvm::CallingConv::C;

  // Core ABI attributes
  bool NoReturn = false;
  bool NoUnwind = false;
  bool HasSRet = false;
  bool IsVariadic = false;
  bool IsInstanceMethod = false;
  // Are these ABI Relavent(?)
  bool IsChainCall = false;
  bool IsDelegateCall = false;

  // Register usage controls
  bool HasRegParm = false;
  unsigned RegParm = 0;
  bool NoCallerSavedRegs = false;
  // Security/extensions(are they ABI related?)
  bool NoCfCheck = false;
  bool CmseNSCall = false;

  // Optimization hints
  bool ReturnsRetained = false;
  unsigned MaxVectorWidth = 0;

  FunctionABIInfo() = default;
  FunctionABIInfo(llvm::CallingConv::ID CC) : CC(CC), EffectiveCC(CC) {}
};

// Not an Immediate requirement for BPF
struct RequiredArgs {
private:
  unsigned NumRequired;
  static constexpr unsigned All = ~0U;

public:
  RequiredArgs() : NumRequired(All) {}
  explicit RequiredArgs(unsigned N) : NumRequired(N) {}

  static RequiredArgs forPrototypedFunction(unsigned NumArgs) {
    return RequiredArgs(NumArgs);
  }

  static RequiredArgs forVariadicFunction(unsigned NumRequired) {
    return RequiredArgs(NumRequired);
  }

  bool allowsOptionalArgs() const { return NumRequired != All; }

  unsigned getNumRequiredArgs() const {
    return allowsOptionalArgs() ? NumRequired : 0;
  }

  bool operator==(const RequiredArgs &Other) const {
    return NumRequired == Other.NumRequired;
  }
};

// Implementation detail of ABIFunctionInfo, factored out so it can be named
// in the TrailingObjects base class of ABIFunctionInfo.
struct ABIFunctionInfoArgInfo {
  const Type *ABIType;
  ABIArgInfo ArgInfo;

  ABIFunctionInfoArgInfo()
      : ABIType(nullptr), ArgInfo(ABIArgInfo::getDirect()) {}
  ABIFunctionInfoArgInfo(Type *T)
      : ABIType(T), ArgInfo(ABIArgInfo::getDirect()) {}
  ABIFunctionInfoArgInfo(Type *T, ABIArgInfo A) : ABIType(T), ArgInfo(A) {}
};

class ABIFunctionInfo final
    : public llvm::FoldingSetNode,
      private TrailingObjects<ABIFunctionInfo, ABIFunctionInfoArgInfo> {
  typedef ABIFunctionInfoArgInfo ArgInfo;

private:
  const Type *ReturnType;
  ABIArgInfo ReturnInfo;
  unsigned NumArgs;
  FunctionABIInfo ABIInfo;
  RequiredArgs
      Required; // For Variadic Functions but we can focus on this later

  ABIFunctionInfo(const Type *RetTy, unsigned NumArguments)
      : ReturnType(RetTy), ReturnInfo(ABIArgInfo::getDirect()),
        NumArgs(NumArguments) {}

  friend class TrailingObjects;

public:
  static ABIFunctionInfo *
  create(llvm::CallingConv::ID CC, const Type *ReturnType,
         llvm::ArrayRef<const Type *> ArgTypes,
         const FunctionABIInfo &ABIInfo = FunctionABIInfo(),
         RequiredArgs Required = RequiredArgs());

  const Type *getReturnType() const { return ReturnType; }
  ABIArgInfo &getReturnInfo() { return ReturnInfo; }
  const ABIArgInfo &getReturnInfo() const { return ReturnInfo; }

  llvm::CallingConv::ID getCallingConvention() const { return ABIInfo.CC; }

  const FunctionABIInfo &getExtInfo() const { return ABIInfo; }
  RequiredArgs getRequiredArgs() const { return Required; }
  llvm::ArrayRef<ArgInfo> arguments() const {
    return {getTrailingObjects<ArgInfo>(), NumArgs};
  }

  llvm::MutableArrayRef<ArgInfo> arguments() {
    return {getTrailingObjects<ArgInfo>(), NumArgs};
  }

  ArgInfo &getArgInfo(unsigned Index) {
    assert(Index < NumArgs && "Invalid argument index");
    return arguments()[Index];
  }

  const ArgInfo &getArgInfo(unsigned Index) const {
    assert(Index < NumArgs && "Invalid argument index");
    return arguments()[Index];
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(static_cast<unsigned>(ABIInfo.CC));
    ID.AddPointer(ReturnType);
    ID.AddInteger(static_cast<unsigned>(ReturnInfo.getKind()));
    if (ReturnInfo.getCoerceToType())
      ID.AddPointer(ReturnInfo.getCoerceToType());
    ID.AddInteger(NumArgs);
    for (const auto &ArgInfo : arguments()) {
      ID.AddPointer(ArgInfo.ABIType);
      ID.AddInteger(static_cast<unsigned>(ArgInfo.ArgInfo.getKind()));
      if (ArgInfo.ArgInfo.getCoerceToType())
        ID.AddPointer(ArgInfo.ArgInfo.getCoerceToType());
    }
    ID.AddInteger(Required.getNumRequiredArgs());
    ID.AddBoolean(Required.allowsOptionalArgs());
    ID.AddBoolean(ABIInfo.NoReturn);
    ID.AddBoolean(ABIInfo.IsVariadic);
    // TODO: Add more flags
  }
};
} // namespace abi
} // namespace llvm

#endif // !LLVM_ABI_ABIFUNCTIONINFO_H
