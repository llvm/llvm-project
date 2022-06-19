//===- VFABIDemangling.cpp - Vector Function ABI demangling utilities. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"

using namespace llvm;

namespace {
/// Utilities for the Vector Function ABI name parser.

/// Return types for the parser functions.
enum class ParseRet {
  OK,   // Found.
  None, // Not found.
  Error // Syntax error.
};

/// Extracts the `<isa>` information from the mangled string, and
/// sets the `ISA` accordingly.
ParseRet tryParseISA(StringRef &MangledName, VFISAKind &ISA) {
  if (MangledName.empty())
    return ParseRet::Error;

  if (MangledName.startswith(VFABI::_LLVM_)) {
    MangledName = MangledName.drop_front(strlen(VFABI::_LLVM_));
    ISA = VFISAKind::LLVM;
  } else {
    ISA = StringSwitch<VFISAKind>(MangledName.take_front(1))
              .Case("n", VFISAKind::AdvancedSIMD)
              .Case("s", VFISAKind::SVE)
              .Case("b", VFISAKind::SSE)
              .Case("c", VFISAKind::AVX)
              .Case("d", VFISAKind::AVX2)
              .Case("e", VFISAKind::AVX512)
              .Default(VFISAKind::Unknown);
    MangledName = MangledName.drop_front(1);
  }

  return ParseRet::OK;
}

/// Extracts the `<mask>` information from the mangled string, and
/// sets `IsMasked` accordingly. The input string `MangledName` is
/// left unmodified.
ParseRet tryParseMask(StringRef &MangledName, bool &IsMasked) {
  if (MangledName.consume_front("M")) {
    IsMasked = true;
    return ParseRet::OK;
  }

  if (MangledName.consume_front("N")) {
    IsMasked = false;
    return ParseRet::OK;
  }

  return ParseRet::Error;
}

/// Extract the `<vlen>` information from the mangled string, and
/// sets `VF` accordingly. A `<vlen> == "x"` token is interpreted as a scalable
/// vector length. On success, the `<vlen>` token is removed from
/// the input string `ParseString`.
///
ParseRet tryParseVLEN(StringRef &ParseString, unsigned &VF, bool &IsScalable) {
  if (ParseString.consume_front("x")) {
    // Set VF to 0, to be later adjusted to a value grater than zero
    // by looking at the signature of the vector function with
    // `getECFromSignature`.
    VF = 0;
    IsScalable = true;
    return ParseRet::OK;
  }

  if (ParseString.consumeInteger(10, VF))
    return ParseRet::Error;

  // The token `0` is invalid for VLEN.
  if (VF == 0)
    return ParseRet::Error;

  IsScalable = false;
  return ParseRet::OK;
}

/// The function looks for the following strings at the beginning of
/// the input string `ParseString`:
///
///  <token> <number>
///
/// On success, it removes the parsed parameter from `ParseString`,
/// sets `PKind` to the correspondent enum value, sets `Pos` to
/// <number>, and return success.  On a syntax error, it return a
/// parsing error. If nothing is parsed, it returns None.
///
/// The function expects <token> to be one of "ls", "Rs", "Us" or
/// "Ls".
ParseRet tryParseLinearTokenWithRuntimeStep(StringRef &ParseString,
                                            VFParamKind &PKind, int &Pos,
                                            const StringRef Token) {
  if (ParseString.consume_front(Token)) {
    PKind = VFABI::getVFParamKindFromString(Token);
    if (ParseString.consumeInteger(10, Pos))
      return ParseRet::Error;
    return ParseRet::OK;
  }

  return ParseRet::None;
}

/// The function looks for the following stringt at the beginning of
/// the input string `ParseString`:
///
///  <token> <number>
///
/// <token> is one of "ls", "Rs", "Us" or "Ls".
///
/// On success, it removes the parsed parameter from `ParseString`,
/// sets `PKind` to the correspondent enum value, sets `StepOrPos` to
/// <number>, and return success.  On a syntax error, it return a
/// parsing error. If nothing is parsed, it returns None.
ParseRet tryParseLinearWithRuntimeStep(StringRef &ParseString,
                                       VFParamKind &PKind, int &StepOrPos) {
  ParseRet Ret;

  // "ls" <RuntimeStepPos>
  Ret = tryParseLinearTokenWithRuntimeStep(ParseString, PKind, StepOrPos, "ls");
  if (Ret != ParseRet::None)
    return Ret;

  // "Rs" <RuntimeStepPos>
  Ret = tryParseLinearTokenWithRuntimeStep(ParseString, PKind, StepOrPos, "Rs");
  if (Ret != ParseRet::None)
    return Ret;

  // "Ls" <RuntimeStepPos>
  Ret = tryParseLinearTokenWithRuntimeStep(ParseString, PKind, StepOrPos, "Ls");
  if (Ret != ParseRet::None)
    return Ret;

  // "Us" <RuntimeStepPos>
  Ret = tryParseLinearTokenWithRuntimeStep(ParseString, PKind, StepOrPos, "Us");
  if (Ret != ParseRet::None)
    return Ret;

  return ParseRet::None;
}

/// The function looks for the following strings at the beginning of
/// the input string `ParseString`:
///
///  <token> {"n"} <number>
///
/// On success, it removes the parsed parameter from `ParseString`,
/// sets `PKind` to the correspondent enum value, sets `LinearStep` to
/// <number>, and return success.  On a syntax error, it return a
/// parsing error. If nothing is parsed, it returns None.
///
/// The function expects <token> to be one of "l", "R", "U" or
/// "L".
ParseRet tryParseCompileTimeLinearToken(StringRef &ParseString,
                                        VFParamKind &PKind, int &LinearStep,
                                        const StringRef Token) {
  if (ParseString.consume_front(Token)) {
    PKind = VFABI::getVFParamKindFromString(Token);
    const bool Negate = ParseString.consume_front("n");
    if (ParseString.consumeInteger(10, LinearStep))
      LinearStep = 1;
    if (Negate)
      LinearStep *= -1;
    return ParseRet::OK;
  }

  return ParseRet::None;
}

/// The function looks for the following strings at the beginning of
/// the input string `ParseString`:
///
/// ["l" | "R" | "U" | "L"] {"n"} <number>
///
/// On success, it removes the parsed parameter from `ParseString`,
/// sets `PKind` to the correspondent enum value, sets `LinearStep` to
/// <number>, and return success.  On a syntax error, it return a
/// parsing error. If nothing is parsed, it returns None.
ParseRet tryParseLinearWithCompileTimeStep(StringRef &ParseString,
                                           VFParamKind &PKind, int &StepOrPos) {
  // "l" {"n"} <CompileTimeStep>
  if (tryParseCompileTimeLinearToken(ParseString, PKind, StepOrPos, "l") ==
      ParseRet::OK)
    return ParseRet::OK;

  // "R" {"n"} <CompileTimeStep>
  if (tryParseCompileTimeLinearToken(ParseString, PKind, StepOrPos, "R") ==
      ParseRet::OK)
    return ParseRet::OK;

  // "L" {"n"} <CompileTimeStep>
  if (tryParseCompileTimeLinearToken(ParseString, PKind, StepOrPos, "L") ==
      ParseRet::OK)
    return ParseRet::OK;

  // "U" {"n"} <CompileTimeStep>
  if (tryParseCompileTimeLinearToken(ParseString, PKind, StepOrPos, "U") ==
      ParseRet::OK)
    return ParseRet::OK;

  return ParseRet::None;
}

/// Looks into the <parameters> part of the mangled name in search
/// for valid paramaters at the beginning of the string
/// `ParseString`.
///
/// On success, it removes the parsed parameter from `ParseString`,
/// sets `PKind` to the correspondent enum value, sets `StepOrPos`
/// accordingly, and return success.  On a syntax error, it return a
/// parsing error. If nothing is parsed, it returns None.
ParseRet tryParseParameter(StringRef &ParseString, VFParamKind &PKind,
                           int &StepOrPos) {
  if (ParseString.consume_front("v")) {
    PKind = VFParamKind::Vector;
    StepOrPos = 0;
    return ParseRet::OK;
  }

  if (ParseString.consume_front("u")) {
    PKind = VFParamKind::OMP_Uniform;
    StepOrPos = 0;
    return ParseRet::OK;
  }

  const ParseRet HasLinearRuntime =
      tryParseLinearWithRuntimeStep(ParseString, PKind, StepOrPos);
  if (HasLinearRuntime != ParseRet::None)
    return HasLinearRuntime;

  const ParseRet HasLinearCompileTime =
      tryParseLinearWithCompileTimeStep(ParseString, PKind, StepOrPos);
  if (HasLinearCompileTime != ParseRet::None)
    return HasLinearCompileTime;

  return ParseRet::None;
}

/// Looks into the <parameters> part of the mangled name in search
/// of a valid 'aligned' clause. The function should be invoked
/// after parsing a parameter via `tryParseParameter`.
///
/// On success, it removes the parsed parameter from `ParseString`,
/// sets `PKind` to the correspondent enum value, sets `StepOrPos`
/// accordingly, and return success.  On a syntax error, it return a
/// parsing error. If nothing is parsed, it returns None.
ParseRet tryParseAlign(StringRef &ParseString, Align &Alignment) {
  uint64_t Val;
  //    "a" <number>
  if (ParseString.consume_front("a")) {
    if (ParseString.consumeInteger(10, Val))
      return ParseRet::Error;

    if (!isPowerOf2_64(Val))
      return ParseRet::Error;

    Alignment = Align(Val);

    return ParseRet::OK;
  }

  return ParseRet::None;
}
#ifndef NDEBUG
// Verify the assumtion that all vectors in the signature of a vector
// function have the same number of elements.
bool verifyAllVectorsHaveSameWidth(FunctionType *Signature) {
  SmallVector<VectorType *, 2> VecTys;
  if (auto *RetTy = dyn_cast<VectorType>(Signature->getReturnType()))
    VecTys.push_back(RetTy);
  for (auto *Ty : Signature->params())
    if (auto *VTy = dyn_cast<VectorType>(Ty))
      VecTys.push_back(VTy);

  if (VecTys.size() <= 1)
    return true;

  assert(VecTys.size() > 1 && "Invalid number of elements.");
  const ElementCount EC = VecTys[0]->getElementCount();
  return llvm::all_of(llvm::drop_begin(VecTys), [&EC](VectorType *VTy) {
    return (EC == VTy->getElementCount());
  });
}

#endif // NDEBUG

// Extract the VectorizationFactor from a given function signature,
// under the assumtion that all vectors have the same number of
// elements, i.e. same ElementCount.Min.
ElementCount getECFromSignature(FunctionType *Signature) {
  assert(verifyAllVectorsHaveSameWidth(Signature) &&
         "Invalid vector signature.");

  if (auto *RetTy = dyn_cast<VectorType>(Signature->getReturnType()))
    return RetTy->getElementCount();
  for (auto *Ty : Signature->params())
    if (auto *VTy = dyn_cast<VectorType>(Ty))
      return VTy->getElementCount();

  return ElementCount::getFixed(/*Min=*/1);
}
} // namespace

// Format of the ABI name:
// _ZGV<isa><mask><vlen><parameters>_<scalarname>[(<redirection>)]
Optional<VFInfo> VFABI::tryDemangleForVFABI(StringRef MangledName,
                                            const Module &M) {
  const StringRef OriginalName = MangledName;
  // Assume there is no custom name <redirection>, and therefore the
  // vector name consists of
  // _ZGV<isa><mask><vlen><parameters>_<scalarname>.
  StringRef VectorName = MangledName;

  // Parse the fixed size part of the manled name
  if (!MangledName.consume_front("_ZGV"))
    return None;

  // Extract ISA. An unknow ISA is also supported, so we accept all
  // values.
  VFISAKind ISA;
  if (tryParseISA(MangledName, ISA) != ParseRet::OK)
    return None;

  // Extract <mask>.
  bool IsMasked;
  if (tryParseMask(MangledName, IsMasked) != ParseRet::OK)
    return None;

  // Parse the variable size, starting from <vlen>.
  unsigned VF;
  bool IsScalable;
  if (tryParseVLEN(MangledName, VF, IsScalable) != ParseRet::OK)
    return None;

  // Parse the <parameters>.
  ParseRet ParamFound;
  SmallVector<VFParameter, 8> Parameters;
  do {
    const unsigned ParameterPos = Parameters.size();
    VFParamKind PKind;
    int StepOrPos;
    ParamFound = tryParseParameter(MangledName, PKind, StepOrPos);

    // Bail off if there is a parsing error in the parsing of the parameter.
    if (ParamFound == ParseRet::Error)
      return None;

    if (ParamFound == ParseRet::OK) {
      Align Alignment;
      // Look for the alignment token "a <number>".
      const ParseRet AlignFound = tryParseAlign(MangledName, Alignment);
      // Bail off if there is a syntax error in the align token.
      if (AlignFound == ParseRet::Error)
        return None;

      // Add the parameter.
      Parameters.push_back({ParameterPos, PKind, StepOrPos, Alignment});
    }
  } while (ParamFound == ParseRet::OK);

  // A valid MangledName must have at least one valid entry in the
  // <parameters>.
  if (Parameters.empty())
    return None;

  // Check for the <scalarname> and the optional <redirection>, which
  // are separated from the prefix with "_"
  if (!MangledName.consume_front("_"))
    return None;

  // The rest of the string must be in the format:
  // <scalarname>[(<redirection>)]
  const StringRef ScalarName =
      MangledName.take_while([](char In) { return In != '('; });

  if (ScalarName.empty())
    return None;

  // Reduce MangledName to [(<redirection>)].
  MangledName = MangledName.ltrim(ScalarName);
  // Find the optional custom name redirection.
  if (MangledName.consume_front("(")) {
    if (!MangledName.consume_back(")"))
      return None;
    // Update the vector variant with the one specified by the user.
    VectorName = MangledName;
    // If the vector name is missing, bail out.
    if (VectorName.empty())
      return None;
  }

  // LLVM internal mapping via the TargetLibraryInfo (TLI) must be
  // redirected to an existing name.
  if (ISA == VFISAKind::LLVM && VectorName == OriginalName)
    return None;

  // When <mask> is "M", we need to add a parameter that is used as
  // global predicate for the function.
  if (IsMasked) {
    const unsigned Pos = Parameters.size();
    Parameters.push_back({Pos, VFParamKind::GlobalPredicate});
  }

  // Asserts for parameters of type `VFParamKind::GlobalPredicate`, as
  // prescribed by the Vector Function ABI specifications supported by
  // this parser:
  // 1. Uniqueness.
  // 2. Must be the last in the parameter list.
  const auto NGlobalPreds = std::count_if(
      Parameters.begin(), Parameters.end(), [](const VFParameter PK) {
        return PK.ParamKind == VFParamKind::GlobalPredicate;
      });
  assert(NGlobalPreds < 2 && "Cannot have more than one global predicate.");
  if (NGlobalPreds)
    assert(Parameters.back().ParamKind == VFParamKind::GlobalPredicate &&
           "The global predicate must be the last parameter");

  // Adjust the VF for scalable signatures. The EC.Min is not encoded
  // in the name of the function, but it is encoded in the IR
  // signature of the function. We need to extract this information
  // because it is needed by the loop vectorizer, which reasons in
  // terms of VectorizationFactor or ElementCount. In particular, we
  // need to make sure that the VF field of the VFShape class is never
  // set to 0.
  if (IsScalable) {
    const Function *F = M.getFunction(VectorName);
    // The declaration of the function must be present in the module
    // to be able to retrieve its signature.
    if (!F)
      return None;
    const ElementCount EC = getECFromSignature(F->getFunctionType());
    VF = EC.getKnownMinValue();
  }

  // 1. We don't accept a zero lanes vectorization factor.
  // 2. We don't accept the demangling if the vector function is not
  // present in the module.
  if (VF == 0)
    return None;
  if (!M.getFunction(VectorName))
    return None;

  const VFShape Shape({ElementCount::get(VF, IsScalable), Parameters});
  return VFInfo({Shape, std::string(ScalarName), std::string(VectorName), ISA});
}

VFParamKind VFABI::getVFParamKindFromString(const StringRef Token) {
  const VFParamKind ParamKind = StringSwitch<VFParamKind>(Token)
                                    .Case("v", VFParamKind::Vector)
                                    .Case("l", VFParamKind::OMP_Linear)
                                    .Case("R", VFParamKind::OMP_LinearRef)
                                    .Case("L", VFParamKind::OMP_LinearVal)
                                    .Case("U", VFParamKind::OMP_LinearUVal)
                                    .Case("ls", VFParamKind::OMP_LinearPos)
                                    .Case("Ls", VFParamKind::OMP_LinearValPos)
                                    .Case("Rs", VFParamKind::OMP_LinearRefPos)
                                    .Case("Us", VFParamKind::OMP_LinearUValPos)
                                    .Case("u", VFParamKind::OMP_Uniform)
                                    .Default(VFParamKind::Unknown);

  if (ParamKind != VFParamKind::Unknown)
    return ParamKind;

  // This function should never be invoked with an invalid input.
  llvm_unreachable("This fuction should be invoken only on parameters"
                   " that have a textual representation in the mangled name"
                   " of the Vector Function ABI");
}
