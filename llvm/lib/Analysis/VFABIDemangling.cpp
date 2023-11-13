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
ParseRet tryParseVLEN(StringRef &ParseString,
                      std::optional<unsigned> &ParsedVF) {
  if (ParseString.consume_front("x")) {
    // We can't determine the VF of a scalable vector by looking at the vlen
    // string (just 'x'), so say we successfully parsed it but return a nullopt
    // so that the caller knows it must look at the arguments to determine
    // the minimum VF based on types.
    ParsedVF = std::nullopt;
    return ParseRet::OK;
  }

  unsigned VF = 0;
  if (ParseString.consumeInteger(10, VF))
    return ParseRet::Error;

  // The token `0` is invalid for VLEN.
  if (VF == 0)
    return ParseRet::Error;

  ParsedVF = VF;
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
/// parsing error. If nothing is parsed, it returns std::nullopt.
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
/// parsing error. If nothing is parsed, it returns std::nullopt.
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
/// parsing error. If nothing is parsed, it returns std::nullopt.
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
/// parsing error. If nothing is parsed, it returns std::nullopt.
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
/// parsing error. If nothing is parsed, it returns std::nullopt.
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
/// parsing error. If nothing is parsed, it returns std::nullopt.
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

// Given a type, return the size in bits if it is a supported element type
// for vectorized function calls, or nullopt if not.
std::optional<unsigned> getSizeFromScalarType(Type *Ty) {
  // The scalar function should only take scalar arguments.
  if (!Ty->isIntegerTy() && !Ty->isFloatingPointTy() && !Ty->isPointerTy())
    return std::nullopt;

  unsigned SizeInBits = Ty->getPrimitiveSizeInBits();
  switch (SizeInBits) {
  // Legal power-of-two scalars are supported.
  case 64:
  case 32:
  case 16:
  case 8:
    return SizeInBits;
  case 0:
    // We're assuming a 64b pointer size here for SVE; if another non-64b
    // target adds support for scalable vectors, we may need DataLayout to
    // determine the size.
    if (Ty->isPointerTy())
      return 64;
    break;
  default:
    break;
  }

  return std::nullopt;
}

// Extract the VectorizationFactor from a given function signature, based
// on the widest scalar element types that will become vector parameters.
std::optional<ElementCount>
getScalableECFromSignature(FunctionType *Signature, const VFISAKind ISA,
                           const SmallVectorImpl<VFParameter> &Params) {
  // Look up the minimum known register size in order to calculate minimum VF.
  // Only AArch64 SVE is supported at present.
  unsigned MinRegSizeInBits;
  switch (ISA) {
  case VFISAKind::SVE:
    MinRegSizeInBits = 128;
    break;
  default:
    return std::nullopt;
  }

  unsigned WidestTypeInBits = 0;
  for (auto &Param : Params) {
    // Check any parameters that will be widened to vectors. Uniform or linear
    // parameters may be misleading for determining the VF of a given function.
    if (Param.ParamKind == VFParamKind::Vector) {
      // If the scalar function doesn't actually have a corresponding argument,
      // reject the mapping.
      if (Param.ParamPos + 1 > Signature->getNumParams())
        return std::nullopt;
      Type *PTy = Signature->getParamType(Param.ParamPos);

      std::optional<unsigned> SizeInBits = getSizeFromScalarType(PTy);
      if (SizeInBits)
        WidestTypeInBits = std::max(WidestTypeInBits, *SizeInBits);
    }
  }

  // Also check the return type.
  std::optional<unsigned> ReturnSizeInBits =
      getSizeFromScalarType(Signature->getReturnType());
  if (ReturnSizeInBits)
    WidestTypeInBits = std::max(WidestTypeInBits, *ReturnSizeInBits);

  // SVE bases the VF on the widest element types present, and vector arguments
  // containing types of that width are always considered to be packed.
  // Arguments with narrower elements are considered to be unpacked.
  if (WidestTypeInBits)
    return ElementCount::getScalable(MinRegSizeInBits / WidestTypeInBits);

  return std::nullopt;
}
} // namespace

// Format of the ABI name:
// _ZGV<isa><mask><vlen><parameters>_<scalarname>[(<redirection>)]
std::optional<VFInfo> VFABI::tryDemangleForVFABI(StringRef MangledName,
                                                 const CallInst &CI) {
  const StringRef OriginalName = MangledName;
  // Assume there is no custom name <redirection>, and therefore the
  // vector name consists of
  // _ZGV<isa><mask><vlen><parameters>_<scalarname>.
  StringRef VectorName = MangledName;

  // Parse the fixed size part of the manled name
  if (!MangledName.consume_front("_ZGV"))
    return std::nullopt;

  // Extract ISA. An unknow ISA is also supported, so we accept all
  // values.
  VFISAKind ISA;
  if (tryParseISA(MangledName, ISA) != ParseRet::OK)
    return std::nullopt;

  // Extract <mask>.
  bool IsMasked;
  if (tryParseMask(MangledName, IsMasked) != ParseRet::OK)
    return std::nullopt;

  // Parse the variable size, starting from <vlen>.
  std::optional<unsigned> ParsedVF;
  if (tryParseVLEN(MangledName, ParsedVF) != ParseRet::OK)
    return std::nullopt;

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
      return std::nullopt;

    if (ParamFound == ParseRet::OK) {
      Align Alignment;
      // Look for the alignment token "a <number>".
      const ParseRet AlignFound = tryParseAlign(MangledName, Alignment);
      // Bail off if there is a syntax error in the align token.
      if (AlignFound == ParseRet::Error)
        return std::nullopt;

      // Add the parameter.
      Parameters.push_back({ParameterPos, PKind, StepOrPos, Alignment});
    }
  } while (ParamFound == ParseRet::OK);

  // A valid MangledName must have at least one valid entry in the
  // <parameters>.
  if (Parameters.empty())
    return std::nullopt;

  // Figure out the number of lanes in vectors for this function variant. This
  // is easy for fixed length, as the vlen encoding just gives us the value
  // directly. However, if the vlen mangling indicated that this function
  // variant expects scalable vectors, then we need to figure out the minimum
  // based on the widest scalar types in vector arguments.
  std::optional<ElementCount> EC;
  if (ParsedVF) {
    // Fixed length VF
    EC = ElementCount::getFixed(*ParsedVF);
  } else {
    // Scalable VF, need to work out the minimum from the element types
    // in the scalar function arguments.
    EC = getScalableECFromSignature(CI.getFunctionType(), ISA, Parameters);

    if (!EC)
      return std::nullopt;
  }

  // Check for the <scalarname> and the optional <redirection>, which
  // are separated from the prefix with "_"
  if (!MangledName.consume_front("_"))
    return std::nullopt;

  // The rest of the string must be in the format:
  // <scalarname>[(<redirection>)]
  const StringRef ScalarName =
      MangledName.take_while([](char In) { return In != '('; });

  if (ScalarName.empty())
    return std::nullopt;

  // Reduce MangledName to [(<redirection>)].
  MangledName = MangledName.ltrim(ScalarName);
  // Find the optional custom name redirection.
  if (MangledName.consume_front("(")) {
    if (!MangledName.consume_back(")"))
      return std::nullopt;
    // Update the vector variant with the one specified by the user.
    VectorName = MangledName;
    // If the vector name is missing, bail out.
    if (VectorName.empty())
      return std::nullopt;
  }

  // LLVM internal mapping via the TargetLibraryInfo (TLI) must be
  // redirected to an existing name.
  if (ISA == VFISAKind::LLVM && VectorName == OriginalName)
    return std::nullopt;

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
  const auto NGlobalPreds =
      llvm::count_if(Parameters, [](const VFParameter &PK) {
        return PK.ParamKind == VFParamKind::GlobalPredicate;
      });
  assert(NGlobalPreds < 2 && "Cannot have more than one global predicate.");
  if (NGlobalPreds)
    assert(Parameters.back().ParamKind == VFParamKind::GlobalPredicate &&
           "The global predicate must be the last parameter");

  const VFShape Shape({*EC, Parameters});
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
