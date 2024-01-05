//===- VFABIDemangling.cpp - Vector Function ABI demangling utilities. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "vfabi-demangling"

namespace {
/// Utilities for the Vector Function ABI name parser.

/// Return types for the parser functions.
enum class ParseRet {
  OK,   // Found.
  None, // Not found.
  Error // Syntax error.
};

/// Extracts the `<isa>` information from the mangled string, and
/// sets the `ISA` accordingly. If successful, the <isa> token is removed
/// from the input string `MangledName`.
static ParseRet tryParseISA(StringRef &MangledName, VFISAKind &ISA) {
  if (MangledName.empty())
    return ParseRet::Error;

  if (MangledName.starts_with(VFABI::_LLVM_)) {
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
/// sets `IsMasked` accordingly. If successful, the <mask> token is removed
/// from the input string `MangledName`.
static ParseRet tryParseMask(StringRef &MangledName, bool &IsMasked) {
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
/// sets `ParsedVF` accordingly. A `<vlen> == "x"` token is interpreted as a
/// scalable vector length and the boolean is set to true, otherwise a nonzero
/// unsigned integer will be directly used as a VF. On success, the `<vlen>`
/// token is removed from the input string `ParseString`.
static ParseRet tryParseVLEN(StringRef &ParseString, VFISAKind ISA,
                             std::pair<unsigned, bool> &ParsedVF) {
  if (ParseString.consume_front("x")) {
    // SVE is the only scalable ISA currently supported.
    if (ISA != VFISAKind::SVE) {
      LLVM_DEBUG(dbgs() << "Vector function variant declared with scalable VF "
                        << "but ISA is not SVE\n");
      return ParseRet::Error;
    }
    // We can't determine the VF of a scalable vector by looking at the vlen
    // string (just 'x'), so say we successfully parsed it but return a 'true'
    // for the scalable field with an invalid VF field so that we know to look
    // up the actual VF based on element types from the parameters or return.
    ParsedVF = {0, true};
    return ParseRet::OK;
  }

  unsigned VF = 0;
  if (ParseString.consumeInteger(10, VF))
    return ParseRet::Error;

  // The token `0` is invalid for VLEN.
  if (VF == 0)
    return ParseRet::Error;

  ParsedVF = {VF, false};
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
static ParseRet tryParseLinearTokenWithRuntimeStep(StringRef &ParseString,
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

/// The function looks for the following string at the beginning of
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
static ParseRet tryParseLinearWithRuntimeStep(StringRef &ParseString,
                                              VFParamKind &PKind,
                                              int &StepOrPos) {
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
static ParseRet tryParseCompileTimeLinearToken(StringRef &ParseString,
                                               VFParamKind &PKind,
                                               int &LinearStep,
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
static ParseRet tryParseLinearWithCompileTimeStep(StringRef &ParseString,
                                                  VFParamKind &PKind,
                                                  int &StepOrPos) {
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
static ParseRet tryParseParameter(StringRef &ParseString, VFParamKind &PKind,
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
static ParseRet tryParseAlign(StringRef &ParseString, Align &Alignment) {
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

// Returns the 'natural' VF for a given scalar element type, based on the
// current architecture.
//
// For SVE (currently the only scalable architecture with a defined name
// mangling), we assume a minimum vector size of 128b and return a VF based on
// the number of elements of the given type which would fit in such a vector.
static std::optional<ElementCount> getElementCountForTy(const VFISAKind ISA,
                                                        const Type *Ty) {
  // Only AArch64 SVE is supported at present.
  assert(ISA == VFISAKind::SVE &&
         "Scalable VF decoding only implemented for SVE\n");

  if (Ty->isIntegerTy(64) || Ty->isDoubleTy() || Ty->isPointerTy())
    return ElementCount::getScalable(2);
  if (Ty->isIntegerTy(32) || Ty->isFloatTy())
    return ElementCount::getScalable(4);
  if (Ty->isIntegerTy(16) || Ty->is16bitFPTy())
    return ElementCount::getScalable(8);
  if (Ty->isIntegerTy(8))
    return ElementCount::getScalable(16);

  return std::nullopt;
}

// Extract the VectorizationFactor from a given function signature, based
// on the widest scalar element types that will become vector parameters.
static std::optional<ElementCount>
getScalableECFromSignature(const FunctionType *Signature, const VFISAKind ISA,
                           const SmallVectorImpl<VFParameter> &Params) {
  // Start with a very wide EC and drop when we find smaller ECs based on type.
  ElementCount MinEC =
      ElementCount::getScalable(std::numeric_limits<unsigned int>::max());
  for (auto &Param : Params) {
    // Only vector parameters are used when determining the VF; uniform or
    // linear are left as scalars, so do not affect VF.
    if (Param.ParamKind == VFParamKind::Vector) {
      // If the scalar function doesn't actually have a corresponding argument,
      // reject the mapping.
      if (Param.ParamPos >= Signature->getNumParams())
        return std::nullopt;
      Type *PTy = Signature->getParamType(Param.ParamPos);

      std::optional<ElementCount> EC = getElementCountForTy(ISA, PTy);
      // If we have an unknown scalar element type we can't find a reasonable
      // VF.
      if (!EC)
        return std::nullopt;

      // Find the smallest VF, based on the widest scalar type.
      if (ElementCount::isKnownLT(*EC, MinEC))
        MinEC = *EC;
    }
  }

  // Also check the return type if not void.
  Type *RetTy = Signature->getReturnType();
  if (!RetTy->isVoidTy()) {
    std::optional<ElementCount> ReturnEC = getElementCountForTy(ISA, RetTy);
    // If we have an unknown scalar element type we can't find a reasonable VF.
    if (!ReturnEC)
      return std::nullopt;
    if (ElementCount::isKnownLT(*ReturnEC, MinEC))
      MinEC = *ReturnEC;
  }

  // The SVE Vector function call ABI bases the VF on the widest element types
  // present, and vector arguments containing types of that width are always
  // considered to be packed. Arguments with narrower elements are considered
  // to be unpacked.
  if (MinEC.getKnownMinValue() < std::numeric_limits<unsigned int>::max())
    return MinEC;

  return std::nullopt;
}
} // namespace

// Format of the ABI name:
// _ZGV<isa><mask><vlen><parameters>_<scalarname>[(<redirection>)]
std::optional<VFInfo> VFABI::tryDemangleForVFABI(StringRef MangledName,
                                                 const FunctionType *FTy) {
  const StringRef OriginalName = MangledName;
  // Assume there is no custom name <redirection>, and therefore the
  // vector name consists of
  // _ZGV<isa><mask><vlen><parameters>_<scalarname>.
  StringRef VectorName = MangledName;

  // Parse the fixed size part of the mangled name
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
  std::pair<unsigned, bool> ParsedVF;
  if (tryParseVLEN(MangledName, ISA, ParsedVF) != ParseRet::OK)
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
  // variant expects scalable vectors we need to work it out based on the
  // demangled parameter types and the scalar function signature.
  std::optional<ElementCount> EC;
  if (ParsedVF.second) {
    EC = getScalableECFromSignature(FTy, ISA, Parameters);
    if (!EC)
      return std::nullopt;
  } else
    EC = ElementCount::getFixed(ParsedVF.first);

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
