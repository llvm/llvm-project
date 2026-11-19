//===- Types.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Types.h"
#include "Common/CodeGenDAGPatterns.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace gi {

//===- Global Data --------------------------------------------------------===//

std::set<LLTCodeGen> KnownTypes;
std::map<LLTByHwMode, unsigned> LLTByHwModeIDs;

unsigned getLLTByHwModeID(const LLTByHwMode &Ty) {
  auto [It, Inserted] = LLTByHwModeIDs.try_emplace(Ty, LLTByHwModeIDs.size());
  if (Inserted && It->second >= 256)
    report_fatal_error("More LLT by HwMode than fit in a 8-bit index");
  return It->second;
}

//===- LLTCodeGen ---------------------------------------------------------===//

std::string LLTCodeGen::getCxxEnumValue() const {
  std::string Str;
  raw_string_ostream OS(Str);

  emitCxxEnumValue(OS);
  return Str;
}

void LLTCodeGen::emitCxxEnumValue(raw_ostream &OS) const {
  if (Ty.isScalar()) {
    if (Ty.isBFloat16())
      OS << "GILLT_bf16";
    else if (Ty.isPPCF128())
      OS << "GILLT_ppcf128";
    else if (Ty.isX86FP80())
      OS << "GILLT_x86fp80";
    else if (Ty.isFloat())
      OS << "GILLT_f" << Ty.getSizeInBits();
    else if (Ty.isInteger())
      OS << "GILLT_i" << Ty.getSizeInBits();
    else
      OS << "GILLT_s" << Ty.getSizeInBits();
    return;
  }
  if (Ty.isVector()) {
    OS << (Ty.isScalable() ? "GILLT_nxv" : "GILLT_v")
       << Ty.getElementCount().getKnownMinValue();

    LLT ElemTy = Ty.getElementType();
    if (ElemTy.isBFloat16())
      OS << "bf16";
    else if (ElemTy.isPPCF128())
      OS << "ppcf128";
    else if (ElemTy.isX86FP80())
      OS << "x86fp80";
    else if (ElemTy.isFloat())
      OS << "f" << ElemTy.getSizeInBits();
    else if (ElemTy.isInteger())
      OS << "i" << ElemTy.getSizeInBits();
    else
      OS << "s" << ElemTy.getSizeInBits();
    return;
  }

  if (Ty.isPointer()) {
    OS << "GILLT_p" << Ty.getAddressSpace();
    if (Ty.getSizeInBits() > 0)
      OS << "s" << Ty.getSizeInBits();
    return;
  }

  llvm_unreachable("Unhandled LLT");
}

void LLTCodeGen::emitCxxConstructorCall(raw_ostream &OS) const {
  auto EmitScalarType = [&OS](LLT T) {
    if (T.isInteger())
      OS << "LLT(LLT::Kind::INTEGER, ElementCount::getFixed(0), "
         << T.getScalarSizeInBits() << ")";
    else if (T.isBFloat16())
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 16, "
            "LLT::FpSemantics::S_BFloat)";
    else if (T.isPPCF128())
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 128, "
            "LLT::FpSemantics::S_PPCDoubleDouble)";
    else if (T.isX86FP80())
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 80, "
            "LLT::FpSemantics::S_x87DoubleExtended)";
    else if (T.isFloat(16))
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 16, "
            "LLT::FpSemantics::S_IEEEhalf)";
    else if (T.isFloat(32))
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 32, "
            "LLT::FpSemantics::S_IEEEsingle)";
    else if (T.isFloat(64))
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 64, "
            "LLT::FpSemantics::S_IEEEdouble)";
    else if (T.isFloat(128))
      OS << "LLT(LLT::Kind::FLOAT, ElementCount::getFixed(0), 128, "
            "LLT::FpSemantics::S_IEEEquad)";
    else
      OS << "LLT::scalar(" << T.getScalarSizeInBits() << ")";
  };

  if (Ty.isScalar()) {
    EmitScalarType(Ty);
    return;
  }

  if (Ty.isVector()) {
    OS << "LLT::vector("
       << (Ty.isScalable() ? "ElementCount::getScalable("
                           : "ElementCount::getFixed(")
       << Ty.getElementCount().getKnownMinValue() << "), ";
    EmitScalarType(Ty.getElementType());
    OS << ")";
    return;
  }

  if (Ty.isPointer() && Ty.getSizeInBits() > 0) {
    OS << "LLT::pointer(" << Ty.getAddressSpace() << ", " << Ty.getSizeInBits()
       << ")";
    return;
  }

  llvm_unreachable("Unhandled LLT");
}

/// This ordering is used for std::unique() and llvm::sort(). There's no
/// particular logic behind the order but either A < B or B < A must be
/// true if A != B.
bool LLTCodeGen::operator<(const LLTCodeGen &Other) const {
  return Ty.getUniqueRAWLLTData() < Other.Ty.getUniqueRAWLLTData();
}

//===- LLTCodeGen Helpers -------------------------------------------------===//

std::optional<LLTCodeGen> MVTToLLT(MVT VT) {
  if (VT.isVector() && !VT.getVectorElementCount().isScalar())
    return LLTCodeGen(LLT(VT));

  if (VT.isInteger() || VT.isFloatingPoint())
    return LLTCodeGen(LLT(VT));

  return std::nullopt;
}

std::optional<LLTCodeGen> MVTToGenericLLT(MVT VT) {
  if (VT.isVector() && !VT.getVectorElementCount().isScalar()) {
    unsigned ElemBits = VT.getVectorElementType().getSizeInBits();
    return LLTCodeGen(
        LLT::vector(VT.getVectorElementCount(), LLT::scalar(ElemBits)));
  }

  if (VT.isInteger() || VT.isFloatingPoint())
    return LLTCodeGen(LLT::scalar(VT.getSizeInBits()));

  return std::nullopt;
}

std::optional<LLTOrLLTByHwMode> getLLTOrLLTByHwMode(const TypeSetByHwMode &VTy,
                                                    bool Generic) {
  if (!VTy.isValueTypeByHwMode(/*AllowEmpty=*/true))
    return std::nullopt;

  LLTByHwMode Result;
  for (const auto &[Mode, Types] : VTy) {
    // A HwMode with no types means the pattern can't match in that mode. Use
    // an invalid LLT so the type check always fails.
    if (Types.empty()) {
      Result.insertTypeForMode(Mode, LLTCodeGen());
      continue;
    }

    MVT VT = *Types.begin();
    std::optional<LLTCodeGen> Ty;
    if (VTy.isPointer())
      Ty = LLTCodeGen(LLT::pointer(VTy.getPtrAddrSpace(), VT.getSizeInBits()));
    else
      Ty = Generic ? MVTToGenericLLT(VT) : MVTToLLT(VT);
    if (!Ty)
      return std::nullopt;
    Result.insertTypeForMode(Mode, *Ty);
  }

  // If every HwMode has the same type and there is a default, there's no need
  // to look up the type by HwMode.
  const LLTCodeGen &First = Result.begin()->second;
  if (Result.hasDefault() && First.get().isValid() &&
      all_of(Result, [&](const auto &P) { return P.second == First; }))
    return First;

  return Result;
}
} // namespace gi
} // namespace llvm
