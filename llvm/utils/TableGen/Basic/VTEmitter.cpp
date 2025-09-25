//===- VTEmitter.cpp - Generate properties from ValueTypes.td -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
#include <map>
using namespace llvm;

namespace {

class VTEmitter {
private:
  const RecordKeeper &Records;

public:
  VTEmitter(const RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);
};

} // End anonymous namespace.

static void vTtoGetLlvmTyString(raw_ostream &OS, const Record *VT) {
  bool IsVector = VT->getValueAsBit("isVector");
  bool IsRISCVVecTuple = VT->getValueAsBit("isRISCVVecTuple");

  if (IsRISCVVecTuple) {
    unsigned NElem = VT->getValueAsInt("nElem");
    unsigned Sz = VT->getValueAsInt("Size");
    OS << "TargetExtType::get(Context, \"riscv.vector.tuple\", "
          "ScalableVectorType::get(Type::getInt8Ty(Context), "
       << (Sz / (NElem * 8)) << "), " << NElem << ")";
    return;
  }

  if (IsVector)
    OS << (VT->getValueAsBit("isScalable") ? "Scalable" : "Fixed")
       << "VectorType::get(";

  auto OutputVT = IsVector ? VT->getValueAsDef("ElementType") : VT;
  int64_t OutputVTSize = OutputVT->getValueAsInt("Size");

  if (OutputVT->getValueAsBit("isFP")) {
    StringRef FloatTy;
    auto OutputVTName = OutputVT->getValueAsString("LLVMName");
    switch (OutputVTSize) {
    default:
      llvm_unreachable("Unhandled case");
    case 16:
      FloatTy = (OutputVTName == "bf16") ? "BFloatTy" : "HalfTy";
      break;
    case 32:
      FloatTy = "FloatTy";
      break;
    case 64:
      FloatTy = "DoubleTy";
      break;
    case 80:
      FloatTy = "X86_FP80Ty";
      break;
    case 128:
      FloatTy = (OutputVTName == "ppcf128") ? "PPC_FP128Ty" : "FP128Ty";
      break;
    }
    OS << "Type::get" << FloatTy << "(Context)";
  } else if (OutputVT->getValueAsBit("isInteger")) {
    // We only have Type::getInt1Ty, Int8, Int16, Int32, Int64, and Int128
    if ((isPowerOf2_64(OutputVTSize) && OutputVTSize >= 8 &&
         OutputVTSize <= 128) ||
        OutputVTSize == 1)
      OS << "Type::getInt" << OutputVTSize << "Ty(Context)";
    else
      OS << "Type::getIntNTy(Context, " << OutputVTSize << ")";
  } else {
    llvm_unreachable("Unhandled case");
  }

  if (IsVector)
    OS << ", " << VT->getValueAsInt("nElem") << ")";
}

void VTEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("ValueTypes Source Fragment", OS, Records);

  std::vector<const Record *> VTsByNumber{512};
  for (auto *VT : Records.getAllDerivedDefinitions("ValueType")) {
    auto Number = VT->getValueAsInt("Value");
    assert(0 <= Number && Number < (int)VTsByNumber.size() &&
           "ValueType should be uint16_t");
    assert(!VTsByNumber[Number] && "Duplicate ValueType");
    VTsByNumber[Number] = VT;
  }

  struct VTRange {
    StringRef First;
    StringRef Last;
    bool Closed;
  };

  std::map<StringRef, VTRange> VTRanges;

  auto UpdateVTRange = [&VTRanges](const char *Key, StringRef Name,
                                   bool Valid) {
    if (Valid) {
      auto [It, Inserted] = VTRanges.try_emplace(Key);
      if (Inserted)
        It->second.First = Name;
      assert(!It->second.Closed && "Gap detected!");
      It->second.Last = Name;
    } else if (auto It = VTRanges.find(Key); It != VTRanges.end()) {
      It->second.Closed = true;
    }
  };

  OS << "#ifdef GET_VT_ATTR // (Ty, n, sz, Any, Int, FP, Vec, Sc, Tup, NF, "
        "NElem, EltTy)\n";
  for (const auto *VT : VTsByNumber) {
    if (!VT)
      continue;
    auto Name = VT->getValueAsString("LLVMName");
    auto Value = VT->getValueAsInt("Value");
    bool IsInteger = VT->getValueAsBit("isInteger");
    bool IsFP = VT->getValueAsBit("isFP");
    bool IsVector = VT->getValueAsBit("isVector");
    bool IsScalable = VT->getValueAsBit("isScalable");
    bool IsRISCVVecTuple = VT->getValueAsBit("isRISCVVecTuple");
    bool IsCheriCapability = VT->getValueAsBit("isCheriCapability");
    int64_t NF = VT->getValueAsInt("NF");
    bool IsNormalValueType =  VT->getValueAsBit("isNormalValueType");
    int64_t NElem = IsVector ? VT->getValueAsInt("nElem") : 0;
    StringRef EltName = IsVector ? VT->getValueAsDef("ElementType")->getName()
                                 : "INVALID_SIMPLE_VALUE_TYPE";

    UpdateVTRange("INTEGER_FIXEDLEN_VECTOR_VALUETYPE", Name,
                  IsInteger && IsVector && !IsScalable);
    UpdateVTRange("INTEGER_SCALABLE_VECTOR_VALUETYPE", Name,
                  IsInteger && IsScalable);
    UpdateVTRange("FP_FIXEDLEN_VECTOR_VALUETYPE", Name,
                  IsFP && IsVector && !IsScalable);
    UpdateVTRange("FP_SCALABLE_VECTOR_VALUETYPE", Name, IsFP && IsScalable);
    UpdateVTRange("FIXEDLEN_VECTOR_VALUETYPE", Name, IsVector && !IsScalable);
    UpdateVTRange("SCALABLE_VECTOR_VALUETYPE", Name, IsScalable);
    UpdateVTRange("RISCV_VECTOR_TUPLE_VALUETYPE", Name, IsRISCVVecTuple);
    UpdateVTRange("VECTOR_VALUETYPE", Name, IsVector);
    UpdateVTRange("INTEGER_VALUETYPE", Name, IsInteger && !IsVector);
    UpdateVTRange("FP_VALUETYPE", Name, IsFP && !IsVector);
    UpdateVTRange("VALUETYPE", Name, IsNormalValueType);
    UpdateVTRange("CHERI_CAPABILITY_VALUETYPE", Name, IsCheriCapability);

    // clang-format off
    OS << "  GET_VT_ATTR("
       << Name << ", "
       << Value << ", "
       << VT->getValueAsInt("Size") << ", "
       << VT->getValueAsBit("isOverloaded") << ", "
       << (IsInteger ? Name[0] == 'i' ? 3 : 1 : 0) << ", "
       << (IsFP ? Name[0] == 'f' ? 3 : 1 : 0) << ", "
       << IsVector << ", "
       << IsScalable << ", "
       << IsRISCVVecTuple << ", "
       << NF << ", "
       << NElem << ", "
       << EltName << ")\n";
    // clang-format on
  }
  OS << "#endif\n\n";

  OS << "#ifdef GET_VT_RANGES\n";
  for (const auto &KV : VTRanges) {
    assert(KV.second.Closed);
    OS << "  FIRST_" << KV.first << " = " << KV.second.First << ",\n"
       << "  LAST_" << KV.first << " = " << KV.second.Last << ",\n";
  }
  OS << "#endif\n\n";

  OS << "#ifdef GET_VT_VECATTR // (Ty, Sc, Tup, nElem, ElTy)\n";
  for (const auto *VT : VTsByNumber) {
    if (!VT || !VT->getValueAsBit("isVector"))
      continue;
    const auto *ElTy = VT->getValueAsDef("ElementType");
    assert(ElTy);
    // clang-format off
    OS << "  GET_VT_VECATTR("
       << VT->getValueAsString("LLVMName") << ", "
       << VT->getValueAsBit("isScalable") << ", "
       << VT->getValueAsBit("isRISCVVecTuple") << ", "
       << VT->getValueAsInt("nElem") << ", "
       << ElTy->getName() << ")\n";
    // clang-format on
  }
  OS << "#endif\n\n";

  OS << "#ifdef GET_VT_EVT\n";
  for (const auto *VT : VTsByNumber) {
    if (!VT)
      continue;
    bool IsInteger = VT->getValueAsBit("isInteger");
    bool IsVector = VT->getValueAsBit("isVector");
    bool IsFP = VT->getValueAsBit("isFP");
    bool IsRISCVVecTuple = VT->getValueAsBit("isRISCVVecTuple");

    if (!IsInteger && !IsVector && !IsFP && !IsRISCVVecTuple)
      continue;

    OS << "  GET_VT_EVT(" << VT->getValueAsString("LLVMName") << ", ";
    vTtoGetLlvmTyString(OS, VT);
    OS << ")\n";
  }
  OS << "#endif\n\n";
}

static TableGen::Emitter::OptClass<VTEmitter> X("gen-vt", "Generate ValueType");
