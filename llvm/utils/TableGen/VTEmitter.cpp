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
#include <array>
#include <cassert>
#include <map>
using namespace llvm;

namespace {

class VTEmitter {
private:
  RecordKeeper &Records;

public:
  VTEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);
};

} // End anonymous namespace.

void VTEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("ValueTypes Source Fragment", OS);

  std::array<const Record *, 256> VTsByNumber = {};
  auto ValueTypes = Records.getAllDerivedDefinitions("ValueType");
  for (auto *VT : ValueTypes) {
    auto Number = VT->getValueAsInt("Value");
    assert(0 <= Number && Number < (int)VTsByNumber.size() &&
           "ValueType should be uint8_t");
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
      if (!VTRanges.count(Key))
        VTRanges[Key].First = Name;
      assert(!VTRanges[Key].Closed && "Gap detected!");
      VTRanges[Key].Last = Name;
    } else if (VTRanges.count(Key)) {
      VTRanges[Key].Closed = true;
    }
  };

  OS << "#ifdef GET_VT_ATTR // (Ty, n, sz, Any, Int, FP, Vec, Sc)\n";
  for (const auto *VT : VTsByNumber) {
    if (!VT)
      continue;
    auto Name = VT->getValueAsString("LLVMName");
    auto Value = VT->getValueAsInt("Value");
    bool IsInteger = VT->getValueAsInt("isInteger");
    bool IsFP = VT->getValueAsInt("isFP");
    bool IsVector = VT->getValueAsInt("isVector");
    bool IsScalable = VT->getValueAsInt("isScalable");

    UpdateVTRange("INTEGER_FIXEDLEN_VECTOR_VALUETYPE", Name,
                  IsInteger && IsVector && !IsScalable);
    UpdateVTRange("INTEGER_SCALABLE_VECTOR_VALUETYPE", Name,
                  IsInteger && IsScalable);
    UpdateVTRange("FP_FIXEDLEN_VECTOR_VALUETYPE", Name,
                  IsFP && IsVector && !IsScalable);
    UpdateVTRange("FP_SCALABLE_VECTOR_VALUETYPE", Name, IsFP && IsScalable);
    UpdateVTRange("FIXEDLEN_VECTOR_VALUETYPE", Name, IsVector && !IsScalable);
    UpdateVTRange("SCALABLE_VECTOR_VALUETYPE", Name, IsScalable);
    UpdateVTRange("VECTOR_VALUETYPE", Name, IsVector);
    UpdateVTRange("INTEGER_VALUETYPE", Name, IsInteger && !IsVector);
    UpdateVTRange("FP_VALUETYPE", Name, IsFP && !IsVector);
    UpdateVTRange("VALUETYPE", Name, Value < 224);

    // clang-format off
    OS << "  GET_VT_ATTR("
       << Name << ", "
       << Value << ", "
       << VT->getValueAsInt("Size") << ", "
       << VT->getValueAsInt("isOverloaded") << ", "
       << (IsInteger ? Name[0] == 'i' ? 3 : 1 : 0) << ", "
       << (IsFP ? Name[0] == 'f' ? 3 : 1 : 0) << ", "
       << IsVector << ", "
       << IsScalable << ")\n";
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

  OS << "#ifdef GET_VT_VECATTR // (Ty, Sc, nElem, ElTy, ElSz)\n";
  for (const auto *VT : VTsByNumber) {
    if (!VT || !VT->getValueAsInt("isVector"))
      continue;
    const auto *ElTy = VT->getValueAsDef("ElementType");
    assert(ElTy);
    // clang-format off
    OS << "  GET_VT_VECATTR("
       << VT->getValueAsString("LLVMName") << ", "
       << VT->getValueAsInt("isScalable") << ", "
       << VT->getValueAsInt("nElem") << ", "
       << ElTy->getName() << ", "
       << ElTy->getValueAsInt("Size") << ")\n";
    // clang-format on
  }
  OS << "#endif\n\n";
}

static TableGen::Emitter::OptClass<VTEmitter> X("gen-vt", "Generate ValueType");
