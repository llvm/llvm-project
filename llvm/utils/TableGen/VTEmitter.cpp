//===- VTEmitter.cpp - Generate properties from ValueTypes.td -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
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

  OS << "#ifdef GET_VT_ATTR // (Ty, n, sz)\n";
  for (const auto *VT : VTsByNumber) {
    if (!VT)
      continue;
    auto Name = VT->getName();
    Name = StringSwitch<StringRef>(Name)
               .Case("OtherVT", "Other")
               .Case("FlagVT", "Glue")
               .Case("untyped", "Untyped")
               .Case("MetadataVT", "Metadata")
               .Default(Name);
    auto Value = VT->getValueAsInt("Value");

    UpdateVTRange("VALUETYPE", Name, Value < 224);

    // clang-format off
    OS << "  GET_VT_ATTR("
       << Name << ", "
       << Value << ", "
       << VT->getValueAsInt("Size") << ")\n";
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
}

static TableGen::Emitter::OptClass<VTEmitter> X("gen-vt", "Generate ValueType");
