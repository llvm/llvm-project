//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits JSON describing each property grouped by path.
//
//===----------------------------------------------------------------------===//

#include "LLDBTableGenBackends.h"
#include "LLDBTableGenUtils.h"
#include "llvm/Support/JSON.h"
#include "llvm/TableGen/Record.h"
#include <vector>

using namespace llvm;
using namespace lldb_private;

static void emitProperty(const Record *Property, json::OStream &OS) {
  OS.attribute("name", Property->getValueAsString("Name"));
  OS.attribute("type", Property->getValueAsString("Type"));

  bool hasDefaultUnsignedValue = Property->getValue("HasDefaultUnsignedValue");
  bool hasDefaultBooleanValue = Property->getValue("HasDefaultBooleanValue");
  bool hasDefaultStringValue = Property->getValue("HasDefaultStringValue");

  // Emit the default uint value.
  if (hasDefaultBooleanValue) {
    assert(hasDefaultUnsignedValue);
    bool value = Property->getValueAsInt("DefaultUnsignedValue") != 0;
    OS.attribute("default", value ? "true" : "false");
  } else if (hasDefaultUnsignedValue) {
    OS.attribute("default", std::to_string(Property->getValueAsInt(
                                "DefaultUnsignedValue")));
  } else if (hasDefaultStringValue) {
    if (auto *D = Property->getValue("DefaultStringValue"))
      OS.attribute("default", D->getValue()->getAsUnquotedString());
  }

  // FIXME: Get enum values. This requires the enum values to be declared in
  // tablegen as well.

  if (auto *D = Property->getValue("Description"))
    OS.attribute("description", D->getValue()->getAsUnquotedString());
}

/// Emits all property initializers to the raw_ostream.
static void emityProperties(const std::vector<const Record *> &PropertyRecords,
                            json::OStream &OS) {

  for (const Record *R : PropertyRecords)
    OS.object([&] { emitProperty(R, OS); });
}

void lldb_private::EmitPropertyDocsJson(const RecordKeeper &Records,
                                        raw_ostream &OS) {
  json::OStream JsonOS(OS);
  JsonOS.array([&] {
    ArrayRef<const Record *> Properties =
        Records.getAllDerivedDefinitions("Property");
    for (auto &Rec : getRecordsByName(Properties, "Path")) {
      JsonOS.object([&] {
        JsonOS.attribute("path", Rec.first);
        JsonOS.attributeBegin("properties");
        JsonOS.array([&] { emityProperties(Rec.second, JsonOS); });
        JsonOS.attributeEnd(); // properties
      });
    }
  });
}
