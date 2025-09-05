//===- TosaUtilsGen.cpp - Tosa utility generator -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TosaUtilsGen generates common utility functions for Tosa validation.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <list>
#include <optional>

using llvm::formatv;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringMap;
using llvm::StringRef;
using mlir::tblgen::Operator;

//===----------------------------------------------------------------------===//
// Availability Wrapper Class
//===----------------------------------------------------------------------===//

namespace {
// Wrapper class with helper methods for accessing availability defined in
// TableGen.
class Availability {
public:
  explicit Availability(const Record *def);

  // Returns the name of the direct TableGen class for this availability
  // instance.
  StringRef getClass() const;

  // Returns the name of the query function insided the generated C++ interface.
  StringRef getQueryFnName() const;

  // Returns the return type of the query function insided the generated C++
  // interface.
  StringRef getQueryFnRetType() const;

  // Returns the code for merging availability requirements.
  StringRef getMergeActionCode() const;

  // Returns the initializer expression for initializing the final availability
  // requirements.
  StringRef getMergeInitializer() const;

  // Returns the C++ statements for preparing availability instance.
  StringRef getMergeInstancePreparation() const;

  // Returns the concrete availability instance carried in this case.
  StringRef getMergeInstance() const;

  // Returns the underlying LLVM TableGen Record.
  const llvm::Record *getDef() const { return def; }

private:
  // The TableGen definition of this availability.
  const llvm::Record *def;
};
} // namespace

Availability::Availability(const llvm::Record *def) : def(def) {
  assert(def->isSubClassOf("Availability") &&
         "must be subclass of TableGen 'Availability' class");
}

StringRef Availability::getClass() const {
  if (def->getDirectSuperClasses().size() != 1) {
    PrintFatalError(def->getLoc(),
                    "expected to only have one direct superclass");
  }
  return def->getDirectSuperClasses().front().first->getName();
}

StringRef Availability::getQueryFnRetType() const {
  return def->getValueAsString("queryFnRetType");
}

StringRef Availability::getQueryFnName() const {
  return def->getValueAsString("queryFnName");
}

StringRef Availability::getMergeActionCode() const {
  return def->getValueAsString("mergeAction");
}

StringRef Availability::getMergeInitializer() const {
  return def->getValueAsString("initializer");
}

StringRef Availability::getMergeInstancePreparation() const {
  return def->getValueAsString("instancePreparation");
}

StringRef Availability::getMergeInstance() const {
  return def->getValueAsString("instance");
}

// Returns the availability spec of the given `def`.
static std::vector<Availability> getAvailabilities(const Record &def) {
  std::vector<Availability> availabilities;

  if (def.getValue("availability")) {
    std::vector<const Record *> availDefs =
        def.getValueAsListOfDefs("availability");
    availabilities.reserve(availDefs.size());
    for (const Record *avail : availDefs)
      availabilities.emplace_back(avail);
  }

  return availabilities;
}

//===----------------------------------------------------------------------===//
// Tosa Availability Impl AutoGen
//===----------------------------------------------------------------------===//

static void emitAvailabilityImpl(const Operator &srcOp, raw_ostream &os) {
  mlir::tblgen::FmtContext fctx;
  fctx.addSubst("overall", "tblgen_overall");

  std::vector<Availability> opAvailabilities =
      getAvailabilities(srcOp.getDef());

  // First collect all availability classes this op should implement.
  // All availability instances keep information for the generated interface and
  // the instance's specific requirement. Here we remember a random instance so
  // we can get the information regarding the generated interface.
  llvm::StringMap<Availability> availClasses;
  for (const Availability &avail : opAvailabilities)
    availClasses.try_emplace(avail.getClass(), avail);

  // Then generate implementation for each availability class.
  for (const auto &availClass : availClasses) {
    StringRef availClassName = availClass.getKey();
    Availability avail = availClass.getValue();

    // Generate the implementation method signature.
    os << formatv("{0} {1}::{2}() {{\n", avail.getQueryFnRetType(),
                  srcOp.getCppClassName(), avail.getQueryFnName());

    // Create the variable for the final requirement and initialize it.
    os << formatv("  {0} tblgen_overall = {1};\n", avail.getQueryFnRetType(),
                  avail.getMergeInitializer());

    // Update with the op's specific availability spec.
    for (const Availability &avail : opAvailabilities)
      if (avail.getClass() == availClassName &&
          (!avail.getMergeInstancePreparation().empty() ||
           !avail.getMergeActionCode().empty())) {
        os << "  {\n    "
           // Prepare this instance.
           << avail.getMergeInstancePreparation()
           << "\n    "
           // Merge this instance.
           << std::string(
                  tgfmt(avail.getMergeActionCode(),
                        &fctx.addSubst("instance", avail.getMergeInstance())))
           << ";\n  }\n";
      }

    os << "  return tblgen_overall;\n";
    os << "}\n";
  }
}

static bool emitAvailabilityImpl(const RecordKeeper &recordKeeper,
                                 raw_ostream &os) {
  llvm::emitSourceFileHeader("Tosa Op Availability Implementations", os,
                             recordKeeper);

  auto defs = recordKeeper.getAllDerivedDefinitions("Tosa_Op");
  for (const auto *def : defs) {
    Operator op(def);
    if (def->getValueAsBit("autogenAvailability"))
      emitAvailabilityImpl(op, os);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Op Availability Implementation Hook Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genOpAvailabilityImpl("gen-tosa-avail-impls",
                          "Generate Tosa operation utility definitions",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return emitAvailabilityImpl(records, os);
                          });
