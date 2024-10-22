//===- JSONBackend.cpp - Generate a JSON dump of all records. -*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This TableGen back end generates a machine-readable representation
// of all the classes and records defined by the input, in JSON format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "json-emitter"

using namespace llvm;

namespace {

class JSONEmitter {
private:
  const RecordKeeper &Records;

  json::Value translateInit(const Init &I);

public:
  explicit JSONEmitter(const RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);
};

} // end anonymous namespace

json::Value JSONEmitter::translateInit(const Init &I) {
  // Init subclasses that we return as JSON primitive values of one
  // kind or another.

  if (isa<UnsetInit>(&I))
    return nullptr;
  if (const auto *Bit = dyn_cast<BitInit>(&I))
    return Bit->getValue() ? 1 : 0;
  if (const auto *Bits = dyn_cast<BitsInit>(&I)) {
    json::Array Array;
    for (unsigned Idx = 0, E = Bits->getNumBits(); Idx < E; ++Idx)
      Array.push_back(translateInit(*Bits->getBit(Idx)));
    return std::move(Array);
  }
  if (const auto *Int = dyn_cast<IntInit>(&I))
    return Int->getValue();
  if (const auto *Str = dyn_cast<StringInit>(&I))
    return Str->getValue();
  if (const auto *List = dyn_cast<ListInit>(&I)) {
    json::Array Array;
    for (const auto *Val : *List)
      Array.push_back(translateInit(*Val));
    return std::move(Array);
  }

  // Init subclasses that we return as JSON objects containing a
  // 'kind' discriminator. For these, we also provide the same
  // translation back into TableGen input syntax that -print-records
  // would give.

  json::Object Obj;
  Obj["printable"] = I.getAsString();

  if (const auto *Def = dyn_cast<DefInit>(&I)) {
    Obj["kind"] = "def";
    Obj["def"] = Def->getDef()->getName();
    return std::move(Obj);
  }
  if (const auto *Var = dyn_cast<VarInit>(&I)) {
    Obj["kind"] = "var";
    Obj["var"] = Var->getName();
    return std::move(Obj);
  }
  if (const auto *VarBit = dyn_cast<VarBitInit>(&I)) {
    if (const auto *Var = dyn_cast<VarInit>(VarBit->getBitVar())) {
      Obj["kind"] = "varbit";
      Obj["var"] = Var->getName();
      Obj["index"] = VarBit->getBitNum();
      return std::move(Obj);
    }
  }
  if (const auto *Dag = dyn_cast<DagInit>(&I)) {
    Obj["kind"] = "dag";
    Obj["operator"] = translateInit(*Dag->getOperator());
    if (auto name = Dag->getName())
      Obj["name"] = name->getAsUnquotedString();
    json::Array Args;
    for (unsigned Idx = 0, E = Dag->getNumArgs(); Idx < E; ++Idx) {
      json::Array Arg;
      Arg.push_back(translateInit(*Dag->getArg(Idx)));
      if (const auto ArgName = Dag->getArgName(Idx))
        Arg.push_back(ArgName->getAsUnquotedString());
      else
        Arg.push_back(nullptr);
      Args.push_back(std::move(Arg));
    }
    Obj["args"] = std::move(Args);
    return std::move(Obj);
  }

  // Final fallback: anything that gets past here is simply given a
  // kind field of 'complex', and the only other field is the standard
  // 'printable' representation.
  assert(!I.isConcrete());
  Obj["kind"] = "complex";
  return std::move(Obj);
}

void JSONEmitter::run(raw_ostream &OS) {
  json::Object Root;

  Root["!tablegen_json_version"] = 1;

  // Prepare the arrays that will list the instances of every class.
  // We mostly fill those in by iterating over the superclasses of
  // each def, but we also want to ensure we store an empty list for a
  // class with no instances at all, so we do a preliminary iteration
  // over the classes, invoking std::map::operator[] to default-
  // construct the array for each one.
  std::map<std::string, json::Array> InstanceLists;
  for (const auto &[ClassName, ClassRec] : Records.getClasses())
    InstanceLists.emplace(ClassRec->getNameInitAsString(), json::Array());

  // Main iteration over the defs.
  for (const auto &[DefName, Def] : Records.getDefs()) {
    const std::string Name = Def->getNameInitAsString();

    json::Object Obj;
    json::Array Fields;

    for (const RecordVal &RV : Def->getValues()) {
      if (!Def->isTemplateArg(RV.getNameInit())) {
        auto Name = RV.getNameInitAsString();
        if (RV.isNonconcreteOK())
          Fields.push_back(Name);
        Obj[Name] = translateInit(*RV.getValue());
      }
    }

    Obj["!fields"] = std::move(Fields);

    json::Array SuperClasses;
    // Add this def to the instance list for each of its superclasses.
    for (const auto &[SuperClass, Loc] : Def->getSuperClasses()) {
      std::string SuperName = SuperClass->getNameInitAsString();
      SuperClasses.push_back(SuperName);
      InstanceLists[SuperName].push_back(Name);
    }

    Obj["!superclasses"] = std::move(SuperClasses);

    Obj["!name"] = Name;
    Obj["!anonymous"] = Def->isAnonymous();

    json::Array Locs;
    for (const SMLoc Loc : Def->getLoc())
      Locs.push_back(SrcMgr.getFormattedLocationNoOffset(Loc));
    Obj["!locs"] = std::move(Locs);

    Root[Name] = std::move(Obj);
  }

  // Make a JSON object from the std::map of instance lists.
  json::Object InstanceOf;
  for (auto &[ClassName, Instances] : InstanceLists)
    InstanceOf[ClassName] = std::move(Instances);
  Root["!instanceof"] = std::move(InstanceOf);

  // Done. Write the output.
  OS << json::Value(std::move(Root)) << "\n";
}

void llvm::EmitJSON(const RecordKeeper &RK, raw_ostream &OS) {
  JSONEmitter(RK).run(OS);
}
