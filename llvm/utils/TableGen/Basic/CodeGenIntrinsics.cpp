//===- CodeGenIntrinsics.cpp - Intrinsic Class Wrapper --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Intrinsic' TableGen class.
//
//===----------------------------------------------------------------------===//

#include "CodeGenIntrinsics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cassert>
using namespace llvm;

//===----------------------------------------------------------------------===//
// CodeGenIntrinsic Implementation
//===----------------------------------------------------------------------===//

CodeGenIntrinsicContext::CodeGenIntrinsicContext(const RecordKeeper &RC) {
  for (const Record *Rec : RC.getAllDerivedDefinitions("IntrinsicProperty"))
    if (Rec->getValueAsBit("IsDefault"))
      DefaultProperties.push_back(Rec);

  // The maximum number of values that an intrinsic can return is the size of
  // of `IIT_RetNumbers` list - 1 (since we index into this list using the
  // number of return values as the index).
  const auto *IIT_RetNumbers =
      dyn_cast_or_null<ListInit>(RC.getGlobal("IIT_RetNumbers"));
  if (!IIT_RetNumbers)
    PrintFatalError("unable to find 'IIT_RetNumbers' list");
  MaxNumReturn = IIT_RetNumbers->size() - 1;
}

CodeGenIntrinsicTable::CodeGenIntrinsicTable(const RecordKeeper &RC) {
  CodeGenIntrinsicContext Ctx(RC);

  ArrayRef<const Record *> Defs = RC.getAllDerivedDefinitions("Intrinsic");
  Intrinsics.reserve(Defs.size());

  for (const Record *Def : Defs)
    Intrinsics.emplace_back(CodeGenIntrinsic(Def, Ctx));

  llvm::sort(Intrinsics,
             [](const CodeGenIntrinsic &LHS, const CodeGenIntrinsic &RHS) {
               // Order target independent intrinsics before target dependent
               // ones.
               bool LHSHasTarget = !LHS.TargetPrefix.empty();
               bool RHSHasTarget = !RHS.TargetPrefix.empty();

               // To ensure deterministic sorted order when duplicates are
               // present, use record ID as a tie-breaker similar to
               // sortAndReportDuplicates in Utils.cpp.
               unsigned LhsID = LHS.TheDef->getID();
               unsigned RhsID = RHS.TheDef->getID();

               return std::tie(LHSHasTarget, LHS.Name, LhsID) <
                      std::tie(RHSHasTarget, RHS.Name, RhsID);
             });

  Targets.push_back({"", 0, 0});
  for (size_t I = 0, E = Intrinsics.size(); I < E; ++I)
    if (Intrinsics[I].TargetPrefix != Targets.back().Name) {
      Targets.back().Count = I - Targets.back().Offset;
      Targets.push_back({Intrinsics[I].TargetPrefix, I, 0});
    }
  Targets.back().Count = Intrinsics.size() - Targets.back().Offset;

  CheckDuplicateIntrinsics();
  CheckTargetIndependentIntrinsics();
  CheckOverloadSuffixConflicts();
}

// Check for duplicate intrinsic names.
void CodeGenIntrinsicTable::CheckDuplicateIntrinsics() const {
  // Since the Intrinsics vector is already sorted by name, if there are 2 or
  // more intrinsics with duplicate names, they will appear adjacent in sorted
  // order. Note that if the intrinsic name was derived from the record name
  // there cannot be be duplicate as TableGen parser would have flagged that.
  // However, if the name was specified in the intrinsic definition, then its
  // possible to have duplicate names.
  auto I = std::adjacent_find(
      Intrinsics.begin(), Intrinsics.end(),
      [](const CodeGenIntrinsic &Int1, const CodeGenIntrinsic &Int2) {
        return Int1.Name == Int2.Name;
      });
  if (I == Intrinsics.end())
    return;

  // Found a duplicate intrinsics.
  const CodeGenIntrinsic &First = *I;
  const CodeGenIntrinsic &Second = *(I + 1);
  PrintError(Second.TheDef,
             Twine("Intrinsic `") + First.Name + "` is already defined");
  PrintFatalNote(First.TheDef, "Previous definition here");
}

// For target independent intrinsics, check that their second dotted component
// does not match any target name.
void CodeGenIntrinsicTable::CheckTargetIndependentIntrinsics() const {
  SmallDenseSet<StringRef> TargetNames;
  for (const auto &Target : ArrayRef(Targets).drop_front())
    TargetNames.insert(Target.Name);

  // Set of target independent intrinsics.
  const auto &Set = Targets[0];
  for (const auto &Int : ArrayRef(&Intrinsics[Set.Offset], Set.Count)) {
    StringRef Name = Int.Name;
    StringRef Prefix = Name.drop_front(5).split('.').first;
    if (!TargetNames.contains(Prefix))
      continue;
    PrintFatalError(Int.TheDef,
                    "target independent intrinsic `" + Name +
                        "' has prefix `llvm." + Prefix +
                        "` that conflicts with intrinsics for target `" +
                        Prefix + "`");
  }
}

// Return true if the given Suffix looks like a mangled type. Note that this
// check is conservative, but allows all existing LLVM intrinsic suffixes to be
// considered as not looking like a mangling suffix.
static bool doesSuffixLookLikeMangledType(StringRef Suffix) {
  // Try to match against possible mangling suffixes for various types.
  // See getMangledTypeStr() for the mangling suffixes possible. It includes
  //  pointer       : p[0-9]+
  //  array         : a[0-9]+.+
  //  struct:       : s_/sl_.+
  //  function      : f_.+
  //  vector        : v/nxv[0-9]+.+
  //  target type   : t.+
  //  integer       : i[0-9]+
  //  named types   : See `NamedTypes` below.

  // Match anything with an _, so match function and struct types.
  if (Suffix.contains('_'))
    return true;

  // [av][0-9]+.+, simplified to [av][0-9].+
  if (Suffix.size() >= 2 && is_contained("av", Suffix[0]) && isDigit(Suffix[1]))
    return true;

  // nxv[0-9]+.+, simplified to nxv[0-9].+
  if (Suffix.size() >= 4 && Suffix.starts_with("nxv") && isDigit(Suffix[3]))
    return true;

  // t.+
  if (Suffix.size() > 1 && Suffix.starts_with('t'))
    return false;

  // [pi][0-9]+
  if (Suffix.size() > 1 && is_contained("pi", Suffix[0]) &&
      all_of(Suffix.drop_front(), isDigit))
    return true;

  // Match one of the named types.
  static constexpr StringLiteral NamedTypes[] = {
      "isVoid", "Metadata", "f16",  "f32",     "f64",
      "f80",    "f128",     "bf16", "ppcf128", "x86amx"};
  return is_contained(NamedTypes, Suffix);
}

// Check for conflicts with overloaded intrinsics. If there exists an overloaded
// intrinsic with base name `llvm.target.foo`, LLVM will add a mangling suffix
// to it to encode the overload types. This mangling suffix is 1 or more .
// prefixed mangled type string as defined in `getMangledTypeStr`. If there
// exists another intrinsic `llvm.target.foo[.<suffixN>]+`, which has the same
// prefix as the overloaded intrinsic, its possible that there may be a name
// conflict with the overloaded intrinsic and either one may interfere with name
// lookup for the other, leading to wrong intrinsic ID being assigned.
//
// The actual name lookup in the intrinsic name table is done by a search
// on each successive '.' separted component of the intrinsic name (see
// `lookupLLVMIntrinsicByName`). Consider first the case where there exists a
// non-overloaded intrinsic `llvm.target.foo[.suffix]+`. For the non-overloaded
// intrinsics, the name lookup is an exact match, so the presence of the
// overloaded intrinsic with the same prefix will not interfere with the
// search. However, a lookup intended to match the overloaded intrinsic might be
// affected by the presence of another entry in the name table with the same
// prefix.
//
// Since LLVM's name lookup first selects the target specific (or target
// independent) slice of the name table to look into, intrinsics in 2 different
// targets cannot conflict with each other. Within a specific target,
// if we have an overloaded intrinsic with name `llvm.target.foo` and another
// one with same prefix and one or more suffixes `llvm.target.foo[.<suffixN>]+`,
// then the name search will try to first match against suffix0, then suffix1
// etc. If suffix0 can match a mangled type, then the search for an
// `llvm.target.foo` with a mangling suffix can match against suffix0,
// preventing a match with `llvm.target.foo`. If suffix0 cannot match a mangled
// type, then that cannot happen, so we do not need to check for later suffixes.
//
// Generalizing, the `llvm.target.foo[.suffixN]+` will cause a conflict if the
// first suffix (.suffix0) can match a mangled type (and then we do not need to
// check later suffixes) and will not cause a conflict if it cannot (and then
// again, we do not need to check for later suffixes).
void CodeGenIntrinsicTable::CheckOverloadSuffixConflicts() const {
  for (const TargetSet &Set : Targets) {
    const CodeGenIntrinsic *Overloaded = nullptr;
    for (const CodeGenIntrinsic &Int : (*this)[Set]) {
      // If we do not have an overloaded intrinsic to check against, nothing
      // to do except potentially identifying this as a candidate for checking
      // against in future iteration.
      if (!Overloaded) {
        if (Int.isOverloaded)
          Overloaded = &Int;
        continue;
      }

      StringRef Name = Int.Name;
      StringRef OverloadName = Overloaded->Name;
      // If we have an overloaded intrinsic to check again, check if its name is
      // a proper prefix of this intrinsic.
      if (Name.starts_with(OverloadName) && Name[OverloadName.size()] == '.') {
        // If yes, verify suffixes and flag an error.
        StringRef Suffixes = Name.drop_front(OverloadName.size() + 1);

        // Only need to look at the first suffix.
        StringRef Suffix0 = Suffixes.split('.').first;

        if (!doesSuffixLookLikeMangledType(Suffix0))
          continue;

        unsigned SuffixSize = OverloadName.size() + 1 + Suffix0.size();
        // If suffix looks like mangling suffix, flag it as an error.
        PrintError(Int.TheDef->getLoc(),
                   "intrinsic `" + Name + "` cannot share prefix `" +
                       Name.take_front(SuffixSize) +
                       "` with another overloaded intrinsic `" + OverloadName +
                       "`");
        PrintNote(Overloaded->TheDef->getLoc(),
                  "Overloaded intrinsic `" + OverloadName + "` defined here");
        continue;
      }

      // If we find an intrinsic that is not a proper prefix, any later
      // intrinsic is also not going to be a proper prefix, so invalidate the
      // overloaded to check against.
      Overloaded = nullptr;
    }
  }
}

const CodeGenIntrinsic &CodeGenIntrinsicMap::operator[](const Record *Record) {
  if (!Record->isSubClassOf("Intrinsic"))
    PrintFatalError("Intrinsic defs should be subclass of 'Intrinsic' class");

  auto [Iter, Inserted] = Map.try_emplace(Record);
  if (Inserted)
    Iter->second = std::make_unique<CodeGenIntrinsic>(Record, Ctx);
  return *Iter->second;
}

CodeGenIntrinsic::CodeGenIntrinsic(const Record *R,
                                   const CodeGenIntrinsicContext &Ctx)
    : TheDef(R) {
  StringRef DefName = TheDef->getName();
  ArrayRef<SMLoc> DefLoc = R->getLoc();

  if (!DefName.starts_with("int_"))
    PrintFatalError(DefLoc,
                    "Intrinsic '" + DefName + "' does not start with 'int_'!");

  EnumName = DefName.substr(4);

  // Ignore a missing ClangBuiltinName field.
  ClangBuiltinName =
      R->getValueAsOptionalString("ClangBuiltinName").value_or("");
  // Ignore a missing MSBuiltinName field.
  MSBuiltinName = R->getValueAsOptionalString("MSBuiltinName").value_or("");

  TargetPrefix = R->getValueAsString("TargetPrefix");
  Name = R->getValueAsString("LLVMName").str();

  if (Name == "") {
    // If an explicit name isn't specified, derive one from the DefName.
    Name = "llvm." + EnumName.str();
    llvm::replace(Name, '_', '.');
  } else {
    // Verify it starts with "llvm.".
    if (!StringRef(Name).starts_with("llvm."))
      PrintFatalError(DefLoc, "Intrinsic '" + DefName +
                                  "'s name does not start with 'llvm.'!");
  }

  // If TargetPrefix is specified, make sure that Name starts with
  // "llvm.<targetprefix>.".
  if (!TargetPrefix.empty()) {
    StringRef Prefix = StringRef(Name).drop_front(5); // Drop llvm.
    if (!Prefix.consume_front(TargetPrefix) || !Prefix.starts_with('.'))
      PrintFatalError(DefLoc, "Intrinsic '" + DefName +
                                  "' does not start with 'llvm." +
                                  TargetPrefix + ".'!");
  }

  unsigned NumRet = R->getValueAsListInit("RetTypes")->size();
  if (NumRet > Ctx.MaxNumReturn)
    PrintFatalError(DefLoc, "intrinsics can only return upto " +
                                Twine(Ctx.MaxNumReturn) + " values, '" +
                                DefName + "' returns " + Twine(NumRet) +
                                " values");

  const Record *TypeInfo = R->getValueAsDef("TypeInfo");
  if (!TypeInfo->isSubClassOf("TypeInfoGen"))
    PrintFatalError(DefLoc, "TypeInfo field in " + DefName +
                                " should be of subclass of TypeInfoGen!");

  isOverloaded = TypeInfo->getValueAsBit("isOverloaded");
  const ListInit *TypeList = TypeInfo->getValueAsListInit("Types");

  // Types field is a concatenation of Return types followed by Param types.
  unsigned Idx = 0;
  for (; Idx < NumRet; ++Idx)
    IS.RetTys.push_back(TypeList->getElementAsRecord(Idx));

  for (unsigned E = TypeList->size(); Idx < E; ++Idx)
    IS.ParamTys.push_back(TypeList->getElementAsRecord(Idx));

  // Parse the intrinsic properties.
  const ListInit *PropList = R->getValueAsListInit("IntrProperties");
  for (unsigned i = 0, e = PropList->size(); i != e; ++i) {
    const Record *Property = PropList->getElementAsRecord(i);
    assert(Property->isSubClassOf("IntrinsicProperty") &&
           "Expected a property!");

    setProperty(Property);
  }

  // Set default properties to true.
  setDefaultProperties(Ctx.DefaultProperties);

  // Also record the SDPatternOperator Properties.
  Properties = parseSDPatternOperatorProperties(R);

  // Sort the argument attributes for later benefit.
  for (auto &Attrs : ArgumentAttributes)
    llvm::sort(Attrs);
}

void CodeGenIntrinsic::setDefaultProperties(
    ArrayRef<const Record *> DefaultProperties) {
  // opt-out of using default attributes.
  if (TheDef->getValueAsBit("DisableDefaultAttributes"))
    return;

  for (const Record *Rec : DefaultProperties)
    setProperty(Rec);
}

void CodeGenIntrinsic::setProperty(const Record *R) {
  if (R->getName() == "IntrNoMem")
    ME = MemoryEffects::none();
  else if (R->getName() == "IntrReadMem") {
    if (ME.onlyWritesMemory())
      PrintFatalError(TheDef->getLoc(),
                      Twine("IntrReadMem cannot be used after IntrNoMem or "
                            "IntrWriteMem. Default is ReadWrite"));
    ME &= MemoryEffects::readOnly();
  } else if (R->getName() == "IntrWriteMem") {
    if (ME.onlyReadsMemory())
      PrintFatalError(TheDef->getLoc(),
                      Twine("IntrWriteMem cannot be used after IntrNoMem or "
                            "IntrReadMem. Default is ReadWrite"));
    ME &= MemoryEffects::writeOnly();
  } else if (R->getName() == "IntrArgMemOnly")
    ME &= MemoryEffects::argMemOnly();
  else if (R->getName() == "IntrInaccessibleMemOnly")
    ME &= MemoryEffects::inaccessibleMemOnly();
  else if (R->getName() == "IntrInaccessibleMemOrArgMemOnly")
    ME &= MemoryEffects::inaccessibleOrArgMemOnly();
  else if (R->getName() == "Commutative")
    isCommutative = true;
  else if (R->getName() == "Throws")
    canThrow = true;
  else if (R->getName() == "IntrNoDuplicate")
    isNoDuplicate = true;
  else if (R->getName() == "IntrNoMerge")
    isNoMerge = true;
  else if (R->getName() == "IntrConvergent")
    isConvergent = true;
  else if (R->getName() == "IntrNoReturn")
    isNoReturn = true;
  else if (R->getName() == "IntrNoCallback")
    isNoCallback = true;
  else if (R->getName() == "IntrNoSync")
    isNoSync = true;
  else if (R->getName() == "IntrNoFree")
    isNoFree = true;
  else if (R->getName() == "IntrWillReturn")
    isWillReturn = !isNoReturn;
  else if (R->getName() == "IntrCold")
    isCold = true;
  else if (R->getName() == "IntrSpeculatable")
    isSpeculatable = true;
  else if (R->getName() == "IntrHasSideEffects")
    hasSideEffects = true;
  else if (R->getName() == "IntrStrictFP")
    isStrictFP = true;
  else if (R->isSubClassOf("NoCapture")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NoCapture);
  } else if (R->isSubClassOf("NoAlias")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NoAlias);
  } else if (R->isSubClassOf("NoUndef")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NoUndef);
  } else if (R->isSubClassOf("NonNull")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NonNull);
  } else if (R->isSubClassOf("Returned")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, Returned);
  } else if (R->isSubClassOf("ReadOnly")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, ReadOnly);
  } else if (R->isSubClassOf("WriteOnly")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, WriteOnly);
  } else if (R->isSubClassOf("ReadNone")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, ReadNone);
  } else if (R->isSubClassOf("ImmArg")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, ImmArg);
  } else if (R->isSubClassOf("Align")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    uint64_t Align = R->getValueAsInt("Align");
    addArgAttribute(ArgNo, Alignment, Align);
  } else if (R->isSubClassOf("Dereferenceable")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    uint64_t Bytes = R->getValueAsInt("Bytes");
    addArgAttribute(ArgNo, Dereferenceable, Bytes);
  } else
    llvm_unreachable("Unknown property!");
}

bool CodeGenIntrinsic::isParamAPointer(unsigned ParamIdx) const {
  if (ParamIdx >= IS.ParamTys.size())
    return false;
  return (IS.ParamTys[ParamIdx]->isSubClassOf("LLVMQualPointerType") ||
          IS.ParamTys[ParamIdx]->isSubClassOf("LLVMAnyPointerType"));
}

bool CodeGenIntrinsic::isParamImmArg(unsigned ParamIdx) const {
  // Convert argument index to attribute index starting from `FirstArgIndex`.
  ++ParamIdx;
  if (ParamIdx >= ArgumentAttributes.size())
    return false;
  ArgAttribute Val{ImmArg, 0};
  return std::binary_search(ArgumentAttributes[ParamIdx].begin(),
                            ArgumentAttributes[ParamIdx].end(), Val);
}

void CodeGenIntrinsic::addArgAttribute(unsigned Idx, ArgAttrKind AK,
                                       uint64_t V) {
  if (Idx >= ArgumentAttributes.size())
    ArgumentAttributes.resize(Idx + 1);
  ArgumentAttributes[Idx].emplace_back(AK, V);
}
