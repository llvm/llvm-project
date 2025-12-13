//===- TGParser.h - Parser for TableGen Files -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class represents the Parser for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TABLEGEN_TGPARSER_H
#define LLVM_LIB_TABLEGEN_TGPARSER_H

#include "TGLexer.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <map>

namespace llvm {
class SourceMgr;
class Twine;
struct ForeachLoop;
struct MultiClass;
struct SubClassReference;
struct SubMultiClassReference;

struct LetRecord {
  const StringInit *Name;
  std::vector<unsigned> Bits;
  const Init *Value;
  SMLoc Loc;
  LetRecord(const StringInit *N, ArrayRef<unsigned> B, const Init *V, SMLoc L)
      : Name(N), Bits(B), Value(V), Loc(L) {}
};

/// RecordsEntry - Holds exactly one of a Record, ForeachLoop, or
/// AssertionInfo.
struct RecordsEntry {
  std::unique_ptr<Record> Rec;
  std::unique_ptr<ForeachLoop> Loop;
  std::unique_ptr<Record::AssertionInfo> Assertion;
  std::unique_ptr<Record::DumpInfo> Dump;

  void dump() const;

  RecordsEntry() = default;
  RecordsEntry(std::unique_ptr<Record> Rec);
  RecordsEntry(std::unique_ptr<ForeachLoop> Loop);
  RecordsEntry(std::unique_ptr<Record::AssertionInfo> Assertion);
  RecordsEntry(std::unique_ptr<Record::DumpInfo> Dump);
};

/// ForeachLoop - Record the iteration state associated with a for loop.
/// This is used to instantiate items in the loop body.
///
/// IterVar is allowed to be null, in which case no iteration variable is
/// defined in the loop at all. (This happens when a ForeachLoop is
/// constructed by desugaring an if statement.)
struct ForeachLoop {
  SMLoc Loc;
  const VarInit *IterVar;
  const Init *ListValue;
  std::vector<RecordsEntry> Entries;

  void dump() const;

  ForeachLoop(SMLoc Loc, const VarInit *IVar, const Init *LValue)
      : Loc(Loc), IterVar(IVar), ListValue(LValue) {}
};

struct DefsetRecord {
  SMLoc Loc;
  const RecTy *EltTy = nullptr;
  SmallVector<Init *, 16> Elements;
};

struct MultiClass {
  Record Rec; // Placeholder for template args and Name.
  std::vector<RecordsEntry> Entries;

  void dump() const;

  MultiClass(StringRef Name, SMLoc Loc, RecordKeeper &Records)
      : Rec(Name, Loc, Records, Record::RK_MultiClass) {}
};

class TGVarScope {
public:
  enum ScopeKind { SK_Local, SK_Record, SK_ForeachLoop, SK_MultiClass };

private:
  ScopeKind Kind;
  std::unique_ptr<TGVarScope> Parent;
  // A scope to hold variable definitions from defvar.
  std::map<std::string, const Init *, std::less<>> Vars;
  Record *CurRec = nullptr;
  ForeachLoop *CurLoop = nullptr;
  MultiClass *CurMultiClass = nullptr;

public:
  TGVarScope(std::unique_ptr<TGVarScope> Parent)
      : Kind(SK_Local), Parent(std::move(Parent)) {}
  TGVarScope(std::unique_ptr<TGVarScope> Parent, Record *Rec)
      : Kind(SK_Record), Parent(std::move(Parent)), CurRec(Rec) {}
  TGVarScope(std::unique_ptr<TGVarScope> Parent, ForeachLoop *Loop)
      : Kind(SK_ForeachLoop), Parent(std::move(Parent)), CurLoop(Loop) {}
  TGVarScope(std::unique_ptr<TGVarScope> Parent, MultiClass *Multiclass)
      : Kind(SK_MultiClass), Parent(std::move(Parent)),
        CurMultiClass(Multiclass) {}

  std::unique_ptr<TGVarScope> extractParent() {
    // This is expected to be called just before we are destructed, so
    // it doesn't much matter what state we leave 'parent' in.
    return std::move(Parent);
  }

  const Init *getVar(RecordKeeper &Records, MultiClass *ParsingMultiClass,
                     const StringInit *Name, SMRange NameLoc,
                     bool TrackReferenceLocs) const;

  bool varAlreadyDefined(StringRef Name) const {
    // When we check whether a variable is already defined, for the purpose of
    // reporting an error on redefinition, we don't look up to the parent
    // scope, because it's all right to shadow an outer definition with an
    // inner one.
    return Vars.find(Name) != Vars.end();
  }

  void addVar(StringRef Name, const Init *I) {
    bool Ins = Vars.try_emplace(Name.str(), I).second;
    (void)Ins;
    assert(Ins && "Local variable already exists");
  }

  bool isOutermost() const { return Parent == nullptr; }
};

class TGParser {
  TGLexer Lex;
  std::vector<SmallVector<LetRecord, 4>> LetStack;
  std::map<std::string, std::unique_ptr<MultiClass>> MultiClasses;
  std::map<std::string, const RecTy *> TypeAliases;

  /// Loops - Keep track of any foreach loops we are within.
  ///
  std::vector<std::unique_ptr<ForeachLoop>> Loops;

  SmallVector<DefsetRecord *, 2> Defsets;

  /// CurMultiClass - If we are parsing a 'multiclass' definition, this is the
  /// current value.
  MultiClass *CurMultiClass;

  /// CurScope - Innermost of the current nested scopes for 'defvar' variables.
  std::unique_ptr<TGVarScope> CurScope;

  // Record tracker
  RecordKeeper &Records;

  // A "named boolean" indicating how to parse identifiers. Usually
  // identifiers map to some existing object but in special cases
  // (e.g. parsing def names) no such object exists yet because we are
  // in the middle of creating in. For those situations, allow the
  // parser to ignore missing object errors.
  enum IDParseMode {
    ParseValueMode, // We are parsing a value we expect to look up.
    ParseNameMode,  // We are parsing a name of an object that does not yet
                    // exist.
  };

  bool NoWarnOnUnusedTemplateArgs = false;
  bool TrackReferenceLocs = false;

public:
  TGParser(SourceMgr &SM, ArrayRef<std::string> Macros, RecordKeeper &records,
           const bool NoWarnOnUnusedTemplateArgs = false,
           const bool TrackReferenceLocs = false)
      : Lex(SM, Macros), CurMultiClass(nullptr), Records(records),
        NoWarnOnUnusedTemplateArgs(NoWarnOnUnusedTemplateArgs),
        TrackReferenceLocs(TrackReferenceLocs) {}

  /// ParseFile - Main entrypoint for parsing a tblgen file. These parser
  /// routines return true on error, or false on success.
  bool ParseFile();

  bool Error(SMLoc L, const Twine &Msg) const {
    PrintError(L, Msg);
    return true;
  }
  bool TokError(const Twine &Msg) const { return Error(Lex.getLoc(), Msg); }
  const TGLexer::DependenciesSetTy &getDependencies() const {
    return Lex.getDependencies();
  }

  TGVarScope *PushScope() {
    CurScope = std::make_unique<TGVarScope>(std::move(CurScope));
    // Returns a pointer to the new scope, so that the caller can pass it back
    // to PopScope which will check by assertion that the pushes and pops
    // match up properly.
    return CurScope.get();
  }
  TGVarScope *PushScope(Record *Rec) {
    CurScope = std::make_unique<TGVarScope>(std::move(CurScope), Rec);
    return CurScope.get();
  }
  TGVarScope *PushScope(ForeachLoop *Loop) {
    CurScope = std::make_unique<TGVarScope>(std::move(CurScope), Loop);
    return CurScope.get();
  }
  TGVarScope *PushScope(MultiClass *Multiclass) {
    CurScope = std::make_unique<TGVarScope>(std::move(CurScope), Multiclass);
    return CurScope.get();
  }
  void PopScope(TGVarScope *ExpectedStackTop) {
    assert(ExpectedStackTop == CurScope.get() &&
           "Mismatched pushes and pops of local variable scopes");
    CurScope = CurScope->extractParent();
  }

private: // Semantic analysis methods.
  bool AddValue(Record *TheRec, SMLoc Loc, const RecordVal &RV);
  /// Set the value of a RecordVal within the given record. If `OverrideDefLoc`
  /// is set, the provided location overrides any existing location of the
  /// RecordVal.
  bool SetValue(Record *TheRec, SMLoc Loc, const Init *ValName,
                ArrayRef<unsigned> BitList, const Init *V,
                bool AllowSelfAssignment = false, bool OverrideDefLoc = true);
  bool AddSubClass(Record *Rec, SubClassReference &SubClass);
  bool AddSubClass(RecordsEntry &Entry, SubClassReference &SubClass);
  bool AddSubMultiClass(MultiClass *CurMC,
                        SubMultiClassReference &SubMultiClass);

  using SubstStack = SmallVector<std::pair<const Init *, const Init *>, 8>;

  bool addEntry(RecordsEntry E);
  bool resolve(const ForeachLoop &Loop, SubstStack &Stack, bool Final,
               std::vector<RecordsEntry> *Dest, SMLoc *Loc = nullptr);
  bool resolve(const std::vector<RecordsEntry> &Source, SubstStack &Substs,
               bool Final, std::vector<RecordsEntry> *Dest,
               SMLoc *Loc = nullptr);
  bool addDefOne(std::unique_ptr<Record> Rec);

  using ArgValueHandler = std::function<void(const Init *, const Init *)>;
  bool resolveArguments(
      const Record *Rec, ArrayRef<const ArgumentInit *> ArgValues, SMLoc Loc,
      ArgValueHandler ArgValueHandler = [](const Init *, const Init *) {});
  bool resolveArgumentsOfClass(MapResolver &R, const Record *Rec,
                               ArrayRef<const ArgumentInit *> ArgValues,
                               SMLoc Loc);
  bool resolveArgumentsOfMultiClass(SubstStack &Substs, MultiClass *MC,
                                    ArrayRef<const ArgumentInit *> ArgValues,
                                    const Init *DefmName, SMLoc Loc);

private: // Parser methods.
  bool consume(tgtok::TokKind K);
  bool ParseObjectList(MultiClass *MC = nullptr);
  bool ParseObject(MultiClass *MC);
  bool ParseClass();
  bool ParseMultiClass();
  bool ParseDefm(MultiClass *CurMultiClass);
  bool ParseDef(MultiClass *CurMultiClass);
  bool ParseDefset();
  bool ParseDeftype();
  bool ParseDefvar(Record *CurRec = nullptr);
  bool ParseDump(MultiClass *CurMultiClass, Record *CurRec = nullptr);
  bool ParseForeach(MultiClass *CurMultiClass);
  bool ParseIf(MultiClass *CurMultiClass);
  bool ParseIfBody(MultiClass *CurMultiClass, StringRef Kind);
  bool ParseAssert(MultiClass *CurMultiClass, Record *CurRec = nullptr);
  bool ParseTopLevelLet(MultiClass *CurMultiClass);
  void ParseLetList(SmallVectorImpl<LetRecord> &Result);

  bool ParseObjectBody(Record *CurRec);
  bool ParseBody(Record *CurRec);
  bool ParseBodyItem(Record *CurRec);

  bool ParseTemplateArgList(Record *CurRec);
  const Init *ParseDeclaration(Record *CurRec, bool ParsingTemplateArgs);
  const VarInit *ParseForeachDeclaration(const Init *&ForeachListValue);

  SubClassReference ParseSubClassReference(Record *CurRec, bool isDefm);
  SubMultiClassReference ParseSubMultiClassReference(MultiClass *CurMC);

  const Init *ParseIDValue(Record *CurRec, const StringInit *Name,
                           SMRange NameLoc, IDParseMode Mode = ParseValueMode);
  const Init *ParseSimpleValue(Record *CurRec, const RecTy *ItemType = nullptr,
                               IDParseMode Mode = ParseValueMode);
  const Init *ParseValue(Record *CurRec, const RecTy *ItemType = nullptr,
                         IDParseMode Mode = ParseValueMode);
  void ParseValueList(SmallVectorImpl<const Init *> &Result, Record *CurRec,
                      const RecTy *ItemType = nullptr);
  bool ParseTemplateArgValueList(SmallVectorImpl<const ArgumentInit *> &Result,
                                 SmallVectorImpl<SMLoc> &ArgLocs,
                                 Record *CurRec, const Record *ArgsRec);
  void ParseDagArgList(
      SmallVectorImpl<std::pair<const Init *, const StringInit *>> &Result,
      Record *CurRec);
  bool ParseOptionalRangeList(SmallVectorImpl<unsigned> &Ranges);
  bool ParseOptionalBitList(SmallVectorImpl<unsigned> &Ranges);
  const TypedInit *ParseSliceElement(Record *CurRec);
  const TypedInit *ParseSliceElements(Record *CurRec, bool Single = false);
  void ParseRangeList(SmallVectorImpl<unsigned> &Result);
  bool ParseRangePiece(SmallVectorImpl<unsigned> &Ranges,
                       const TypedInit *FirstItem = nullptr);
  const RecTy *ParseType();
  const Init *ParseOperation(Record *CurRec, const RecTy *ItemType);
  const Init *ParseOperationSubstr(Record *CurRec, const RecTy *ItemType);
  const Init *ParseOperationFind(Record *CurRec, const RecTy *ItemType);
  const Init *ParseOperationForEachFilter(Record *CurRec,
                                          const RecTy *ItemType);
  const Init *ParseOperationCond(Record *CurRec, const RecTy *ItemType);
  const RecTy *ParseOperatorType();
  const Init *ParseObjectName(MultiClass *CurMultiClass);
  const Record *ParseClassID();
  MultiClass *ParseMultiClassID();
  bool ApplyLetStack(Record *CurRec);
  bool ApplyLetStack(RecordsEntry &Entry);
  bool CheckTemplateArgValues(SmallVectorImpl<const ArgumentInit *> &Values,
                              ArrayRef<SMLoc> ValuesLocs,
                              const Record *ArgsRec);
};

} // end namespace llvm

#endif
