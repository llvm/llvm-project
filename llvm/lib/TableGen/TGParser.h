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
    StringInit *Name;
    std::vector<unsigned> Bits;
    Init *Value;
    SMLoc Loc;
    LetRecord(StringInit *N, ArrayRef<unsigned> B, Init *V, SMLoc L)
      : Name(N), Bits(B), Value(V), Loc(L) {
    }
  };

  /// RecordsEntry - Holds exactly one of a Record, ForeachLoop, or
  /// AssertionInfo.
  struct RecordsEntry {
    std::unique_ptr<Record> Rec;
    std::unique_ptr<ForeachLoop> Loop;
    std::unique_ptr<Record::AssertionInfo> Assertion;

    void dump() const;

    RecordsEntry() {}
    RecordsEntry(std::unique_ptr<Record> Rec) : Rec(std::move(Rec)) {}
    RecordsEntry(std::unique_ptr<ForeachLoop> Loop)
        : Loop(std::move(Loop)) {}
    RecordsEntry(std::unique_ptr<Record::AssertionInfo> Assertion)
        : Assertion(std::move(Assertion)) {}
  };

  /// ForeachLoop - Record the iteration state associated with a for loop.
  /// This is used to instantiate items in the loop body.
  ///
  /// IterVar is allowed to be null, in which case no iteration variable is
  /// defined in the loop at all. (This happens when a ForeachLoop is
  /// constructed by desugaring an if statement.)
  struct ForeachLoop {
    SMLoc Loc;
    VarInit *IterVar;
    Init *ListValue;
    std::vector<RecordsEntry> Entries;

    void dump() const;

    ForeachLoop(SMLoc Loc, VarInit *IVar, Init *LValue)
      : Loc(Loc), IterVar(IVar), ListValue(LValue) {}
  };

  struct DefsetRecord {
    SMLoc Loc;
    RecTy *EltTy = nullptr;
    SmallVector<Init *, 16> Elements;
  };

class TGLocalVarScope {
  // A scope to hold local variable definitions from defvar.
  std::map<std::string, Init *, std::less<>> vars;
  std::unique_ptr<TGLocalVarScope> parent;

public:
  TGLocalVarScope() = default;
  TGLocalVarScope(std::unique_ptr<TGLocalVarScope> parent)
      : parent(std::move(parent)) {}

  std::unique_ptr<TGLocalVarScope> extractParent() {
    // This is expected to be called just before we are destructed, so
    // it doesn't much matter what state we leave 'parent' in.
    return std::move(parent);
  }

  Init *getVar(StringRef Name) const {
    auto It = vars.find(Name);
    if (It != vars.end())
      return It->second;
    if (parent)
      return parent->getVar(Name);
    return nullptr;
  }

  bool varAlreadyDefined(StringRef Name) const {
    // When we check whether a variable is already defined, for the purpose of
    // reporting an error on redefinition, we don't look up to the parent
    // scope, because it's all right to shadow an outer definition with an
    // inner one.
    return vars.find(Name) != vars.end();
  }

  void addVar(StringRef Name, Init *I) {
    bool Ins = vars.insert(std::make_pair(std::string(Name), I)).second;
    (void)Ins;
    assert(Ins && "Local variable already exists");
  }
};

struct MultiClass {
  Record Rec;  // Placeholder for template args and Name.
  std::vector<RecordsEntry> Entries;

  void dump() const;

  MultiClass(StringRef Name, SMLoc Loc, RecordKeeper &Records) :
    Rec(Name, Loc, Records) {}
};

class TGParser {
  TGLexer Lex;
  std::vector<SmallVector<LetRecord, 4>> LetStack;
  std::map<std::string, std::unique_ptr<MultiClass>> MultiClasses;

  /// Loops - Keep track of any foreach loops we are within.
  ///
  std::vector<std::unique_ptr<ForeachLoop>> Loops;

  SmallVector<DefsetRecord *, 2> Defsets;

  /// CurMultiClass - If we are parsing a 'multiclass' definition, this is the
  /// current value.
  MultiClass *CurMultiClass;

  /// CurLocalScope - Innermost of the current nested scopes for 'defvar' local
  /// variables.
  std::unique_ptr<TGLocalVarScope> CurLocalScope;

  // Record tracker
  RecordKeeper &Records;

  // A "named boolean" indicating how to parse identifiers.  Usually
  // identifiers map to some existing object but in special cases
  // (e.g. parsing def names) no such object exists yet because we are
  // in the middle of creating in.  For those situations, allow the
  // parser to ignore missing object errors.
  enum IDParseMode {
    ParseValueMode,   // We are parsing a value we expect to look up.
    ParseNameMode,    // We are parsing a name of an object that does not yet
                      // exist.
  };

  bool NoWarnOnUnusedTemplateArgs = false;

public:
  TGParser(SourceMgr &SM, ArrayRef<std::string> Macros, RecordKeeper &records,
           const bool NoWarnOnUnusedTemplateArgs = false)
      : Lex(SM, Macros), CurMultiClass(nullptr), Records(records),
        NoWarnOnUnusedTemplateArgs(NoWarnOnUnusedTemplateArgs) {}

  /// ParseFile - Main entrypoint for parsing a tblgen file.  These parser
  /// routines return true on error, or false on success.
  bool ParseFile();

  bool Error(SMLoc L, const Twine &Msg) const {
    PrintError(L, Msg);
    return true;
  }
  bool TokError(const Twine &Msg) const {
    return Error(Lex.getLoc(), Msg);
  }
  const TGLexer::DependenciesSetTy &getDependencies() const {
    return Lex.getDependencies();
  }

  TGLocalVarScope *PushLocalScope() {
    CurLocalScope = std::make_unique<TGLocalVarScope>(std::move(CurLocalScope));
    // Returns a pointer to the new scope, so that the caller can pass it back
    // to PopLocalScope which will check by assertion that the pushes and pops
    // match up properly.
    return CurLocalScope.get();
  }
  void PopLocalScope(TGLocalVarScope *ExpectedStackTop) {
    assert(ExpectedStackTop == CurLocalScope.get() &&
           "Mismatched pushes and pops of local variable scopes");
    CurLocalScope = CurLocalScope->extractParent();
  }

private: // Semantic analysis methods.
  bool AddValue(Record *TheRec, SMLoc Loc, const RecordVal &RV);
  bool SetValue(Record *TheRec, SMLoc Loc, Init *ValName,
                ArrayRef<unsigned> BitList, Init *V,
                bool AllowSelfAssignment = false);
  bool AddSubClass(Record *Rec, SubClassReference &SubClass);
  bool AddSubClass(RecordsEntry &Entry, SubClassReference &SubClass);
  bool AddSubMultiClass(MultiClass *CurMC,
                        SubMultiClassReference &SubMultiClass);

  using SubstStack = SmallVector<std::pair<Init *, Init *>, 8>;

  bool addEntry(RecordsEntry E);
  bool resolve(const ForeachLoop &Loop, SubstStack &Stack, bool Final,
               std::vector<RecordsEntry> *Dest, SMLoc *Loc = nullptr);
  bool resolve(const std::vector<RecordsEntry> &Source, SubstStack &Substs,
               bool Final, std::vector<RecordsEntry> *Dest,
               SMLoc *Loc = nullptr);
  bool addDefOne(std::unique_ptr<Record> Rec);

private:  // Parser methods.
  bool consume(tgtok::TokKind K);
  bool ParseObjectList(MultiClass *MC = nullptr);
  bool ParseObject(MultiClass *MC);
  bool ParseClass();
  bool ParseMultiClass();
  bool ParseDefm(MultiClass *CurMultiClass);
  bool ParseDef(MultiClass *CurMultiClass);
  bool ParseDefset();
  bool ParseDefvar();
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
  Init *ParseDeclaration(Record *CurRec, bool ParsingTemplateArgs);
  VarInit *ParseForeachDeclaration(Init *&ForeachListValue);

  SubClassReference ParseSubClassReference(Record *CurRec, bool isDefm);
  SubMultiClassReference ParseSubMultiClassReference(MultiClass *CurMC);

  Init *ParseIDValue(Record *CurRec, StringInit *Name, SMLoc NameLoc,
                     IDParseMode Mode = ParseValueMode);
  Init *ParseSimpleValue(Record *CurRec, RecTy *ItemType = nullptr,
                         IDParseMode Mode = ParseValueMode);
  Init *ParseValue(Record *CurRec, RecTy *ItemType = nullptr,
                   IDParseMode Mode = ParseValueMode);
  void ParseValueList(SmallVectorImpl<llvm::Init*> &Result,
                      Record *CurRec, RecTy *ItemType = nullptr);
  bool ParseTemplateArgValueList(SmallVectorImpl<llvm::Init *> &Result,
                                 Record *CurRec, Record *ArgsRec);
  void ParseDagArgList(
      SmallVectorImpl<std::pair<llvm::Init*, StringInit*>> &Result,
      Record *CurRec);
  bool ParseOptionalRangeList(SmallVectorImpl<unsigned> &Ranges);
  bool ParseOptionalBitList(SmallVectorImpl<unsigned> &Ranges);
  void ParseRangeList(SmallVectorImpl<unsigned> &Result);
  bool ParseRangePiece(SmallVectorImpl<unsigned> &Ranges,
                       TypedInit *FirstItem = nullptr);
  RecTy *ParseType();
  Init *ParseOperation(Record *CurRec, RecTy *ItemType);
  Init *ParseOperationSubstr(Record *CurRec, RecTy *ItemType);
  Init *ParseOperationFind(Record *CurRec, RecTy *ItemType);
  Init *ParseOperationForEachFilter(Record *CurRec, RecTy *ItemType);
  Init *ParseOperationCond(Record *CurRec, RecTy *ItemType);
  RecTy *ParseOperatorType();
  Init *ParseObjectName(MultiClass *CurMultiClass);
  Record *ParseClassID();
  MultiClass *ParseMultiClassID();
  bool ApplyLetStack(Record *CurRec);
  bool ApplyLetStack(RecordsEntry &Entry);
  bool CheckTemplateArgValues(SmallVectorImpl<llvm::Init *> &Values,
                              SMLoc Loc, Record *ArgsRec);
};

} // end namespace llvm

#endif
