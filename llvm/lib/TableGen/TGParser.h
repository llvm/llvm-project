//===- TGParser.h - Parser for TableGen Files -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Parser for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef TGPARSER_H
#define TGPARSER_H

#include "llvm/TableGen/Record.h"
#include "TGLexer.h"
#include "llvm/TableGen/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include <map>

namespace llvm {
  class Record;
  class RecordVal;
  class RecordKeeper;
  class RecTy;
  class Init;
  struct MultiClass;
  struct SubClassReference;
  struct SubMultiClassReference;

  struct LetRecord {
    std::string Name;
    std::vector<unsigned> Bits;
    Init *Value;
    SMLoc Loc;
    LetRecord(const std::string &N, const std::vector<unsigned> &B, Init *V,
              SMLoc L)
      : Name(N), Bits(B), Value(V), Loc(L) {
    }
  };

  /// ForeachLoop - Record the iteration state associated with a for loop.
  /// This is used to instantiate items in the loop body.
  struct ForeachLoop {
    VarInit *IterVar;
    ListInit *ListValue;

    ForeachLoop(VarInit *IVar, ListInit *LValue)
      : IterVar(IVar), ListValue(LValue) {}
  };

class TGParser {
  TGLexer Lex;
  std::vector<std::vector<LetRecord> > LetStack;
  std::map<std::string, MultiClass*> MultiClasses;

  /// Loops - Keep track of any foreach loops we are within.
  ///
  typedef std::vector<ForeachLoop> LoopVector;
  LoopVector Loops;

  /// CurMultiClass - If we are parsing a 'multiclass' definition, this is the
  /// current value.
  MultiClass *CurMultiClass;

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
    ParseForeachMode  // We are parsing a foreach init.
  };

public:
  TGParser(SourceMgr &SrcMgr, RecordKeeper &records) :
    Lex(SrcMgr), CurMultiClass(0), Records(records) {}

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
  const std::vector<std::string> &getDependencies() const {
    return Lex.getDependencies();
  }

private:  // Semantic analysis methods.
  bool AddValue(Record *TheRec, SMLoc Loc, const RecordVal &RV);
  bool SetValue(Record *TheRec, SMLoc Loc, Init *ValName,
                const std::vector<unsigned> &BitList, Init *V);
  bool SetValue(Record *TheRec, SMLoc Loc, const std::string &ValName,
                const std::vector<unsigned> &BitList, Init *V) {
    return SetValue(TheRec, Loc, StringInit::get(ValName), BitList, V);
  }
  bool AddSubClass(Record *Rec, SubClassReference &SubClass);
  bool AddSubMultiClass(MultiClass *CurMC,
                        SubMultiClassReference &SubMultiClass);

  // IterRecord: Map an iterator name to a value.
  struct IterRecord {
    VarInit *IterVar;
    Init *IterValue;
    IterRecord(VarInit *Var, Init *Val) : IterVar(Var), IterValue(Val) {}
  };

  // IterSet: The set of all iterator values at some point in the
  // iteration space.
  typedef std::vector<IterRecord> IterSet;

  bool ProcessForeachDefs(Record *CurRec, SMLoc Loc);
  bool ProcessForeachDefs(Record *CurRec, SMLoc Loc, IterSet &IterVals);

private:  // Parser methods.
  bool ParseObjectList(MultiClass *MC = 0);
  bool ParseObject(MultiClass *MC);
  bool ParseClass();
  bool ParseMultiClass();
  Record *InstantiateMulticlassDef(MultiClass &MC,
                                   Record *DefProto,
                                   Init *DefmPrefix,
                                   SMLoc DefmPrefixLoc);
  bool ResolveMulticlassDefArgs(MultiClass &MC,
                                Record *DefProto,
                                SMLoc DefmPrefixLoc,
                                SMLoc SubClassLoc,
                                const std::vector<Init *> &TArgs,
                                std::vector<Init *> &TemplateVals,
                                bool DeleteArgs);
  bool ResolveMulticlassDef(MultiClass &MC,
                            Record *CurRec,
                            Record *DefProto,
                            SMLoc DefmPrefixLoc);
  bool ParseDefm(MultiClass *CurMultiClass);
  bool ParseDef(MultiClass *CurMultiClass);
  bool ParseForeach(MultiClass *CurMultiClass);
  bool ParseTopLevelLet(MultiClass *CurMultiClass);
  std::vector<LetRecord> ParseLetList();

  bool ParseObjectBody(Record *CurRec);
  bool ParseBody(Record *CurRec);
  bool ParseBodyItem(Record *CurRec);

  bool ParseTemplateArgList(Record *CurRec);
  Init *ParseDeclaration(Record *CurRec, bool ParsingTemplateArgs);
  VarInit *ParseForeachDeclaration(ListInit *&ForeachListValue);

  SubClassReference ParseSubClassReference(Record *CurRec, bool isDefm);
  SubMultiClassReference ParseSubMultiClassReference(MultiClass *CurMC);

  Init *ParseIDValue(Record *CurRec, IDParseMode Mode = ParseValueMode);
  Init *ParseIDValue(Record *CurRec, const std::string &Name, SMLoc NameLoc,
                     IDParseMode Mode = ParseValueMode);
  Init *ParseSimpleValue(Record *CurRec, RecTy *ItemType = 0,
                         IDParseMode Mode = ParseValueMode);
  Init *ParseValue(Record *CurRec, RecTy *ItemType = 0,
                   IDParseMode Mode = ParseValueMode);
  std::vector<Init*> ParseValueList(Record *CurRec, Record *ArgsRec = 0,
                                    RecTy *EltTy = 0);
  std::vector<std::pair<llvm::Init*, std::string> > ParseDagArgList(Record *);
  bool ParseOptionalRangeList(std::vector<unsigned> &Ranges);
  bool ParseOptionalBitList(std::vector<unsigned> &Ranges);
  std::vector<unsigned> ParseRangeList();
  bool ParseRangePiece(std::vector<unsigned> &Ranges);
  RecTy *ParseType();
  Init *ParseOperation(Record *CurRec);
  RecTy *ParseOperatorType();
  Init *ParseObjectName(MultiClass *CurMultiClass);
  Record *ParseClassID();
  MultiClass *ParseMultiClassID();
  Record *ParseDefmID();
};

} // end namespace llvm

#endif
