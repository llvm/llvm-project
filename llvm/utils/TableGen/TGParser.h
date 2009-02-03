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

#include "TGLexer.h"
#include <map>

namespace llvm {
  class Record;
  class RecordVal;
  struct RecTy;
  struct Init;
  struct MultiClass;
  struct SubClassReference;
  
  struct LetRecord {
    std::string Name;
    std::vector<unsigned> Bits;
    Init *Value;
    TGLexer::LocTy Loc;
    LetRecord(const std::string &N, const std::vector<unsigned> &B, Init *V,
              TGLexer::LocTy L)
      : Name(N), Bits(B), Value(V), Loc(L) {
    }
  };
  
class TGParser {
  TGLexer Lex;
  std::vector<std::vector<LetRecord> > LetStack;
  std::map<std::string, MultiClass*> MultiClasses;
  
  /// CurMultiClass - If we are parsing a 'multiclass' definition, this is the 
  /// current value.
  MultiClass *CurMultiClass;
public:
  typedef TGLexer::LocTy LocTy;
  
  TGParser(MemoryBuffer *StartBuf) : Lex(StartBuf), CurMultiClass(0) {}
  
  void setIncludeDirs(const std::vector<std::string> &D){Lex.setIncludeDirs(D);}

  /// ParseFile - Main entrypoint for parsing a tblgen file.  These parser
  /// routines return true on error, or false on success.
  bool ParseFile();
  
  bool Error(LocTy L, const std::string &Msg) const {
    Lex.PrintError(L, Msg);
    return true;
  }
  bool TokError(const std::string &Msg) const {
    return Error(Lex.getLoc(), Msg);
  }
private:  // Semantic analysis methods.
  bool AddValue(Record *TheRec, LocTy Loc, const RecordVal &RV);
  bool SetValue(Record *TheRec, LocTy Loc, const std::string &ValName, 
                const std::vector<unsigned> &BitList, Init *V);
  bool AddSubClass(Record *Rec, class SubClassReference &SubClass);

private:  // Parser methods.
  bool ParseObjectList();
  bool ParseObject();
  bool ParseClass();
  bool ParseMultiClass();
  bool ParseMultiClassDef(MultiClass *CurMC);
  bool ParseDefm();
  bool ParseTopLevelLet();
  std::vector<LetRecord> ParseLetList();

  Record *ParseDef(MultiClass *CurMultiClass);
  bool ParseObjectBody(Record *CurRec);
  bool ParseBody(Record *CurRec);
  bool ParseBodyItem(Record *CurRec);

  bool ParseTemplateArgList(Record *CurRec);
  std::string ParseDeclaration(Record *CurRec, bool ParsingTemplateArgs);

  SubClassReference ParseSubClassReference(Record *CurRec, bool isDefm);

  Init *ParseIDValue(Record *CurRec);
  Init *ParseIDValue(Record *CurRec, const std::string &Name, LocTy NameLoc);
  Init *ParseSimpleValue(Record *CurRec);
  Init *ParseValue(Record *CurRec);
  std::vector<Init*> ParseValueList(Record *CurRec);
  std::vector<std::pair<llvm::Init*, std::string> > ParseDagArgList(Record *);
  bool ParseOptionalRangeList(std::vector<unsigned> &Ranges);
  bool ParseOptionalBitList(std::vector<unsigned> &Ranges);
  std::vector<unsigned> ParseRangeList();
  bool ParseRangePiece(std::vector<unsigned> &Ranges);
  RecTy *ParseType();
  std::string ParseObjectName();
  Record *ParseClassID();
  Record *ParseDefmID();
};
  
} // end namespace llvm

#endif
