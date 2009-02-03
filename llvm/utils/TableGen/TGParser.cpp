//===- TGParser.cpp - Parser for TableGen Files ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Parser for TableGen.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "TGParser.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Support Code for the Semantic Actions.
//===----------------------------------------------------------------------===//

namespace llvm {
struct MultiClass {
  Record Rec;  // Placeholder for template args and Name.
  std::vector<Record*> DefPrototypes;
    
  MultiClass(const std::string &Name) : Rec(Name) {}
};
  
struct SubClassReference {
  TGParser::LocTy RefLoc;
  Record *Rec;
  std::vector<Init*> TemplateArgs;
  SubClassReference() : RefLoc(0), Rec(0) {}
  
  bool isInvalid() const { return Rec == 0; }
};
  
} // end namespace llvm

bool TGParser::AddValue(Record *CurRec, LocTy Loc, const RecordVal &RV) {
  if (CurRec == 0)
    CurRec = &CurMultiClass->Rec;
  
  if (RecordVal *ERV = CurRec->getValue(RV.getName())) {
    // The value already exists in the class, treat this as a set.
    if (ERV->setValue(RV.getValue()))
      return Error(Loc, "New definition of '" + RV.getName() + "' of type '" +
                   RV.getType()->getAsString() + "' is incompatible with " +
                   "previous definition of type '" + 
                   ERV->getType()->getAsString() + "'");
  } else {
    CurRec->addValue(RV);
  }
  return false;
}

/// SetValue -
/// Return true on error, false on success.
bool TGParser::SetValue(Record *CurRec, LocTy Loc, const std::string &ValName, 
                        const std::vector<unsigned> &BitList, Init *V) {
  if (!V) return false;

  if (CurRec == 0) CurRec = &CurMultiClass->Rec;

  RecordVal *RV = CurRec->getValue(ValName);
  if (RV == 0)
    return Error(Loc, "Value '" + ValName + "' unknown!");

  // Do not allow assignments like 'X = X'.  This will just cause infinite loops
  // in the resolution machinery.
  if (BitList.empty())
    if (VarInit *VI = dynamic_cast<VarInit*>(V))
      if (VI->getName() == ValName)
        return false;
  
  // If we are assigning to a subset of the bits in the value... then we must be
  // assigning to a field of BitsRecTy, which must have a BitsInit
  // initializer.
  //
  if (!BitList.empty()) {
    BitsInit *CurVal = dynamic_cast<BitsInit*>(RV->getValue());
    if (CurVal == 0)
      return Error(Loc, "Value '" + ValName + "' is not a bits type");

    // Convert the incoming value to a bits type of the appropriate size...
    Init *BI = V->convertInitializerTo(new BitsRecTy(BitList.size()));
    if (BI == 0) {
      V->convertInitializerTo(new BitsRecTy(BitList.size()));
      return Error(Loc, "Initializer is not compatible with bit range");
    }
                   
    // We should have a BitsInit type now.
    BitsInit *BInit = dynamic_cast<BitsInit*>(BI);
    assert(BInit != 0);

    BitsInit *NewVal = new BitsInit(CurVal->getNumBits());

    // Loop over bits, assigning values as appropriate.
    for (unsigned i = 0, e = BitList.size(); i != e; ++i) {
      unsigned Bit = BitList[i];
      if (NewVal->getBit(Bit))
        return Error(Loc, "Cannot set bit #" + utostr(Bit) + " of value '" +
                     ValName + "' more than once");
      NewVal->setBit(Bit, BInit->getBit(i));
    }

    for (unsigned i = 0, e = CurVal->getNumBits(); i != e; ++i)
      if (NewVal->getBit(i) == 0)
        NewVal->setBit(i, CurVal->getBit(i));

    V = NewVal;
  }

  if (RV->setValue(V))
   return Error(Loc, "Value '" + ValName + "' of type '" + 
                RV->getType()->getAsString() + 
                "' is incompatible with initializer '" + V->getAsString() +"'");
  return false;
}

/// AddSubClass - Add SubClass as a subclass to CurRec, resolving its template
/// args as SubClass's template arguments.
bool TGParser::AddSubClass(Record *CurRec, class SubClassReference &SubClass) {
  Record *SC = SubClass.Rec;
  // Add all of the values in the subclass into the current class.
  const std::vector<RecordVal> &Vals = SC->getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (AddValue(CurRec, SubClass.RefLoc, Vals[i]))
      return true;

  const std::vector<std::string> &TArgs = SC->getTemplateArgs();

  // Ensure that an appropriate number of template arguments are specified.
  if (TArgs.size() < SubClass.TemplateArgs.size())
    return Error(SubClass.RefLoc, "More template args specified than expected");
  
  // Loop over all of the template arguments, setting them to the specified
  // value or leaving them as the default if necessary.
  for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
    if (i < SubClass.TemplateArgs.size()) {
      // If a value is specified for this template arg, set it now.
      if (SetValue(CurRec, SubClass.RefLoc, TArgs[i], std::vector<unsigned>(), 
                   SubClass.TemplateArgs[i]))
        return true;
      
      // Resolve it next.
      CurRec->resolveReferencesTo(CurRec->getValue(TArgs[i]));
      
      // Now remove it.
      CurRec->removeValue(TArgs[i]);

    } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
      return Error(SubClass.RefLoc,"Value not specified for template argument #"
                   + utostr(i) + " (" + TArgs[i] + ") of subclass '" + 
                   SC->getName() + "'!");
    }
  }

  // Since everything went well, we can now set the "superclass" list for the
  // current record.
  const std::vector<Record*> &SCs = SC->getSuperClasses();
  for (unsigned i = 0, e = SCs.size(); i != e; ++i) {
    if (CurRec->isSubClassOf(SCs[i]))
      return Error(SubClass.RefLoc,
                   "Already subclass of '" + SCs[i]->getName() + "'!\n");
    CurRec->addSuperClass(SCs[i]);
  }
  
  if (CurRec->isSubClassOf(SC))
    return Error(SubClass.RefLoc,
                 "Already subclass of '" + SC->getName() + "'!\n");
  CurRec->addSuperClass(SC);
  return false;
}

//===----------------------------------------------------------------------===//
// Parser Code
//===----------------------------------------------------------------------===//

/// isObjectStart - Return true if this is a valid first token for an Object.
static bool isObjectStart(tgtok::TokKind K) {
  return K == tgtok::Class || K == tgtok::Def ||
         K == tgtok::Defm || K == tgtok::Let || K == tgtok::MultiClass; 
}

/// ParseObjectName - If an object name is specified, return it.  Otherwise,
/// return an anonymous name.
///   ObjectName ::= ID
///   ObjectName ::= /*empty*/
///
std::string TGParser::ParseObjectName() {
  if (Lex.getCode() == tgtok::Id) {
    std::string Ret = Lex.getCurStrVal();
    Lex.Lex();
    return Ret;
  }
  
  static unsigned AnonCounter = 0;
  return "anonymous."+utostr(AnonCounter++);
}


/// ParseClassID - Parse and resolve a reference to a class name.  This returns
/// null on error.
///
///    ClassID ::= ID
///
Record *TGParser::ParseClassID() {
  if (Lex.getCode() != tgtok::Id) {
    TokError("expected name for ClassID");
    return 0;
  }
  
  Record *Result = Records.getClass(Lex.getCurStrVal());
  if (Result == 0)
    TokError("Couldn't find class '" + Lex.getCurStrVal() + "'");
  
  Lex.Lex();
  return Result;
}

Record *TGParser::ParseDefmID() {
  if (Lex.getCode() != tgtok::Id) {
    TokError("expected multiclass name");
    return 0;
  }
  
  MultiClass *MC = MultiClasses[Lex.getCurStrVal()];
  if (MC == 0) {
    TokError("Couldn't find multiclass '" + Lex.getCurStrVal() + "'");
    return 0;
  }
  
  Lex.Lex();
  return &MC->Rec;
}  



/// ParseSubClassReference - Parse a reference to a subclass or to a templated
/// subclass.  This returns a SubClassRefTy with a null Record* on error.
///
///  SubClassRef ::= ClassID
///  SubClassRef ::= ClassID '<' ValueList '>'
///
SubClassReference TGParser::
ParseSubClassReference(Record *CurRec, bool isDefm) {
  SubClassReference Result;
  Result.RefLoc = Lex.getLoc();
  
  if (isDefm)
    Result.Rec = ParseDefmID();
  else
    Result.Rec = ParseClassID();
  if (Result.Rec == 0) return Result;
  
  // If there is no template arg list, we're done.
  if (Lex.getCode() != tgtok::less)
    return Result;
  Lex.Lex();  // Eat the '<'
  
  if (Lex.getCode() == tgtok::greater) {
    TokError("subclass reference requires a non-empty list of template values");
    Result.Rec = 0;
    return Result;
  }
  
  Result.TemplateArgs = ParseValueList(CurRec);
  if (Result.TemplateArgs.empty()) {
    Result.Rec = 0;   // Error parsing value list.
    return Result;
  }
    
  if (Lex.getCode() != tgtok::greater) {
    TokError("expected '>' in template value list");
    Result.Rec = 0;
    return Result;
  }
  Lex.Lex();
  
  return Result;
}

/// ParseRangePiece - Parse a bit/value range.
///   RangePiece ::= INTVAL
///   RangePiece ::= INTVAL '-' INTVAL
///   RangePiece ::= INTVAL INTVAL
bool TGParser::ParseRangePiece(std::vector<unsigned> &Ranges) {
  if (Lex.getCode() != tgtok::IntVal) {
    TokError("expected integer or bitrange");
    return true;
  }
  int64_t Start = Lex.getCurIntVal();
  int64_t End;
  
  if (Start < 0)
    return TokError("invalid range, cannot be negative");
  
  switch (Lex.Lex()) {  // eat first character.
  default: 
    Ranges.push_back(Start);
    return false;
  case tgtok::minus:
    if (Lex.Lex() != tgtok::IntVal) {
      TokError("expected integer value as end of range");
      return true;
    }
    End = Lex.getCurIntVal();
    break;
  case tgtok::IntVal:
    End = -Lex.getCurIntVal();
    break;
  }
  if (End < 0) 
    return TokError("invalid range, cannot be negative");
  Lex.Lex();
  
  // Add to the range.
  if (Start < End) {
    for (; Start <= End; ++Start)
      Ranges.push_back(Start);
  } else {
    for (; Start >= End; --Start)
      Ranges.push_back(Start);
  }
  return false;
}

/// ParseRangeList - Parse a list of scalars and ranges into scalar values.
///
///   RangeList ::= RangePiece (',' RangePiece)*
///
std::vector<unsigned> TGParser::ParseRangeList() {
  std::vector<unsigned> Result;
  
  // Parse the first piece.
  if (ParseRangePiece(Result))
    return std::vector<unsigned>();
  while (Lex.getCode() == tgtok::comma) {
    Lex.Lex();  // Eat the comma.

    // Parse the next range piece.
    if (ParseRangePiece(Result))
      return std::vector<unsigned>();
  }
  return Result;
}

/// ParseOptionalRangeList - Parse either a range list in <>'s or nothing.
///   OptionalRangeList ::= '<' RangeList '>'
///   OptionalRangeList ::= /*empty*/
bool TGParser::ParseOptionalRangeList(std::vector<unsigned> &Ranges) {
  if (Lex.getCode() != tgtok::less)
    return false;
  
  LocTy StartLoc = Lex.getLoc();
  Lex.Lex(); // eat the '<'
  
  // Parse the range list.
  Ranges = ParseRangeList();
  if (Ranges.empty()) return true;
  
  if (Lex.getCode() != tgtok::greater) {
    TokError("expected '>' at end of range list");
    return Error(StartLoc, "to match this '<'");
  }
  Lex.Lex();   // eat the '>'.
  return false;
}

/// ParseOptionalBitList - Parse either a bit list in {}'s or nothing.
///   OptionalBitList ::= '{' RangeList '}'
///   OptionalBitList ::= /*empty*/
bool TGParser::ParseOptionalBitList(std::vector<unsigned> &Ranges) {
  if (Lex.getCode() != tgtok::l_brace)
    return false;
  
  LocTy StartLoc = Lex.getLoc();
  Lex.Lex(); // eat the '{'
  
  // Parse the range list.
  Ranges = ParseRangeList();
  if (Ranges.empty()) return true;
  
  if (Lex.getCode() != tgtok::r_brace) {
    TokError("expected '}' at end of bit list");
    return Error(StartLoc, "to match this '{'");
  }
  Lex.Lex();   // eat the '}'.
  return false;
}


/// ParseType - Parse and return a tblgen type.  This returns null on error.
///
///   Type ::= STRING                       // string type
///   Type ::= BIT                          // bit type
///   Type ::= BITS '<' INTVAL '>'          // bits<x> type
///   Type ::= INT                          // int type
///   Type ::= LIST '<' Type '>'            // list<x> type
///   Type ::= CODE                         // code type
///   Type ::= DAG                          // dag type
///   Type ::= ClassID                      // Record Type
///
RecTy *TGParser::ParseType() {
  switch (Lex.getCode()) {
  default: TokError("Unknown token when expecting a type"); return 0;
  case tgtok::String: Lex.Lex(); return new StringRecTy();
  case tgtok::Bit:    Lex.Lex(); return new BitRecTy();
  case tgtok::Int:    Lex.Lex(); return new IntRecTy();
  case tgtok::Code:   Lex.Lex(); return new CodeRecTy();
  case tgtok::Dag:    Lex.Lex(); return new DagRecTy();
  case tgtok::Id:
    if (Record *R = ParseClassID()) return new RecordRecTy(R);
    return 0;
  case tgtok::Bits: {
    if (Lex.Lex() != tgtok::less) { // Eat 'bits'
      TokError("expected '<' after bits type");
      return 0;
    }
    if (Lex.Lex() != tgtok::IntVal) {  // Eat '<'
      TokError("expected integer in bits<n> type");
      return 0;
    }
    uint64_t Val = Lex.getCurIntVal();
    if (Lex.Lex() != tgtok::greater) {  // Eat count.
      TokError("expected '>' at end of bits<n> type");
      return 0;
    }
    Lex.Lex();  // Eat '>'
    return new BitsRecTy(Val);
  }
  case tgtok::List: {
    if (Lex.Lex() != tgtok::less) { // Eat 'bits'
      TokError("expected '<' after list type");
      return 0;
    }
    Lex.Lex();  // Eat '<'
    RecTy *SubType = ParseType();
    if (SubType == 0) return 0;
    
    if (Lex.getCode() != tgtok::greater) {
      TokError("expected '>' at end of list<ty> type");
      return 0;
    }
    Lex.Lex();  // Eat '>'
    return new ListRecTy(SubType);
  }
  }      
}

/// ParseIDValue - Parse an ID as a value and decode what it means.
///
///  IDValue ::= ID [def local value]
///  IDValue ::= ID [def template arg]
///  IDValue ::= ID [multiclass local value]
///  IDValue ::= ID [multiclass template argument]
///  IDValue ::= ID [def name]
///
Init *TGParser::ParseIDValue(Record *CurRec) {
  assert(Lex.getCode() == tgtok::Id && "Expected ID in ParseIDValue");
  std::string Name = Lex.getCurStrVal();
  LocTy Loc = Lex.getLoc();
  Lex.Lex();
  return ParseIDValue(CurRec, Name, Loc);
}

/// ParseIDValue - This is just like ParseIDValue above, but it assumes the ID
/// has already been read.
Init *TGParser::ParseIDValue(Record *CurRec, 
                             const std::string &Name, LocTy NameLoc) {
  if (CurRec) {
    if (const RecordVal *RV = CurRec->getValue(Name))
      return new VarInit(Name, RV->getType());
    
    std::string TemplateArgName = CurRec->getName()+":"+Name;
    if (CurRec->isTemplateArg(TemplateArgName)) {
      const RecordVal *RV = CurRec->getValue(TemplateArgName);
      assert(RV && "Template arg doesn't exist??");
      return new VarInit(TemplateArgName, RV->getType());
    }
  }
  
  if (CurMultiClass) {
    std::string MCName = CurMultiClass->Rec.getName()+"::"+Name;
    if (CurMultiClass->Rec.isTemplateArg(MCName)) {
      const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
      assert(RV && "Template arg doesn't exist??");
      return new VarInit(MCName, RV->getType());
    }
  }
  
  if (Record *D = Records.getDef(Name))
    return new DefInit(D);

  Error(NameLoc, "Variable not defined: '" + Name + "'");
  return 0;
}

/// ParseSimpleValue - Parse a tblgen value.  This returns null on error.
///
///   SimpleValue ::= IDValue
///   SimpleValue ::= INTVAL
///   SimpleValue ::= STRVAL
///   SimpleValue ::= CODEFRAGMENT
///   SimpleValue ::= '?'
///   SimpleValue ::= '{' ValueList '}'
///   SimpleValue ::= ID '<' ValueListNE '>'
///   SimpleValue ::= '[' ValueList ']'
///   SimpleValue ::= '(' IDValue DagArgList ')'
///   SimpleValue ::= CONCATTOK '(' Value ',' Value ')'
///   SimpleValue ::= SHLTOK '(' Value ',' Value ')'
///   SimpleValue ::= SRATOK '(' Value ',' Value ')'
///   SimpleValue ::= SRLTOK '(' Value ',' Value ')'
///   SimpleValue ::= STRCONCATTOK '(' Value ',' Value ')'
///
Init *TGParser::ParseSimpleValue(Record *CurRec) {
  Init *R = 0;
  switch (Lex.getCode()) {
  default: TokError("Unknown token when parsing a value"); break;
  case tgtok::IntVal: R = new IntInit(Lex.getCurIntVal()); Lex.Lex(); break;
  case tgtok::StrVal: R = new StringInit(Lex.getCurStrVal()); Lex.Lex(); break;
  case tgtok::CodeFragment:
      R = new CodeInit(Lex.getCurStrVal()); Lex.Lex(); break;
  case tgtok::question: R = new UnsetInit(); Lex.Lex(); break;
  case tgtok::Id: {
    LocTy NameLoc = Lex.getLoc();
    std::string Name = Lex.getCurStrVal();
    if (Lex.Lex() != tgtok::less)  // consume the Id.
      return ParseIDValue(CurRec, Name, NameLoc);    // Value ::= IDValue
    
    // Value ::= ID '<' ValueListNE '>'
    if (Lex.Lex() == tgtok::greater) {
      TokError("expected non-empty value list");
      return 0;
    }
    std::vector<Init*> ValueList = ParseValueList(CurRec);
    if (ValueList.empty()) return 0;
    
    if (Lex.getCode() != tgtok::greater) {
      TokError("expected '>' at end of value list");
      return 0;
    }
    Lex.Lex();  // eat the '>'
    
    // This is a CLASS<initvalslist> expression.  This is supposed to synthesize
    // a new anonymous definition, deriving from CLASS<initvalslist> with no
    // body.
    Record *Class = Records.getClass(Name);
    if (!Class) {
      Error(NameLoc, "Expected a class name, got '" + Name + "'");
      return 0;
    }
    
    // Create the new record, set it as CurRec temporarily.
    static unsigned AnonCounter = 0;
    Record *NewRec = new Record("anonymous.val."+utostr(AnonCounter++));
    SubClassReference SCRef;
    SCRef.RefLoc = NameLoc;
    SCRef.Rec = Class;
    SCRef.TemplateArgs = ValueList;
    // Add info about the subclass to NewRec.
    if (AddSubClass(NewRec, SCRef))
      return 0;
    NewRec->resolveReferences();
    Records.addDef(NewRec);
    
    // The result of the expression is a reference to the new record.
    return new DefInit(NewRec);
  }    
  case tgtok::l_brace: {           // Value ::= '{' ValueList '}'
    LocTy BraceLoc = Lex.getLoc();
    Lex.Lex(); // eat the '{'
    std::vector<Init*> Vals;
    
    if (Lex.getCode() != tgtok::r_brace) {
      Vals = ParseValueList(CurRec);
      if (Vals.empty()) return 0;
    }
    if (Lex.getCode() != tgtok::r_brace) {
      TokError("expected '}' at end of bit list value");
      return 0;
    }
    Lex.Lex();  // eat the '}'
    
    BitsInit *Result = new BitsInit(Vals.size());
    for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
      Init *Bit = Vals[i]->convertInitializerTo(new BitRecTy());
      if (Bit == 0) {
        Error(BraceLoc, "Element #" + utostr(i) + " (" + Vals[i]->getAsString()+
              ") is not convertable to a bit");
        return 0;
      }
      Result->setBit(Vals.size()-i-1, Bit);
    }
    return Result;
  }
  case tgtok::l_square: {          // Value ::= '[' ValueList ']'
    Lex.Lex(); // eat the '['
    std::vector<Init*> Vals;
    
    if (Lex.getCode() != tgtok::r_square) {
      Vals = ParseValueList(CurRec);
      if (Vals.empty()) return 0;
    }
    if (Lex.getCode() != tgtok::r_square) {
      TokError("expected ']' at end of list value");
      return 0;
    }
    Lex.Lex();  // eat the ']'
    return new ListInit(Vals);
  }
  case tgtok::l_paren: {         // Value ::= '(' IDValue DagArgList ')'
    Lex.Lex();   // eat the '('
    if (Lex.getCode() != tgtok::Id) {
      TokError("expected identifier in dag init");
      return 0;
    }
    
    Init *Operator = ParseIDValue(CurRec);
    if (Operator == 0) return 0;
    
    std::vector<std::pair<llvm::Init*, std::string> > DagArgs;
    if (Lex.getCode() != tgtok::r_paren) {
      DagArgs = ParseDagArgList(CurRec);
      if (DagArgs.empty()) return 0;
    }
    
    if (Lex.getCode() != tgtok::r_paren) {
      TokError("expected ')' in dag init");
      return 0;
    }
    Lex.Lex();  // eat the ')'
    
    return new DagInit(Operator, DagArgs);
  }
  case tgtok::XConcat:
  case tgtok::XSRA: 
  case tgtok::XSRL:
  case tgtok::XSHL:
  case tgtok::XStrConcat: {  // Value ::= !binop '(' Value ',' Value ')'
    BinOpInit::BinaryOp Code;
    switch (Lex.getCode()) {
    default: assert(0 && "Unhandled code!");
    case tgtok::XConcat:    Code = BinOpInit::CONCAT; break;
    case tgtok::XSRA:       Code = BinOpInit::SRA; break;
    case tgtok::XSRL:       Code = BinOpInit::SRL; break;
    case tgtok::XSHL:       Code = BinOpInit::SHL; break;
    case tgtok::XStrConcat: Code = BinOpInit::STRCONCAT; break;
    }
    Lex.Lex();  // eat the operation
    if (Lex.getCode() != tgtok::l_paren) {
      TokError("expected '(' after binary operator");
      return 0;
    }
    Lex.Lex();  // eat the '('
    
    Init *LHS = ParseValue(CurRec);
    if (LHS == 0) return 0;

    if (Lex.getCode() != tgtok::comma) {
      TokError("expected ',' in binary operator");
      return 0;
    }
    Lex.Lex();  // eat the ','
    
    Init *RHS = ParseValue(CurRec);
    if (RHS == 0) return 0;

    if (Lex.getCode() != tgtok::r_paren) {
      TokError("expected ')' in binary operator");
      return 0;
    }
    Lex.Lex();  // eat the ')'
    return (new BinOpInit(Code, LHS, RHS))->Fold();
  }
  }
  
  return R;
}

/// ParseValue - Parse a tblgen value.  This returns null on error.
///
///   Value       ::= SimpleValue ValueSuffix*
///   ValueSuffix ::= '{' BitList '}'
///   ValueSuffix ::= '[' BitList ']'
///   ValueSuffix ::= '.' ID
///
Init *TGParser::ParseValue(Record *CurRec) {
  Init *Result = ParseSimpleValue(CurRec);
  if (Result == 0) return 0;
  
  // Parse the suffixes now if present.
  while (1) {
    switch (Lex.getCode()) {
    default: return Result;
    case tgtok::l_brace: {
      LocTy CurlyLoc = Lex.getLoc();
      Lex.Lex(); // eat the '{'
      std::vector<unsigned> Ranges = ParseRangeList();
      if (Ranges.empty()) return 0;
      
      // Reverse the bitlist.
      std::reverse(Ranges.begin(), Ranges.end());
      Result = Result->convertInitializerBitRange(Ranges);
      if (Result == 0) {
        Error(CurlyLoc, "Invalid bit range for value");
        return 0;
      }
      
      // Eat the '}'.
      if (Lex.getCode() != tgtok::r_brace) {
        TokError("expected '}' at end of bit range list");
        return 0;
      }
      Lex.Lex();
      break;
    }
    case tgtok::l_square: {
      LocTy SquareLoc = Lex.getLoc();
      Lex.Lex(); // eat the '['
      std::vector<unsigned> Ranges = ParseRangeList();
      if (Ranges.empty()) return 0;
      
      Result = Result->convertInitListSlice(Ranges);
      if (Result == 0) {
        Error(SquareLoc, "Invalid range for list slice");
        return 0;
      }
      
      // Eat the ']'.
      if (Lex.getCode() != tgtok::r_square) {
        TokError("expected ']' at end of list slice");
        return 0;
      }
      Lex.Lex();
      break;
    }
    case tgtok::period:
      if (Lex.Lex() != tgtok::Id) {  // eat the .
        TokError("expected field identifier after '.'");
        return 0;
      }
      if (!Result->getFieldType(Lex.getCurStrVal())) {
        TokError("Cannot access field '" + Lex.getCurStrVal() + "' of value '" +
                 Result->getAsString() + "'");
        return 0;
      }
      Result = new FieldInit(Result, Lex.getCurStrVal());
      Lex.Lex();  // eat field name
      break;
    }
  }
}

/// ParseDagArgList - Parse the argument list for a dag literal expression.
///
///    ParseDagArgList ::= Value (':' VARNAME)?
///    ParseDagArgList ::= ParseDagArgList ',' Value (':' VARNAME)?
std::vector<std::pair<llvm::Init*, std::string> > 
TGParser::ParseDagArgList(Record *CurRec) {
  std::vector<std::pair<llvm::Init*, std::string> > Result;
  
  while (1) {
    Init *Val = ParseValue(CurRec);
    if (Val == 0) return std::vector<std::pair<llvm::Init*, std::string> >();
    
    // If the variable name is present, add it.
    std::string VarName;
    if (Lex.getCode() == tgtok::colon) {
      if (Lex.Lex() != tgtok::VarName) { // eat the ':'
        TokError("expected variable name in dag literal");
        return std::vector<std::pair<llvm::Init*, std::string> >();
      }
      VarName = Lex.getCurStrVal();
      Lex.Lex();  // eat the VarName.
    }
    
    Result.push_back(std::make_pair(Val, VarName));
    
    if (Lex.getCode() != tgtok::comma) break;
    Lex.Lex(); // eat the ','    
  }
  
  return Result;
}


/// ParseValueList - Parse a comma separated list of values, returning them as a
/// vector.  Note that this always expects to be able to parse at least one
/// value.  It returns an empty list if this is not possible.
///
///   ValueList ::= Value (',' Value)
///
std::vector<Init*> TGParser::ParseValueList(Record *CurRec) {
  std::vector<Init*> Result;
  Result.push_back(ParseValue(CurRec));
  if (Result.back() == 0) return std::vector<Init*>();
  
  while (Lex.getCode() == tgtok::comma) {
    Lex.Lex();  // Eat the comma
    
    Result.push_back(ParseValue(CurRec));
    if (Result.back() == 0) return std::vector<Init*>();
  }
  
  return Result;
}



/// ParseDeclaration - Read a declaration, returning the name of field ID, or an
/// empty string on error.  This can happen in a number of different context's,
/// including within a def or in the template args for a def (which which case
/// CurRec will be non-null) and within the template args for a multiclass (in
/// which case CurRec will be null, but CurMultiClass will be set).  This can
/// also happen within a def that is within a multiclass, which will set both
/// CurRec and CurMultiClass.
///
///  Declaration ::= FIELD? Type ID ('=' Value)?
///
std::string TGParser::ParseDeclaration(Record *CurRec, 
                                       bool ParsingTemplateArgs) {
  // Read the field prefix if present.
  bool HasField = Lex.getCode() == tgtok::Field;
  if (HasField) Lex.Lex();
  
  RecTy *Type = ParseType();
  if (Type == 0) return "";
  
  if (Lex.getCode() != tgtok::Id) {
    TokError("Expected identifier in declaration");
    return "";
  }
  
  LocTy IdLoc = Lex.getLoc();
  std::string DeclName = Lex.getCurStrVal();
  Lex.Lex();
  
  if (ParsingTemplateArgs) {
    if (CurRec) {
      DeclName = CurRec->getName() + ":" + DeclName;
    } else {
      assert(CurMultiClass);
    }
    if (CurMultiClass)
      DeclName = CurMultiClass->Rec.getName() + "::" + DeclName;
  }
  
  // Add the value.
  if (AddValue(CurRec, IdLoc, RecordVal(DeclName, Type, HasField)))
    return "";
  
  // If a value is present, parse it.
  if (Lex.getCode() == tgtok::equal) {
    Lex.Lex();
    LocTy ValLoc = Lex.getLoc();
    Init *Val = ParseValue(CurRec);
    if (Val == 0 ||
        SetValue(CurRec, ValLoc, DeclName, std::vector<unsigned>(), Val))
      return "";
  }
  
  return DeclName;
}

/// ParseTemplateArgList - Read a template argument list, which is a non-empty
/// sequence of template-declarations in <>'s.  If CurRec is non-null, these are
/// template args for a def, which may or may not be in a multiclass.  If null,
/// these are the template args for a multiclass.
///
///    TemplateArgList ::= '<' Declaration (',' Declaration)* '>'
/// 
bool TGParser::ParseTemplateArgList(Record *CurRec) {
  assert(Lex.getCode() == tgtok::less && "Not a template arg list!");
  Lex.Lex(); // eat the '<'
  
  Record *TheRecToAddTo = CurRec ? CurRec : &CurMultiClass->Rec;
  
  // Read the first declaration.
  std::string TemplArg = ParseDeclaration(CurRec, true/*templateargs*/);
  if (TemplArg.empty())
    return true;
  
  TheRecToAddTo->addTemplateArg(TemplArg);
  
  while (Lex.getCode() == tgtok::comma) {
    Lex.Lex(); // eat the ','
    
    // Read the following declarations.
    TemplArg = ParseDeclaration(CurRec, true/*templateargs*/);
    if (TemplArg.empty())
      return true;
    TheRecToAddTo->addTemplateArg(TemplArg);
  }
  
  if (Lex.getCode() != tgtok::greater)
    return TokError("expected '>' at end of template argument list");
  Lex.Lex(); // eat the '>'.
  return false;
}


/// ParseBodyItem - Parse a single item at within the body of a def or class.
///
///   BodyItem ::= Declaration ';'
///   BodyItem ::= LET ID OptionalBitList '=' Value ';'
bool TGParser::ParseBodyItem(Record *CurRec) {
  if (Lex.getCode() != tgtok::Let) {
    if (ParseDeclaration(CurRec, false).empty()) 
      return true;
    
    if (Lex.getCode() != tgtok::semi)
      return TokError("expected ';' after declaration");
    Lex.Lex();
    return false;
  }

  // LET ID OptionalRangeList '=' Value ';'
  if (Lex.Lex() != tgtok::Id)
    return TokError("expected field identifier after let");
  
  LocTy IdLoc = Lex.getLoc();
  std::string FieldName = Lex.getCurStrVal();
  Lex.Lex();  // eat the field name.
  
  std::vector<unsigned> BitList;
  if (ParseOptionalBitList(BitList)) 
    return true;
  std::reverse(BitList.begin(), BitList.end());
  
  if (Lex.getCode() != tgtok::equal)
    return TokError("expected '=' in let expression");
  Lex.Lex();  // eat the '='.
  
  Init *Val = ParseValue(CurRec);
  if (Val == 0) return true;
  
  if (Lex.getCode() != tgtok::semi)
    return TokError("expected ';' after let expression");
  Lex.Lex();
  
  return SetValue(CurRec, IdLoc, FieldName, BitList, Val);
}

/// ParseBody - Read the body of a class or def.  Return true on error, false on
/// success.
///
///   Body     ::= ';'
///   Body     ::= '{' BodyList '}'
///   BodyList BodyItem*
///
bool TGParser::ParseBody(Record *CurRec) {
  // If this is a null definition, just eat the semi and return.
  if (Lex.getCode() == tgtok::semi) {
    Lex.Lex();
    return false;
  }
  
  if (Lex.getCode() != tgtok::l_brace)
    return TokError("Expected ';' or '{' to start body");
  // Eat the '{'.
  Lex.Lex();
  
  while (Lex.getCode() != tgtok::r_brace)
    if (ParseBodyItem(CurRec))
      return true;

  // Eat the '}'.
  Lex.Lex();
  return false;
}

/// ParseObjectBody - Parse the body of a def or class.  This consists of an
/// optional ClassList followed by a Body.  CurRec is the current def or class
/// that is being parsed.
///
///   ObjectBody      ::= BaseClassList Body
///   BaseClassList   ::= /*empty*/
///   BaseClassList   ::= ':' BaseClassListNE
///   BaseClassListNE ::= SubClassRef (',' SubClassRef)*
///
bool TGParser::ParseObjectBody(Record *CurRec) {
  // If there is a baseclass list, read it.
  if (Lex.getCode() == tgtok::colon) {
    Lex.Lex();
    
    // Read all of the subclasses.
    SubClassReference SubClass = ParseSubClassReference(CurRec, false);
    while (1) {
      // Check for error.
      if (SubClass.Rec == 0) return true;
     
      // Add it.
      if (AddSubClass(CurRec, SubClass))
        return true;
      
      if (Lex.getCode() != tgtok::comma) break;
      Lex.Lex(); // eat ','.
      SubClass = ParseSubClassReference(CurRec, false);
    }
  }

  // Process any variables on the let stack.
  for (unsigned i = 0, e = LetStack.size(); i != e; ++i)
    for (unsigned j = 0, e = LetStack[i].size(); j != e; ++j)
      if (SetValue(CurRec, LetStack[i][j].Loc, LetStack[i][j].Name,
                   LetStack[i][j].Bits, LetStack[i][j].Value))
        return true;
  
  return ParseBody(CurRec);
}


/// ParseDef - Parse and return a top level or multiclass def, return the record
/// corresponding to it.  This returns null on error.
///
///   DefInst ::= DEF ObjectName ObjectBody
///
llvm::Record *TGParser::ParseDef(MultiClass *CurMultiClass) {
  LocTy DefLoc = Lex.getLoc();
  assert(Lex.getCode() == tgtok::Def && "Unknown tok");
  Lex.Lex();  // Eat the 'def' token.  

  // Parse ObjectName and make a record for it.
  Record *CurRec = new Record(ParseObjectName());
  
  if (!CurMultiClass) {
    // Top-level def definition.
    
    // Ensure redefinition doesn't happen.
    if (Records.getDef(CurRec->getName())) {
      Error(DefLoc, "def '" + CurRec->getName() + "' already defined");
      return 0;
    }
    Records.addDef(CurRec);
  } else {
    // Otherwise, a def inside a multiclass, add it to the multiclass.
    for (unsigned i = 0, e = CurMultiClass->DefPrototypes.size(); i != e; ++i)
      if (CurMultiClass->DefPrototypes[i]->getName() == CurRec->getName()) {
        Error(DefLoc, "def '" + CurRec->getName() +
              "' already defined in this multiclass!");
        return 0;
      }
    CurMultiClass->DefPrototypes.push_back(CurRec);
  }
  
  if (ParseObjectBody(CurRec))
    return 0;
  
  if (CurMultiClass == 0)  // Def's in multiclasses aren't really defs.
    CurRec->resolveReferences();
  
  // If ObjectBody has template arguments, it's an error.
  assert(CurRec->getTemplateArgs().empty() && "How'd this get template args?");
  return CurRec;
}


/// ParseClass - Parse a tblgen class definition.
///
///   ClassInst ::= CLASS ID TemplateArgList? ObjectBody
///
bool TGParser::ParseClass() {
  assert(Lex.getCode() == tgtok::Class && "Unexpected token!");
  Lex.Lex();
  
  if (Lex.getCode() != tgtok::Id)
    return TokError("expected class name after 'class' keyword");
  
  Record *CurRec = Records.getClass(Lex.getCurStrVal());
  if (CurRec) {
    // If the body was previously defined, this is an error.
    if (!CurRec->getValues().empty() ||
        !CurRec->getSuperClasses().empty() ||
        !CurRec->getTemplateArgs().empty())
      return TokError("Class '" + CurRec->getName() + "' already defined");
  } else {
    // If this is the first reference to this class, create and add it.
    CurRec = new Record(Lex.getCurStrVal());
    Records.addClass(CurRec);
  }
  Lex.Lex(); // eat the name.
  
  // If there are template args, parse them.
  if (Lex.getCode() == tgtok::less)
    if (ParseTemplateArgList(CurRec))
      return true;

  // Finally, parse the object body.
  return ParseObjectBody(CurRec);
}

/// ParseLetList - Parse a non-empty list of assignment expressions into a list
/// of LetRecords.
///
///   LetList ::= LetItem (',' LetItem)*
///   LetItem ::= ID OptionalRangeList '=' Value
///
std::vector<LetRecord> TGParser::ParseLetList() {
  std::vector<LetRecord> Result;
  
  while (1) {
    if (Lex.getCode() != tgtok::Id) {
      TokError("expected identifier in let definition");
      return std::vector<LetRecord>();
    }
    std::string Name = Lex.getCurStrVal();
    LocTy NameLoc = Lex.getLoc();
    Lex.Lex();  // Eat the identifier. 

    // Check for an optional RangeList.
    std::vector<unsigned> Bits;
    if (ParseOptionalRangeList(Bits)) 
      return std::vector<LetRecord>();
    std::reverse(Bits.begin(), Bits.end());
    
    if (Lex.getCode() != tgtok::equal) {
      TokError("expected '=' in let expression");
      return std::vector<LetRecord>();
    }
    Lex.Lex();  // eat the '='.
    
    Init *Val = ParseValue(0);
    if (Val == 0) return std::vector<LetRecord>();
    
    // Now that we have everything, add the record.
    Result.push_back(LetRecord(Name, Bits, Val, NameLoc));
    
    if (Lex.getCode() != tgtok::comma)
      return Result;
    Lex.Lex();  // eat the comma.    
  }
}

/// ParseTopLevelLet - Parse a 'let' at top level.  This can be a couple of
/// different related productions.
///
///   Object ::= LET LetList IN '{' ObjectList '}'
///   Object ::= LET LetList IN Object
///
bool TGParser::ParseTopLevelLet() {
  assert(Lex.getCode() == tgtok::Let && "Unexpected token");
  Lex.Lex();
  
  // Add this entry to the let stack.
  std::vector<LetRecord> LetInfo = ParseLetList();
  if (LetInfo.empty()) return true;
  LetStack.push_back(LetInfo);

  if (Lex.getCode() != tgtok::In)
    return TokError("expected 'in' at end of top-level 'let'");
  Lex.Lex();
  
  // If this is a scalar let, just handle it now
  if (Lex.getCode() != tgtok::l_brace) {
    // LET LetList IN Object
    if (ParseObject())
      return true;
  } else {   // Object ::= LETCommand '{' ObjectList '}'
    LocTy BraceLoc = Lex.getLoc();
    // Otherwise, this is a group let.
    Lex.Lex();  // eat the '{'.
    
    // Parse the object list.
    if (ParseObjectList())
      return true;
    
    if (Lex.getCode() != tgtok::r_brace) {
      TokError("expected '}' at end of top level let command");
      return Error(BraceLoc, "to match this '{'");
    }
    Lex.Lex();
  }
  
  // Outside this let scope, this let block is not active.
  LetStack.pop_back();
  return false;
}

/// ParseMultiClassDef - Parse a def in a multiclass context.
///
///  MultiClassDef ::= DefInst
///
bool TGParser::ParseMultiClassDef(MultiClass *CurMC) {
  if (Lex.getCode() != tgtok::Def) 
    return TokError("expected 'def' in multiclass body");

  Record *D = ParseDef(CurMC);
  if (D == 0) return true;
  
  // Copy the template arguments for the multiclass into the def.
  const std::vector<std::string> &TArgs = CurMC->Rec.getTemplateArgs();
  
  for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
    const RecordVal *RV = CurMC->Rec.getValue(TArgs[i]);
    assert(RV && "Template arg doesn't exist?");
    D->addValue(*RV);
  }

  return false;
}

/// ParseMultiClass - Parse a multiclass definition.
///
///  MultiClassInst ::= MULTICLASS ID TemplateArgList? '{' MultiClassDef+ '}'
///
bool TGParser::ParseMultiClass() {
  assert(Lex.getCode() == tgtok::MultiClass && "Unexpected token");
  Lex.Lex();  // Eat the multiclass token.

  if (Lex.getCode() != tgtok::Id)
    return TokError("expected identifier after multiclass for name");
  std::string Name = Lex.getCurStrVal();
  
  if (MultiClasses.count(Name))
    return TokError("multiclass '" + Name + "' already defined");
  
  CurMultiClass  = MultiClasses[Name] = new MultiClass(Name);
  Lex.Lex();  // Eat the identifier.
  
  // If there are template args, parse them.
  if (Lex.getCode() == tgtok::less)
    if (ParseTemplateArgList(0))
      return true;

  if (Lex.getCode() != tgtok::l_brace)
    return TokError("expected '{' in multiclass definition");

  if (Lex.Lex() == tgtok::r_brace)  // eat the '{'.
    return TokError("multiclass must contain at least one def");
  
  while (Lex.getCode() != tgtok::r_brace)
    if (ParseMultiClassDef(CurMultiClass))
      return true;
  
  Lex.Lex();  // eat the '}'.
  
  CurMultiClass = 0;
  return false;
}

/// ParseDefm - Parse the instantiation of a multiclass.
///
///   DefMInst ::= DEFM ID ':' DefmSubClassRef ';'
///
bool TGParser::ParseDefm() {
  assert(Lex.getCode() == tgtok::Defm && "Unexpected token!");
  if (Lex.Lex() != tgtok::Id)  // eat the defm.
    return TokError("expected identifier after defm");
  
  LocTy DefmPrefixLoc = Lex.getLoc();
  std::string DefmPrefix = Lex.getCurStrVal();
  if (Lex.Lex() != tgtok::colon)
    return TokError("expected ':' after defm identifier");
  
  // eat the colon.
  Lex.Lex();

  LocTy SubClassLoc = Lex.getLoc();
  SubClassReference Ref = ParseSubClassReference(0, true);
  if (Ref.Rec == 0) return true;
  
  if (Lex.getCode() != tgtok::semi)
    return TokError("expected ';' at end of defm");
  Lex.Lex();
  
  // To instantiate a multiclass, we need to first get the multiclass, then
  // instantiate each def contained in the multiclass with the SubClassRef
  // template parameters.
  MultiClass *MC = MultiClasses[Ref.Rec->getName()];
  assert(MC && "Didn't lookup multiclass correctly?");
  std::vector<Init*> &TemplateVals = Ref.TemplateArgs;
  
  // Verify that the correct number of template arguments were specified.
  const std::vector<std::string> &TArgs = MC->Rec.getTemplateArgs();
  if (TArgs.size() < TemplateVals.size())
    return Error(SubClassLoc,
                 "more template args specified than multiclass expects");
  
  // Loop over all the def's in the multiclass, instantiating each one.
  for (unsigned i = 0, e = MC->DefPrototypes.size(); i != e; ++i) {
    Record *DefProto = MC->DefPrototypes[i];
    
    // Add the suffix to the defm name to get the new name.
    Record *CurRec = new Record(DefmPrefix + DefProto->getName());
    
    SubClassReference Ref;
    Ref.RefLoc = DefmPrefixLoc;
    Ref.Rec = DefProto;
    AddSubClass(CurRec, Ref);
    
    // Loop over all of the template arguments, setting them to the specified
    // value or leaving them as the default if necessary.
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i < TemplateVals.size()) { // A value is specified for this temp-arg?
        // Set it now.
        if (SetValue(CurRec, DefmPrefixLoc, TArgs[i], std::vector<unsigned>(),
                     TemplateVals[i]))
          return true;
        
        // Resolve it next.
        CurRec->resolveReferencesTo(CurRec->getValue(TArgs[i]));
        
        // Now remove it.
        CurRec->removeValue(TArgs[i]);
        
      } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
        return Error(SubClassLoc, "value not specified for template argument #"+
                     utostr(i) + " (" + TArgs[i] + ") of multiclassclass '" +
                     MC->Rec.getName() + "'");
      }
    }
    
    // If the mdef is inside a 'let' expression, add to each def.
    for (unsigned i = 0, e = LetStack.size(); i != e; ++i)
      for (unsigned j = 0, e = LetStack[i].size(); j != e; ++j)
        if (SetValue(CurRec, LetStack[i][j].Loc, LetStack[i][j].Name,
                     LetStack[i][j].Bits, LetStack[i][j].Value)) {
          Error(DefmPrefixLoc, "when instantiating this defm");
          return true;
        }
    
    
    // Ensure redefinition doesn't happen.
    if (Records.getDef(CurRec->getName()))
      return Error(DefmPrefixLoc, "def '" + CurRec->getName() + 
                   "' already defined, instantiating defm with subdef '" + 
                   DefProto->getName() + "'");
    Records.addDef(CurRec);
    CurRec->resolveReferences();
  }
  
  return false;
}

/// ParseObject
///   Object ::= ClassInst
///   Object ::= DefInst
///   Object ::= MultiClassInst
///   Object ::= DefMInst
///   Object ::= LETCommand '{' ObjectList '}'
///   Object ::= LETCommand Object
bool TGParser::ParseObject() {
  switch (Lex.getCode()) {
  default: assert(0 && "This is not an object");
  case tgtok::Let:   return ParseTopLevelLet();
  case tgtok::Def:   return ParseDef(0) == 0;
  case tgtok::Defm:  return ParseDefm();
  case tgtok::Class: return ParseClass();
  case tgtok::MultiClass: return ParseMultiClass();
  }
}

/// ParseObjectList
///   ObjectList :== Object*
bool TGParser::ParseObjectList() {
  while (isObjectStart(Lex.getCode())) {
    if (ParseObject())
      return true;
  }
  return false;
}


bool TGParser::ParseFile() {
  Lex.Lex(); // Prime the lexer.
  if (ParseObjectList()) return true;
  
  // If we have unread input at the end of the file, report it.
  if (Lex.getCode() == tgtok::Eof)
    return false;
  
  return TokError("Unexpected input at top level");
}

