//===- Record.cpp - Record implementation ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the tablegen record classes.
//
//===----------------------------------------------------------------------===//

#include "Record.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

void RecTy::dump() const { print(errs()); }

Init *BitRecTy::convertValue(BitsInit *BI) {
  if (BI->getNumBits() != 1) return 0; // Only accept if just one bit!
  return BI->getBit(0);
}

bool BitRecTy::baseClassOf(const BitsRecTy *RHS) const {
  return RHS->getNumBits() == 1;
}

Init *BitRecTy::convertValue(IntInit *II) {
  int64_t Val = II->getValue();
  if (Val != 0 && Val != 1) return 0;  // Only accept 0 or 1 for a bit!

  return new BitInit(Val != 0);
}

Init *BitRecTy::convertValue(TypedInit *VI) {
  if (dynamic_cast<BitRecTy*>(VI->getType()))
    return VI;  // Accept variable if it is already of bit type!
  return 0;
}

std::string BitsRecTy::getAsString() const {
  return "bits<" + utostr(Size) + ">";
}

Init *BitsRecTy::convertValue(UnsetInit *UI) {
  BitsInit *Ret = new BitsInit(Size);

  for (unsigned i = 0; i != Size; ++i)
    Ret->setBit(i, new UnsetInit());
  return Ret;
}

Init *BitsRecTy::convertValue(BitInit *UI) {
  if (Size != 1) return 0;  // Can only convert single bit...
  BitsInit *Ret = new BitsInit(1);
  Ret->setBit(0, UI);
  return Ret;
}

// convertValue from Int initializer to bits type: Split the integer up into the
// appropriate bits...
//
Init *BitsRecTy::convertValue(IntInit *II) {
  int64_t Value = II->getValue();
  // Make sure this bitfield is large enough to hold the integer value...
  if (Value >= 0) {
    if (Value & ~((1LL << Size)-1))
      return 0;
  } else {
    if ((Value >> Size) != -1 || ((Value & (1LL << (Size-1))) == 0))
      return 0;
  }

  BitsInit *Ret = new BitsInit(Size);
  for (unsigned i = 0; i != Size; ++i)
    Ret->setBit(i, new BitInit(Value & (1LL << i)));

  return Ret;
}

Init *BitsRecTy::convertValue(BitsInit *BI) {
  // If the number of bits is right, return it.  Otherwise we need to expand or
  // truncate...
  if (BI->getNumBits() == Size) return BI;
  return 0;
}

Init *BitsRecTy::convertValue(TypedInit *VI) {
  if (BitsRecTy *BRT = dynamic_cast<BitsRecTy*>(VI->getType()))
    if (BRT->Size == Size) {
      BitsInit *Ret = new BitsInit(Size);
      for (unsigned i = 0; i != Size; ++i)
        Ret->setBit(i, new VarBitInit(VI, i));
      return Ret;
    }
  if (Size == 1 && dynamic_cast<BitRecTy*>(VI->getType())) {
    BitsInit *Ret = new BitsInit(1);
    Ret->setBit(0, VI);
    return Ret;
  }

  return 0;
}

Init *IntRecTy::convertValue(BitInit *BI) {
  return new IntInit(BI->getValue());
}

Init *IntRecTy::convertValue(BitsInit *BI) {
  int64_t Result = 0;
  for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i)
    if (BitInit *Bit = dynamic_cast<BitInit*>(BI->getBit(i))) {
      Result |= Bit->getValue() << i;
    } else {
      return 0;
    }
  return new IntInit(Result);
}

Init *IntRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

Init *StringRecTy::convertValue(UnOpInit *BO) {
  if (BO->getOpcode() == UnOpInit::CAST) {
    Init *L = BO->getOperand()->convertInitializerTo(this);
    if (L == 0) return 0;
    if (L != BO->getOperand())
      return new UnOpInit(UnOpInit::CAST, L, new StringRecTy);
    return BO;
  }

  return convertValue((TypedInit*)BO);
}

Init *StringRecTy::convertValue(BinOpInit *BO) {
  if (BO->getOpcode() == BinOpInit::STRCONCAT) {
    Init *L = BO->getLHS()->convertInitializerTo(this);
    Init *R = BO->getRHS()->convertInitializerTo(this);
    if (L == 0 || R == 0) return 0;
    if (L != BO->getLHS() || R != BO->getRHS())
      return new BinOpInit(BinOpInit::STRCONCAT, L, R, new StringRecTy);
    return BO;
  }
  if (BO->getOpcode() == BinOpInit::NAMECONCAT) {
    if (BO->getType()->getAsString() == getAsString()) {
      Init *L = BO->getLHS()->convertInitializerTo(this);
      Init *R = BO->getRHS()->convertInitializerTo(this);
      if (L == 0 || R == 0) return 0;
      if (L != BO->getLHS() || R != BO->getRHS())
        return new BinOpInit(BinOpInit::NAMECONCAT, L, R, new StringRecTy);
      return BO;
    }
  }

  return convertValue((TypedInit*)BO);
}


Init *StringRecTy::convertValue(TypedInit *TI) {
  if (dynamic_cast<StringRecTy*>(TI->getType()))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

std::string ListRecTy::getAsString() const {
  return "list<" + Ty->getAsString() + ">";
}

Init *ListRecTy::convertValue(ListInit *LI) {
  std::vector<Init*> Elements;

  // Verify that all of the elements of the list are subclasses of the
  // appropriate class!
  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    if (Init *CI = LI->getElement(i)->convertInitializerTo(Ty))
      Elements.push_back(CI);
    else
      return 0;

  ListRecTy *LType = dynamic_cast<ListRecTy*>(LI->getType());
  if (LType == 0) {
    return 0;
  }

  return new ListInit(Elements, new ListRecTy(Ty));
}

Init *ListRecTy::convertValue(TypedInit *TI) {
  // Ensure that TI is compatible with our class.
  if (ListRecTy *LRT = dynamic_cast<ListRecTy*>(TI->getType()))
    if (LRT->getElementType()->typeIsConvertibleTo(getElementType()))
      return TI;
  return 0;
}

Init *CodeRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}

Init *DagRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}

Init *DagRecTy::convertValue(UnOpInit *BO) {
  if (BO->getOpcode() == UnOpInit::CAST) {
    Init *L = BO->getOperand()->convertInitializerTo(this);
    if (L == 0) return 0;
    if (L != BO->getOperand())
      return new UnOpInit(UnOpInit::CAST, L, new DagRecTy);
    return BO;
  }
  return 0;
}

Init *DagRecTy::convertValue(BinOpInit *BO) {
  if (BO->getOpcode() == BinOpInit::CONCAT) {
    Init *L = BO->getLHS()->convertInitializerTo(this);
    Init *R = BO->getRHS()->convertInitializerTo(this);
    if (L == 0 || R == 0) return 0;
    if (L != BO->getLHS() || R != BO->getRHS())
      return new BinOpInit(BinOpInit::CONCAT, L, R, new DagRecTy);
    return BO;
  }
  if (BO->getOpcode() == BinOpInit::NAMECONCAT) {
    if (BO->getType()->getAsString() == getAsString()) {
      Init *L = BO->getLHS()->convertInitializerTo(this);
      Init *R = BO->getRHS()->convertInitializerTo(this);
      if (L == 0 || R == 0) return 0;
      if (L != BO->getLHS() || R != BO->getRHS())
        return new BinOpInit(BinOpInit::CONCAT, L, R, new DagRecTy);
      return BO;
    }
  }
  return 0;
}

std::string RecordRecTy::getAsString() const {
  return Rec->getName();
}

Init *RecordRecTy::convertValue(DefInit *DI) {
  // Ensure that DI is a subclass of Rec.
  if (!DI->getDef()->isSubClassOf(Rec))
    return 0;
  return DI;
}

Init *RecordRecTy::convertValue(TypedInit *TI) {
  // Ensure that TI is compatible with Rec.
  if (RecordRecTy *RRT = dynamic_cast<RecordRecTy*>(TI->getType()))
    if (RRT->getRecord()->isSubClassOf(getRecord()) ||
        RRT->getRecord() == getRecord())
      return TI;
  return 0;
}

bool RecordRecTy::baseClassOf(const RecordRecTy *RHS) const {
  return Rec == RHS->getRecord() || RHS->getRecord()->isSubClassOf(Rec);
}


/// resolveTypes - Find a common type that T1 and T2 convert to.  
/// Return 0 if no such type exists.
///
RecTy *llvm::resolveTypes(RecTy *T1, RecTy *T2) {
  if (!T1->typeIsConvertibleTo(T2)) {
    if (!T2->typeIsConvertibleTo(T1)) {
      // If one is a Record type, check superclasses
      RecordRecTy *RecTy1 = dynamic_cast<RecordRecTy*>(T1);
      if (RecTy1) {
        // See if T2 inherits from a type T1 also inherits from
        const std::vector<Record *> &T1SuperClasses = RecTy1->getRecord()->getSuperClasses();
        for(std::vector<Record *>::const_iterator i = T1SuperClasses.begin(),
              iend = T1SuperClasses.end();
            i != iend;
            ++i) {
          RecordRecTy *SuperRecTy1 = new RecordRecTy(*i);
          RecTy *NewType1 = resolveTypes(SuperRecTy1, T2);
          if (NewType1 != 0) {
            if (NewType1 != SuperRecTy1) {
              delete SuperRecTy1;
            }
            return NewType1;
          }
        }
      }
      RecordRecTy *RecTy2 = dynamic_cast<RecordRecTy*>(T2);
      if (RecTy2) {
        // See if T1 inherits from a type T2 also inherits from
        const std::vector<Record *> &T2SuperClasses = RecTy2->getRecord()->getSuperClasses();
        for(std::vector<Record *>::const_iterator i = T2SuperClasses.begin(),
              iend = T2SuperClasses.end();
            i != iend;
            ++i) {
          RecordRecTy *SuperRecTy2 = new RecordRecTy(*i);
          RecTy *NewType2 = resolveTypes(T1, SuperRecTy2);
          if (NewType2 != 0) {
            if (NewType2 != SuperRecTy2) {
              delete SuperRecTy2;
            }
            return NewType2;
          }
        }
      }
      return 0;
    }
    return T2;
  }
  return T1;
}


//===----------------------------------------------------------------------===//
//    Initializer implementations
//===----------------------------------------------------------------------===//

void Init::dump() const { return print(errs()); }

Init *BitsInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsInit *BI = new BitsInit(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= getNumBits()) {
      delete BI;
      return 0;
    }
    BI->setBit(i, getBit(Bits[i]));
  }
  return BI;
}

std::string BitsInit::getAsString() const {
  //if (!printInHex(OS)) return;
  //if (!printAsVariable(OS)) return;
  //if (!printAsUnset(OS)) return;

  std::string Result = "{ ";
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    if (i) Result += ", ";
    if (Init *Bit = getBit(e-i-1))
      Result += Bit->getAsString();
    else
      Result += "*";
  }
  return Result + " }";
}

bool BitsInit::printInHex(raw_ostream &OS) const {
  // First, attempt to convert the value into an integer value...
  int64_t Result = 0;
  for (unsigned i = 0, e = getNumBits(); i != e; ++i)
    if (BitInit *Bit = dynamic_cast<BitInit*>(getBit(i))) {
      Result |= Bit->getValue() << i;
    } else {
      return true;
    }

  OS << format("0x%x", Result);
  return false;
}

bool BitsInit::printAsVariable(raw_ostream &OS) const {
  // Get the variable that we may be set equal to...
  assert(getNumBits() != 0);
  VarBitInit *FirstBit = dynamic_cast<VarBitInit*>(getBit(0));
  if (FirstBit == 0) return true;
  TypedInit *Var = FirstBit->getVariable();

  // Check to make sure the types are compatible.
  BitsRecTy *Ty = dynamic_cast<BitsRecTy*>(FirstBit->getVariable()->getType());
  if (Ty == 0) return true;
  if (Ty->getNumBits() != getNumBits()) return true; // Incompatible types!

  // Check to make sure all bits are referring to the right bits in the variable
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    VarBitInit *Bit = dynamic_cast<VarBitInit*>(getBit(i));
    if (Bit == 0 || Bit->getVariable() != Var || Bit->getBitNum() != i)
      return true;
  }

  Var->print(OS);
  return false;
}

bool BitsInit::printAsUnset(raw_ostream &OS) const {
  for (unsigned i = 0, e = getNumBits(); i != e; ++i)
    if (!dynamic_cast<UnsetInit*>(getBit(i)))
      return true;
  OS << "?";
  return false;
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
//
Init *BitsInit::resolveReferences(Record &R, const RecordVal *RV) {
  bool Changed = false;
  BitsInit *New = new BitsInit(getNumBits());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    Init *B;
    Init *CurBit = getBit(i);

    do {
      B = CurBit;
      CurBit = CurBit->resolveReferences(R, RV);
      Changed |= B != CurBit;
    } while (B != CurBit);
    New->setBit(i, CurBit);
  }

  if (Changed)
    return New;
  delete New;
  return this;
}

std::string IntInit::getAsString() const {
  return itostr(Value);
}

Init *IntInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsInit *BI = new BitsInit(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= 64) {
      delete BI;
      return 0;
    }
    BI->setBit(i, new BitInit(Value & (INT64_C(1) << Bits[i])));
  }
  return BI;
}

Init *ListInit::convertInitListSlice(const std::vector<unsigned> &Elements) {
  std::vector<Init*> Vals;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] >= getSize())
      return 0;
    Vals.push_back(getElement(Elements[i]));
  }
  return new ListInit(Vals, getType());
}

Record *ListInit::getElementAsRecord(unsigned i) const {
  assert(i < Values.size() && "List element index out of range!");
  DefInit *DI = dynamic_cast<DefInit*>(Values[i]);
  if (DI == 0) throw "Expected record in list!";
  return DI->getDef();
}

Init *ListInit::resolveReferences(Record &R, const RecordVal *RV) {
  std::vector<Init*> Resolved;
  Resolved.reserve(getSize());
  bool Changed = false;

  for (unsigned i = 0, e = getSize(); i != e; ++i) {
    Init *E;
    Init *CurElt = getElement(i);

    do {
      E = CurElt;
      CurElt = CurElt->resolveReferences(R, RV);
      Changed |= E != CurElt;
    } while (E != CurElt);
    Resolved.push_back(E);
  }

  if (Changed)
    return new ListInit(Resolved, getType());
  return this;
}

Init *ListInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                           unsigned Elt) {
  if (Elt >= getSize())
    return 0;  // Out of range reference.
  Init *E = getElement(Elt);
  if (!dynamic_cast<UnsetInit*>(E))  // If the element is set
    return E;                        // Replace the VarListElementInit with it.
  return 0;
}

std::string ListInit::getAsString() const {
  std::string Result = "[";
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (i) Result += ", ";
    Result += Values[i]->getAsString();
  }
  return Result + "]";
}

Init *OpInit::resolveBitReference(Record &R, const RecordVal *IRV,
                                   unsigned Bit) {
  Init *Folded = Fold(&R, 0);

  if (Folded != this) {
    TypedInit *Typed = dynamic_cast<TypedInit *>(Folded);
    if (Typed) {
      return Typed->resolveBitReference(R, IRV, Bit);
    }    
  }
  
  return 0;
}

Init *OpInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                           unsigned Elt) {
  Init *Folded = Fold(&R, 0);

  if (Folded != this) {
    TypedInit *Typed = dynamic_cast<TypedInit *>(Folded);
    if (Typed) {
      return Typed->resolveListElementReference(R, IRV, Elt);
    }    
  }
  
  return 0;
}

Init *UnOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) {
  switch (getOpcode()) {
  default: assert(0 && "Unknown unop");
  case CAST: {
    if (getType()->getAsString() == "string") {
      StringInit *LHSs = dynamic_cast<StringInit*>(LHS);
      if (LHSs) {
        return LHSs;
      }

      DefInit *LHSd = dynamic_cast<DefInit*>(LHS);
      if (LHSd) {
        return new StringInit(LHSd->getDef()->getName());
      }
    }
    else {
      StringInit *LHSs = dynamic_cast<StringInit*>(LHS);
      if (LHSs) {
        std::string Name = LHSs->getValue();

        // From TGParser::ParseIDValue
        if (CurRec) {
          if (const RecordVal *RV = CurRec->getValue(Name)) {
            if (RV->getType() != getType()) {
              throw "type mismatch in nameconcat";
            }
            return new VarInit(Name, RV->getType());
          }

          std::string TemplateArgName = CurRec->getName()+":"+Name;
          if (CurRec->isTemplateArg(TemplateArgName)) {
            const RecordVal *RV = CurRec->getValue(TemplateArgName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType()) {
              throw "type mismatch in nameconcat";
            }

            return new VarInit(TemplateArgName, RV->getType());
          }
        }

        if (CurMultiClass) {
          std::string MCName = CurMultiClass->Rec.getName()+"::"+Name;
          if (CurMultiClass->Rec.isTemplateArg(MCName)) {
            const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
            assert(RV && "Template arg doesn't exist??");
            
            if (RV->getType() != getType()) {
              throw "type mismatch in nameconcat";
            }
            
            return new VarInit(MCName, RV->getType());
          }
        }
        
        if (Record *D = Records.getDef(Name))
          return new DefInit(D);

        errs() << "Variable not defined: '" + Name + "'\n";
        assert(0 && "Variable not found");
        return 0;
      }
    }
    break;
  }
  case CAR: {
    ListInit *LHSl = dynamic_cast<ListInit*>(LHS);
    if (LHSl) {
      if (LHSl->getSize() == 0) {
        assert(0 && "Empty list in car");
        return 0;
      }
      return LHSl->getElement(0);
    }
    break;
  }
  case CDR: {
    ListInit *LHSl = dynamic_cast<ListInit*>(LHS);
    if (LHSl) {
      if (LHSl->getSize() == 0) {
        assert(0 && "Empty list in cdr");
        return 0;
      }
      ListInit *Result = new ListInit(LHSl->begin()+1, LHSl->end(), LHSl->getType());
      return Result;
    }
    break;
  }
  case LNULL: {
    ListInit *LHSl = dynamic_cast<ListInit*>(LHS);
    if (LHSl) {
      if (LHSl->getSize() == 0) {
        return new IntInit(1);
      }
      else {
        return new IntInit(0);
      }
    }
    StringInit *LHSs = dynamic_cast<StringInit*>(LHS);
    if (LHSs) {
      if (LHSs->getValue().empty()) {
        return new IntInit(1);
      }
      else {
        return new IntInit(0);
      }
    }
    
    break;
  }
  }
  return this;
}

Init *UnOpInit::resolveReferences(Record &R, const RecordVal *RV) {
  Init *lhs = LHS->resolveReferences(R, RV);
  
  if (LHS != lhs)
    return (new UnOpInit(getOpcode(), lhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string UnOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CAST: Result = "!cast<" + getType()->getAsString() + ">"; break;
  case CAR: Result = "!car"; break;
  case CDR: Result = "!cdr"; break;
  case LNULL: Result = "!null"; break;
  }
  return Result + "(" + LHS->getAsString() + ")";
}

RecTy *UnOpInit::getFieldType(const std::string &FieldName) const {
  switch (getOpcode()) {
  default: assert(0 && "Unknown unop");
  case CAST: {
    RecordRecTy *RecordType = dynamic_cast<RecordRecTy *>(getType());
    if (RecordType) {
      RecordVal *Field = RecordType->getRecord()->getValue(FieldName);
      if (Field) {
        return Field->getType();
      }
    }
    break;
  }
  }
  return 0;
}

Init *BinOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) {
  switch (getOpcode()) {
  default: assert(0 && "Unknown binop");
  case CONCAT: {
    DagInit *LHSs = dynamic_cast<DagInit*>(LHS);
    DagInit *RHSs = dynamic_cast<DagInit*>(RHS);
    if (LHSs && RHSs) {
      DefInit *LOp = dynamic_cast<DefInit*>(LHSs->getOperator());
      DefInit *ROp = dynamic_cast<DefInit*>(RHSs->getOperator());
      if (LOp->getDef() != ROp->getDef()) {
        bool LIsOps =
          LOp->getDef()->getName() == "outs" ||
          LOp->getDef()->getName() != "ins" ||
          LOp->getDef()->getName() != "defs";
        bool RIsOps =
          ROp->getDef()->getName() == "outs" ||
          ROp->getDef()->getName() != "ins" ||
          ROp->getDef()->getName() != "defs";
        if (!LIsOps || !RIsOps)
          throw "Concated Dag operators do not match!";
      }
      std::vector<Init*> Args;
      std::vector<std::string> ArgNames;
      for (unsigned i = 0, e = LHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(LHSs->getArg(i));
        ArgNames.push_back(LHSs->getArgName(i));
      }
      for (unsigned i = 0, e = RHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(RHSs->getArg(i));
        ArgNames.push_back(RHSs->getArgName(i));
      }
      return new DagInit(LHSs->getOperator(), "", Args, ArgNames);
    }
    break;
  }
  case STRCONCAT: {
    StringInit *LHSs = dynamic_cast<StringInit*>(LHS);
    StringInit *RHSs = dynamic_cast<StringInit*>(RHS);
    if (LHSs && RHSs)
      return new StringInit(LHSs->getValue() + RHSs->getValue());
    break;
  }
  case NAMECONCAT: {
    StringInit *LHSs = dynamic_cast<StringInit*>(LHS);
    StringInit *RHSs = dynamic_cast<StringInit*>(RHS);
    if (LHSs && RHSs) {
      std::string Name(LHSs->getValue() + RHSs->getValue());

      // From TGParser::ParseIDValue
      if (CurRec) {
        if (const RecordVal *RV = CurRec->getValue(Name)) {
          if (RV->getType() != getType()) {
            throw "type mismatch in nameconcat";
          }
          return new VarInit(Name, RV->getType());
        }
        
        std::string TemplateArgName = CurRec->getName()+":"+Name;
        if (CurRec->isTemplateArg(TemplateArgName)) {
          const RecordVal *RV = CurRec->getValue(TemplateArgName);
          assert(RV && "Template arg doesn't exist??");

          if (RV->getType() != getType()) {
            throw "type mismatch in nameconcat";
          }

          return new VarInit(TemplateArgName, RV->getType());
        }
      }

      if (CurMultiClass) {
        std::string MCName = CurMultiClass->Rec.getName()+"::"+Name;
        if (CurMultiClass->Rec.isTemplateArg(MCName)) {
          const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
          assert(RV && "Template arg doesn't exist??");

          if (RV->getType() != getType()) {
            throw "type mismatch in nameconcat";
          }
          
          return new VarInit(MCName, RV->getType());
        }
      }

      if (Record *D = Records.getDef(Name))
        return new DefInit(D);

      errs() << "Variable not defined in !nameconcat: '" + Name + "'\n";
      assert(0 && "Variable not found in !nameconcat");
      return 0;
    }
    break;
  }
  case SHL:
  case SRA:
  case SRL: {
    IntInit *LHSi = dynamic_cast<IntInit*>(LHS);
    IntInit *RHSi = dynamic_cast<IntInit*>(RHS);
    if (LHSi && RHSi) {
      int64_t LHSv = LHSi->getValue(), RHSv = RHSi->getValue();
      int64_t Result;
      switch (getOpcode()) {
      default: assert(0 && "Bad opcode!");
      case SHL: Result = LHSv << RHSv; break;
      case SRA: Result = LHSv >> RHSv; break;
      case SRL: Result = (uint64_t)LHSv >> (uint64_t)RHSv; break;
      }
      return new IntInit(Result);
    }
    break;
  }
  }
  return this;
}

Init *BinOpInit::resolveReferences(Record &R, const RecordVal *RV) {
  Init *lhs = LHS->resolveReferences(R, RV);
  Init *rhs = RHS->resolveReferences(R, RV);
  
  if (LHS != lhs || RHS != rhs)
    return (new BinOpInit(getOpcode(), lhs, rhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string BinOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CONCAT: Result = "!con"; break;
  case SHL: Result = "!shl"; break;
  case SRA: Result = "!sra"; break;
  case SRL: Result = "!srl"; break;
  case STRCONCAT: Result = "!strconcat"; break;
  case NAMECONCAT: 
    Result = "!nameconcat<" + getType()->getAsString() + ">"; break;
  }
  return Result + "(" + LHS->getAsString() + ", " + RHS->getAsString() + ")";
}

static Init *ForeachHelper(Init *LHS, Init *MHS, Init *RHS, RecTy *Type,
                           Record *CurRec, MultiClass *CurMultiClass);

static Init *EvaluateOperation(OpInit *RHSo, Init *LHS, Init *Arg,
                               RecTy *Type, Record *CurRec,
                               MultiClass *CurMultiClass) {
  std::vector<Init *> NewOperands;

  TypedInit *TArg = dynamic_cast<TypedInit*>(Arg);

  // If this is a dag, recurse
  if (TArg && TArg->getType()->getAsString() == "dag") {
    Init *Result = ForeachHelper(LHS, Arg, RHSo, Type,
                                 CurRec, CurMultiClass);
    if (Result != 0) {
      return Result;
    }
    else {
      return 0;
    }
  }

  for (int i = 0; i < RHSo->getNumOperands(); ++i) {
    OpInit *RHSoo = dynamic_cast<OpInit*>(RHSo->getOperand(i));

    if (RHSoo) {
      Init *Result = EvaluateOperation(RHSoo, LHS, Arg,
                                       Type, CurRec, CurMultiClass);
      if (Result != 0) {
        NewOperands.push_back(Result);
      }
      else {
        NewOperands.push_back(Arg);
      }
    }
    else if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
      NewOperands.push_back(Arg);
    }
    else {
      NewOperands.push_back(RHSo->getOperand(i));
    }
  }

  // Now run the operator and use its result as the new leaf
  OpInit *NewOp = RHSo->clone(NewOperands);
  Init *NewVal = NewOp->Fold(CurRec, CurMultiClass);
  if (NewVal != NewOp) {
    delete NewOp;
    return NewVal;
  }
  return 0;
}

static Init *ForeachHelper(Init *LHS, Init *MHS, Init *RHS, RecTy *Type,
                           Record *CurRec, MultiClass *CurMultiClass) {
  DagInit *MHSd = dynamic_cast<DagInit*>(MHS);
  ListInit *MHSl = dynamic_cast<ListInit*>(MHS);

  DagRecTy *DagType = dynamic_cast<DagRecTy*>(Type);
  ListRecTy *ListType = dynamic_cast<ListRecTy*>(Type);

  OpInit *RHSo = dynamic_cast<OpInit*>(RHS);

  if (!RHSo) {
    errs() << "!foreach requires an operator\n";
    assert(0 && "No operator for !foreach");
  }

  TypedInit *LHSt = dynamic_cast<TypedInit*>(LHS);

  if (!LHSt) {
    errs() << "!foreach requires typed variable\n";
    assert(0 && "No typed variable for !foreach");
  }

  if ((MHSd && DagType) || (MHSl && ListType)) {
    if (MHSd) {
      Init *Val = MHSd->getOperator();
      Init *Result = EvaluateOperation(RHSo, LHS, Val,
                                       Type, CurRec, CurMultiClass);
      if (Result != 0) {
        Val = Result;
      }

      std::vector<std::pair<Init *, std::string> > args;
      for (unsigned int i = 0; i < MHSd->getNumArgs(); ++i) {
        Init *Arg;
        std::string ArgName;
        Arg = MHSd->getArg(i);
        ArgName = MHSd->getArgName(i);

        // Process args
        Init *Result = EvaluateOperation(RHSo, LHS, Arg, Type,
                                         CurRec, CurMultiClass);
        if (Result != 0) {
          Arg = Result;
        }

        // TODO: Process arg names
        args.push_back(std::make_pair(Arg, ArgName));
      }

      return new DagInit(Val, "", args);
    }
    if (MHSl) {
      std::vector<Init *> NewOperands;
      std::vector<Init *> NewList(MHSl->begin(), MHSl->end());

      for (ListInit::iterator li = NewList.begin(),
             liend = NewList.end();
           li != liend;
           ++li) {
        Init *Item = *li;
        NewOperands.clear();
        for(int i = 0; i < RHSo->getNumOperands(); ++i) {
          // First, replace the foreach variable with the list item
          if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
            NewOperands.push_back(Item);
          }
          else {
            NewOperands.push_back(RHSo->getOperand(i));
          }
        }

        // Now run the operator and use its result as the new list item
        OpInit *NewOp = RHSo->clone(NewOperands);
        Init *NewItem = NewOp->Fold(CurRec, CurMultiClass);
        if (NewItem != NewOp) {
          *li = NewItem;
          delete NewOp;
        }
      }
      return new ListInit(NewList, MHSl->getType());
    }
  }
  return 0;
}

Init *TernOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) {
  switch (getOpcode()) {
  default: assert(0 && "Unknown binop");
  case SUBST: {
    DefInit *LHSd = dynamic_cast<DefInit*>(LHS);
    VarInit *LHSv = dynamic_cast<VarInit*>(LHS);
    StringInit *LHSs = dynamic_cast<StringInit*>(LHS);

    DefInit *MHSd = dynamic_cast<DefInit*>(MHS);
    VarInit *MHSv = dynamic_cast<VarInit*>(MHS);
    StringInit *MHSs = dynamic_cast<StringInit*>(MHS);

    DefInit *RHSd = dynamic_cast<DefInit*>(RHS);
    VarInit *RHSv = dynamic_cast<VarInit*>(RHS);
    StringInit *RHSs = dynamic_cast<StringInit*>(RHS);

    if ((LHSd && MHSd && RHSd)
        || (LHSv && MHSv && RHSv)
        || (LHSs && MHSs && RHSs)) {
      if (RHSd) {
        Record *Val = RHSd->getDef();
        if (LHSd->getAsString() == RHSd->getAsString()) {
          Val = MHSd->getDef();
        }
        return new DefInit(Val);
      }
      if (RHSv) {
        std::string Val = RHSv->getName();
        if (LHSv->getAsString() == RHSv->getAsString()) {
          Val = MHSv->getName();
        }
        return new VarInit(Val, getType());
      }
      if (RHSs) {
        std::string Val = RHSs->getValue();

        std::string::size_type found;
        do {
          found = Val.find(LHSs->getValue());
          if (found != std::string::npos) {
            Val.replace(found, LHSs->getValue().size(), MHSs->getValue());
          }
        } while (found != std::string::npos);

        return new StringInit(Val);
      }
    }
    break;
  }  

  case FOREACH: {
    Init *Result = ForeachHelper(LHS, MHS, RHS, getType(),
                                 CurRec, CurMultiClass);
    if (Result != 0) {
      return Result;
    }
    break;
  }

  case IF: {
    IntInit *LHSi = dynamic_cast<IntInit*>(LHS);
    if (LHSi) {
      if (LHSi->getValue()) {
        return MHS;
      }
      else {
        return RHS;
      }
    }
    break;
  }
  }

  return this;
}

Init *TernOpInit::resolveReferences(Record &R, const RecordVal *RV) {
  Init *lhs = LHS->resolveReferences(R, RV);

  if (Opc == IF && lhs != LHS) {
    IntInit *Value = dynamic_cast<IntInit*>(lhs);
    if (Value != 0) {
      // Short-circuit
      if (Value->getValue()) {
        Init *mhs = MHS->resolveReferences(R, RV);
        return (new TernOpInit(getOpcode(), lhs, mhs, RHS, getType()))->Fold(&R, 0);
      }
      else {
        Init *rhs = RHS->resolveReferences(R, RV);
        return (new TernOpInit(getOpcode(), lhs, MHS, rhs, getType()))->Fold(&R, 0);
      }
    }
  }
  
  Init *mhs = MHS->resolveReferences(R, RV);
  Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || MHS != mhs || RHS != rhs)
    return (new TernOpInit(getOpcode(), lhs, mhs, rhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string TernOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case SUBST: Result = "!subst"; break;
  case FOREACH: Result = "!foreach"; break; 
  case IF: Result = "!if"; break; 
 }
  return Result + "(" + LHS->getAsString() + ", " + MHS->getAsString() + ", " 
    + RHS->getAsString() + ")";
}

Init *TypedInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) {
  BitsRecTy *T = dynamic_cast<BitsRecTy*>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-bits variable...
  unsigned NumBits = T->getNumBits();

  BitsInit *BI = new BitsInit(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= NumBits) {
      delete BI;
      return 0;
    }
    BI->setBit(i, new VarBitInit(this, Bits[i]));
  }
  return BI;
}

Init *TypedInit::convertInitListSlice(const std::vector<unsigned> &Elements) {
  ListRecTy *T = dynamic_cast<ListRecTy*>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-list variable...

  if (Elements.size() == 1)
    return new VarListElementInit(this, Elements[0]);

  std::vector<Init*> ListInits;
  ListInits.reserve(Elements.size());
  for (unsigned i = 0, e = Elements.size(); i != e; ++i)
    ListInits.push_back(new VarListElementInit(this, Elements[i]));
  return new ListInit(ListInits, T);
}


Init *VarInit::resolveBitReference(Record &R, const RecordVal *IRV,
                                   unsigned Bit) {
  if (R.isTemplateArg(getName())) return 0;
  if (IRV && IRV->getName() != getName()) return 0;

  RecordVal *RV = R.getValue(getName());
  assert(RV && "Reference to a non-existant variable?");
  assert(dynamic_cast<BitsInit*>(RV->getValue()));
  BitsInit *BI = (BitsInit*)RV->getValue();

  assert(Bit < BI->getNumBits() && "Bit reference out of range!");
  Init *B = BI->getBit(Bit);

  if (!dynamic_cast<UnsetInit*>(B))  // If the bit is not set...
    return B;                        // Replace the VarBitInit with it.
  return 0;
}

Init *VarInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                           unsigned Elt) {
  if (R.isTemplateArg(getName())) return 0;
  if (IRV && IRV->getName() != getName()) return 0;

  RecordVal *RV = R.getValue(getName());
  assert(RV && "Reference to a non-existant variable?");
  ListInit *LI = dynamic_cast<ListInit*>(RV->getValue());
  if (!LI) {
    VarInit *VI = dynamic_cast<VarInit*>(RV->getValue());
    assert(VI && "Invalid list element!");
    return new VarListElementInit(VI, Elt);
  }
  
  if (Elt >= LI->getSize())
    return 0;  // Out of range reference.
  Init *E = LI->getElement(Elt);
  if (!dynamic_cast<UnsetInit*>(E))  // If the element is set
    return E;                        // Replace the VarListElementInit with it.
  return 0;
}


RecTy *VarInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RTy = dynamic_cast<RecordRecTy*>(getType()))
    if (const RecordVal *RV = RTy->getRecord()->getValue(FieldName))
      return RV->getType();
  return 0;
}

Init *VarInit::getFieldInit(Record &R, const std::string &FieldName) const {
  if (dynamic_cast<RecordRecTy*>(getType()))
    if (const RecordVal *RV = R.getValue(VarName)) {
      Init *TheInit = RV->getValue();
      assert(TheInit != this && "Infinite loop detected!");
      if (Init *I = TheInit->getFieldInit(R, FieldName))
        return I;
      else
        return 0;
    }
  return 0;
}

/// resolveReferences - This method is used by classes that refer to other
/// variables which may not be defined at the time they expression is formed.
/// If a value is set for the variable later, this method will be called on
/// users of the value to allow the value to propagate out.
///
Init *VarInit::resolveReferences(Record &R, const RecordVal *RV) {
  if (RecordVal *Val = R.getValue(VarName))
    if (RV == Val || (RV == 0 && !dynamic_cast<UnsetInit*>(Val->getValue())))
      return Val->getValue();
  return this;
}

std::string VarBitInit::getAsString() const {
   return TI->getAsString() + "{" + utostr(Bit) + "}";
}

Init *VarBitInit::resolveReferences(Record &R, const RecordVal *RV) {
  if (Init *I = getVariable()->resolveBitReference(R, RV, getBitNum()))
    return I;
  return this;
}

std::string VarListElementInit::getAsString() const {
  return TI->getAsString() + "[" + utostr(Element) + "]";
}

Init *VarListElementInit::resolveReferences(Record &R, const RecordVal *RV) {
  if (Init *I = getVariable()->resolveListElementReference(R, RV,
                                                           getElementNum()))
    return I;
  return this;
}

Init *VarListElementInit::resolveBitReference(Record &R, const RecordVal *RV,
                                              unsigned Bit) {
  // FIXME: This should be implemented, to support references like:
  // bit B = AA[0]{1};
  return 0;
}

Init *VarListElementInit::
resolveListElementReference(Record &R, const RecordVal *RV, unsigned Elt) {
  // FIXME: This should be implemented, to support references like:
  // int B = AA[0][1];
  return 0;
}

RecTy *DefInit::getFieldType(const std::string &FieldName) const {
  if (const RecordVal *RV = Def->getValue(FieldName))
    return RV->getType();
  return 0;
}

Init *DefInit::getFieldInit(Record &R, const std::string &FieldName) const {
  return Def->getValue(FieldName)->getValue();
}


std::string DefInit::getAsString() const {
  return Def->getName();
}

Init *FieldInit::resolveBitReference(Record &R, const RecordVal *RV,
                                     unsigned Bit) {
  if (Init *BitsVal = Rec->getFieldInit(R, FieldName))
    if (BitsInit *BI = dynamic_cast<BitsInit*>(BitsVal)) {
      assert(Bit < BI->getNumBits() && "Bit reference out of range!");
      Init *B = BI->getBit(Bit);

      if (dynamic_cast<BitInit*>(B))  // If the bit is set...
        return B;                     // Replace the VarBitInit with it.
    }
  return 0;
}

Init *FieldInit::resolveListElementReference(Record &R, const RecordVal *RV,
                                             unsigned Elt) {
  if (Init *ListVal = Rec->getFieldInit(R, FieldName))
    if (ListInit *LI = dynamic_cast<ListInit*>(ListVal)) {
      if (Elt >= LI->getSize()) return 0;
      Init *E = LI->getElement(Elt);

      if (!dynamic_cast<UnsetInit*>(E))  // If the bit is set...
        return E;                  // Replace the VarListElementInit with it.
    }
  return 0;
}

Init *FieldInit::resolveReferences(Record &R, const RecordVal *RV) {
  Init *NewRec = RV ? Rec->resolveReferences(R, RV) : Rec;

  Init *BitsVal = NewRec->getFieldInit(R, FieldName);
  if (BitsVal) {
    Init *BVR = BitsVal->resolveReferences(R, RV);
    return BVR->isComplete() ? BVR : this;
  }

  if (NewRec != Rec) {
    return new FieldInit(NewRec, FieldName);
  }
  return this;
}

Init *DagInit::resolveReferences(Record &R, const RecordVal *RV) {
  std::vector<Init*> NewArgs;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    NewArgs.push_back(Args[i]->resolveReferences(R, RV));
  
  Init *Op = Val->resolveReferences(R, RV);
  
  if (Args != NewArgs || Op != Val)
    return new DagInit(Op, "", NewArgs, ArgNames);
    
  return this;
}


std::string DagInit::getAsString() const {
  std::string Result = "(" + Val->getAsString();
  if (!ValName.empty())
    Result += ":" + ValName;
  if (Args.size()) {
    Result += " " + Args[0]->getAsString();
    if (!ArgNames[0].empty()) Result += ":$" + ArgNames[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      Result += ", " + Args[i]->getAsString();
      if (!ArgNames[i].empty()) Result += ":$" + ArgNames[i];
    }
  }
  return Result + ")";
}


//===----------------------------------------------------------------------===//
//    Other implementations
//===----------------------------------------------------------------------===//

RecordVal::RecordVal(const std::string &N, RecTy *T, unsigned P)
  : Name(N), Ty(T), Prefix(P) {
  Value = Ty->convertValue(new UnsetInit());
  assert(Value && "Cannot create unset value for current type!");
}

void RecordVal::dump() const { errs() << *this; }

void RecordVal::print(raw_ostream &OS, bool PrintSem) const {
  if (getPrefix()) OS << "field ";
  OS << *getType() << " " << getName();

  if (getValue())
    OS << " = " << *getValue();

  if (PrintSem) OS << ";\n";
}

unsigned Record::LastID = 0;

void Record::setName(const std::string &Name) {
  if (Records.getDef(getName()) == this) {
    Records.removeDef(getName());
    this->Name = Name;
    Records.addDef(this);
  } else {
    Records.removeClass(getName());
    this->Name = Name;
    Records.addClass(this);
  }
}

/// resolveReferencesTo - If anything in this record refers to RV, replace the
/// reference to RV with the RHS of RV.  If RV is null, we resolve all possible
/// references.
void Record::resolveReferencesTo(const RecordVal *RV) {
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (Init *V = Values[i].getValue())
      Values[i].setValue(V->resolveReferences(*this, RV));
  }
}


void Record::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const Record &R) {
  OS << R.getName();

  const std::vector<std::string> &TArgs = R.getTemplateArgs();
  if (!TArgs.empty()) {
    OS << "<";
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i) OS << ", ";
      const RecordVal *RV = R.getValue(TArgs[i]);
      assert(RV && "Template argument record not found??");
      RV->print(OS, false);
    }
    OS << ">";
  }

  OS << " {";
  const std::vector<Record*> &SC = R.getSuperClasses();
  if (!SC.empty()) {
    OS << "\t//";
    for (unsigned i = 0, e = SC.size(); i != e; ++i)
      OS << " " << SC[i]->getName();
  }
  OS << "\n";

  const std::vector<RecordVal> &Vals = R.getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (Vals[i].getPrefix() && !R.isTemplateArg(Vals[i].getName()))
      OS << Vals[i];
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (!Vals[i].getPrefix() && !R.isTemplateArg(Vals[i].getName()))
      OS << Vals[i];

  return OS << "}\n";
}

/// getValueInit - Return the initializer for a value with the specified name,
/// or throw an exception if the field does not exist.
///
Init *Record::getValueInit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";
  return R->getValue();
}


/// getValueAsString - This method looks up the specified field and returns its
/// value as a string, throwing an exception if the field does not exist or if
/// the value is not a string.
///
std::string Record::getValueAsString(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (const StringInit *SI = dynamic_cast<const StringInit*>(R->getValue()))
    return SI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a string initializer!";
}

/// getValueAsBitsInit - This method looks up the specified field and returns
/// its value as a BitsInit, throwing an exception if the field does not exist
/// or if the value is not the right type.
///
BitsInit *Record::getValueAsBitsInit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (BitsInit *BI = dynamic_cast<BitsInit*>(R->getValue()))
    return BI;
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a BitsInit initializer!";
}

/// getValueAsListInit - This method looks up the specified field and returns
/// its value as a ListInit, throwing an exception if the field does not exist
/// or if the value is not the right type.
///
ListInit *Record::getValueAsListInit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (ListInit *LI = dynamic_cast<ListInit*>(R->getValue()))
    return LI;
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a list initializer!";
}

/// getValueAsListOfDefs - This method looks up the specified field and returns
/// its value as a vector of records, throwing an exception if the field does
/// not exist or if the value is not the right type.
///
std::vector<Record*> 
Record::getValueAsListOfDefs(const std::string &FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<Record*> Defs;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (DefInit *DI = dynamic_cast<DefInit*>(List->getElement(i))) {
      Defs.push_back(DI->getDef());
    } else {
      throw "Record `" + getName() + "', field `" + FieldName +
            "' list is not entirely DefInit!";
    }
  }
  return Defs;
}

/// getValueAsInt - This method looks up the specified field and returns its
/// value as an int64_t, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
int64_t Record::getValueAsInt(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
          FieldName + "'!\n";

  if (IntInit *II = dynamic_cast<IntInit*>(R->getValue()))
    return II->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have an int initializer!";
}

/// getValueAsListOfInts - This method looks up the specified field and returns
/// its value as a vector of integers, throwing an exception if the field does
/// not exist or if the value is not the right type.
///
std::vector<int64_t> 
Record::getValueAsListOfInts(const std::string &FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<int64_t> Ints;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (IntInit *II = dynamic_cast<IntInit*>(List->getElement(i))) {
      Ints.push_back(II->getValue());
    } else {
      throw "Record `" + getName() + "', field `" + FieldName +
            "' does not have a list of ints initializer!";
    }
  }
  return Ints;
}

/// getValueAsDef - This method looks up the specified field and returns its
/// value as a Record, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
Record *Record::getValueAsDef(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";

  if (DefInit *DI = dynamic_cast<DefInit*>(R->getValue()))
    return DI->getDef();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a def initializer!";
}

/// getValueAsBit - This method looks up the specified field and returns its
/// value as a bit, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
bool Record::getValueAsBit(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";

  if (BitInit *BI = dynamic_cast<BitInit*>(R->getValue()))
    return BI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a bit initializer!";
}

/// getValueAsDag - This method looks up the specified field and returns its
/// value as an Dag, throwing an exception if the field does not exist or if
/// the value is not the right type.
///
DagInit *Record::getValueAsDag(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";

  if (DagInit *DI = dynamic_cast<DagInit*>(R->getValue()))
    return DI;
  throw "Record `" + getName() + "', field `" + FieldName +
        "' does not have a dag initializer!";
}

std::string Record::getValueAsCode(const std::string &FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    throw "Record `" + getName() + "' does not have a field named `" +
      FieldName + "'!\n";
  
  if (const CodeInit *CI = dynamic_cast<const CodeInit*>(R->getValue()))
    return CI->getValue();
  throw "Record `" + getName() + "', field `" + FieldName +
    "' does not have a code initializer!";
}


void MultiClass::dump() const {
  errs() << "Record:\n";
  Rec.dump();
  
  errs() << "Defs:\n";
  for (RecordVector::const_iterator r = DefPrototypes.begin(),
         rend = DefPrototypes.end();
       r != rend;
       ++r) {
    (*r)->dump();
  }
}


void RecordKeeper::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const RecordKeeper &RK) {
  OS << "------------- Classes -----------------\n";
  const std::map<std::string, Record*> &Classes = RK.getClasses();
  for (std::map<std::string, Record*>::const_iterator I = Classes.begin(),
         E = Classes.end(); I != E; ++I)
    OS << "class " << *I->second;

  OS << "------------- Defs -----------------\n";
  const std::map<std::string, Record*> &Defs = RK.getDefs();
  for (std::map<std::string, Record*>::const_iterator I = Defs.begin(),
         E = Defs.end(); I != E; ++I)
    OS << "def " << *I->second;
  return OS;
}


/// getAllDerivedDefinitions - This method returns all concrete definitions
/// that derive from the specified class name.  If a class with the specified
/// name does not exist, an error is printed and true is returned.
std::vector<Record*>
RecordKeeper::getAllDerivedDefinitions(const std::string &ClassName) const {
  Record *Class = Records.getClass(ClassName);
  if (!Class)
    throw "ERROR: Couldn't find the `" + ClassName + "' class!\n";

  std::vector<Record*> Defs;
  for (std::map<std::string, Record*>::const_iterator I = getDefs().begin(),
         E = getDefs().end(); I != E; ++I)
    if (I->second->isSubClassOf(Class))
      Defs.push_back(I->second);

  return Defs;
}

