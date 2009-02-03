//===-- MSILWriter.cpp - Library for converting LLVM code to MSIL ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to MSIL code.
//
//===----------------------------------------------------------------------===//

#include "MSILWriter.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/Passes.h"

namespace {
  // TargetMachine for the MSIL 
  struct VISIBILITY_HIDDEN MSILTarget : public TargetMachine {
    const TargetData DataLayout;       // Calculates type size & alignment

    MSILTarget(const Module &M, const std::string &FS)
      : DataLayout(&M) {}

    virtual bool WantsWholeFile() const { return true; }
    virtual bool addPassesToEmitWholeFile(PassManager &PM, raw_ostream &Out,
                                         CodeGenFileType FileType, bool Fast);

    // This class always works, but shouldn't be the default in most cases.
    static unsigned getModuleMatchQuality(const Module &M) { return 1; }

    virtual const TargetData *getTargetData() const { return &DataLayout; }
  };
}

/// MSILTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int MSILTargetMachineModule;
int MSILTargetMachineModule = 0;

static RegisterTarget<MSILTarget> X("msil", "MSIL backend");

bool MSILModule::runOnModule(Module &M) {
  ModulePtr = &M;
  TD = &getAnalysis<TargetData>();
  bool Changed = false;
  // Find named types.  
  TypeSymbolTable& Table = M.getTypeSymbolTable();
  std::set<const Type *> Types = getAnalysis<FindUsedTypes>().getTypes();
  for (TypeSymbolTable::iterator I = Table.begin(), E = Table.end(); I!=E; ) {
    if (!isa<StructType>(I->second) && !isa<OpaqueType>(I->second))
      Table.remove(I++);
    else {
      std::set<const Type *>::iterator T = Types.find(I->second);
      if (T==Types.end())
        Table.remove(I++);
      else {
        Types.erase(T);
        ++I;
      }
    }
  }
  // Find unnamed types.
  unsigned RenameCounter = 0;
  for (std::set<const Type *>::const_iterator I = Types.begin(),
       E = Types.end(); I!=E; ++I)
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      while (ModulePtr->addTypeName("unnamed$"+utostr(RenameCounter), STy))
        ++RenameCounter;
      Changed = true;
    }
  // Pointer for FunctionPass.
  UsedTypes = &getAnalysis<FindUsedTypes>().getTypes();
  return Changed;
}

char MSILModule::ID = 0;
char MSILWriter::ID = 0;

bool MSILWriter::runOnFunction(Function &F) {
  if (F.isDeclaration()) return false;
  LInfo = &getAnalysis<LoopInfo>();
  printFunction(F);
  return false;
}


bool MSILWriter::doInitialization(Module &M) {
  ModulePtr = &M;
  Mang = new Mangler(M);
  Out << ".assembly extern mscorlib {}\n";
  Out << ".assembly MSIL {}\n\n";
  Out << "// External\n";
  printExternals();
  Out << "// Declarations\n";
  printDeclarations(M.getTypeSymbolTable());
  Out << "// Definitions\n";
  printGlobalVariables();
  Out << "// Startup code\n";
  printModuleStartup();
  return false;
}


bool MSILWriter::doFinalization(Module &M) {
  delete Mang;
  return false;
}


void MSILWriter::printModuleStartup() {
  Out <<
  ".method static public int32 $MSIL_Startup() {\n"
  "\t.entrypoint\n"
  "\t.locals (native int i)\n"
  "\t.locals (native int argc)\n"
  "\t.locals (native int ptr)\n"
  "\t.locals (void* argv)\n"
  "\t.locals (string[] args)\n"
  "\tcall\tstring[] [mscorlib]System.Environment::GetCommandLineArgs()\n"
  "\tdup\n"
  "\tstloc\targs\n"
  "\tldlen\n"
  "\tconv.i4\n"
  "\tdup\n"
  "\tstloc\targc\n";
  printPtrLoad(TD->getPointerSize());
  Out <<
  "\tmul\n"
  "\tlocalloc\n"
  "\tstloc\targv\n"
  "\tldc.i4.0\n"
  "\tstloc\ti\n"
  "L_01:\n"
  "\tldloc\ti\n"
  "\tldloc\targc\n"
  "\tceq\n"
  "\tbrtrue\tL_02\n"
  "\tldloc\targs\n"
  "\tldloc\ti\n"
  "\tldelem.ref\n"
  "\tcall\tnative int [mscorlib]System.Runtime.InteropServices.Marshal::"
           "StringToHGlobalAnsi(string)\n"
  "\tstloc\tptr\n"
  "\tldloc\targv\n"
  "\tldloc\ti\n";
  printPtrLoad(TD->getPointerSize());
  Out << 
  "\tmul\n"
  "\tadd\n"
  "\tldloc\tptr\n"
  "\tstind.i\n"
  "\tldloc\ti\n"
  "\tldc.i4.1\n"
  "\tadd\n"
  "\tstloc\ti\n"
  "\tbr\tL_01\n"
  "L_02:\n"
  "\tcall void $MSIL_Init()\n";

  // Call user 'main' function.
  const Function* F = ModulePtr->getFunction("main");
  if (!F || F->isDeclaration()) {
    Out << "\tldc.i4.0\n\tret\n}\n";
    return;
  }
  bool BadSig = true;;
  std::string Args("");
  Function::const_arg_iterator Arg1,Arg2;

  switch (F->arg_size()) {
  case 0:
    BadSig = false;
    break;
  case 1:
    Arg1 = F->arg_begin();
    if (Arg1->getType()->isInteger()) {
      Out << "\tldloc\targc\n";
      Args = getTypeName(Arg1->getType());
      BadSig = false;
    }
    break;
  case 2:
    Arg1 = Arg2 = F->arg_begin(); ++Arg2;
    if (Arg1->getType()->isInteger() && 
        Arg2->getType()->getTypeID() == Type::PointerTyID) {
      Out << "\tldloc\targc\n\tldloc\targv\n";
      Args = getTypeName(Arg1->getType())+","+getTypeName(Arg2->getType());
      BadSig = false;
    }
    break;
  default:
    BadSig = true;
  }

  bool RetVoid = (F->getReturnType()->getTypeID() == Type::VoidTyID);
  if (BadSig || (!F->getReturnType()->isInteger() && !RetVoid)) {
    Out << "\tldc.i4.0\n";
  } else {
    Out << "\tcall\t" << getTypeName(F->getReturnType()) <<
      getConvModopt(F->getCallingConv()) << "main(" << Args << ")\n";
    if (RetVoid)
      Out << "\tldc.i4.0\n";
    else
      Out << "\tconv.i4\n";
  }
  Out << "\tret\n}\n";
}

bool MSILWriter::isZeroValue(const Value* V) {
  if (const Constant *C = dyn_cast<Constant>(V))
    return C->isNullValue();
  return false;
}


std::string MSILWriter::getValueName(const Value* V) {
  // Name into the quotes allow control and space characters.
  return "'"+Mang->getValueName(V)+"'";
}


std::string MSILWriter::getLabelName(const std::string& Name) {
  if (Name.find('.')!=std::string::npos) {
    std::string Tmp(Name);
    // Replace unaccepable characters in the label name.
    for (std::string::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I)
      if (*I=='.') *I = '@';
    return Tmp;
  }
  return Name;
}


std::string MSILWriter::getLabelName(const Value* V) {
  return getLabelName(Mang->getValueName(V));
}


std::string MSILWriter::getConvModopt(unsigned CallingConvID) {
  switch (CallingConvID) {
  case CallingConv::C:
  case CallingConv::Cold:
  case CallingConv::Fast:
    return "modopt([mscorlib]System.Runtime.CompilerServices.CallConvCdecl) ";
  case CallingConv::X86_FastCall:
    return "modopt([mscorlib]System.Runtime.CompilerServices.CallConvFastcall) ";
  case CallingConv::X86_StdCall:
    return "modopt([mscorlib]System.Runtime.CompilerServices.CallConvStdcall) ";
  default:
    cerr << "CallingConvID = " << CallingConvID << '\n';
    assert(0 && "Unsupported calling convention");
  }
  return ""; // Not reached
}


std::string MSILWriter::getArrayTypeName(Type::TypeID TyID, const Type* Ty) {
  std::string Tmp = "";
  const Type* ElemTy = Ty;
  assert(Ty->getTypeID()==TyID && "Invalid type passed");
  // Walk trought array element types.
  for (;;) {
    // Multidimensional array.
    if (ElemTy->getTypeID()==TyID) {
      if (const ArrayType* ATy = dyn_cast<ArrayType>(ElemTy))
        Tmp += utostr(ATy->getNumElements());
      else if (const VectorType* VTy = dyn_cast<VectorType>(ElemTy))
        Tmp += utostr(VTy->getNumElements());
      ElemTy = cast<SequentialType>(ElemTy)->getElementType();
    }
    // Base element type found.
    if (ElemTy->getTypeID()!=TyID) break;
    Tmp += ",";
  }
  return getTypeName(ElemTy, false, true)+"["+Tmp+"]";
}


std::string MSILWriter::getPrimitiveTypeName(const Type* Ty, bool isSigned) {
  unsigned NumBits = 0;
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:
    return "void ";
  case Type::IntegerTyID:
    NumBits = getBitWidth(Ty);
    if(NumBits==1)
      return "bool ";
    if (!isSigned)
      return "unsigned int"+utostr(NumBits)+" ";
    return "int"+utostr(NumBits)+" ";
  case Type::FloatTyID:
    return "float32 ";
  case Type::DoubleTyID:
    return "float64 "; 
  default:
    cerr << "Type = " << *Ty << '\n';
    assert(0 && "Invalid primitive type");
  }
  return ""; // Not reached
}


std::string MSILWriter::getTypeName(const Type* Ty, bool isSigned,
                                    bool isNested) {
  if (Ty->isPrimitiveType() || Ty->isInteger())
    return getPrimitiveTypeName(Ty,isSigned);
  // FIXME: "OpaqueType" support
  switch (Ty->getTypeID()) {
  case Type::PointerTyID:
    return "void* ";
  case Type::StructTyID:
    if (isNested)
      return ModulePtr->getTypeName(Ty);
    return "valuetype '"+ModulePtr->getTypeName(Ty)+"' ";
  case Type::ArrayTyID:
    if (isNested)
      return getArrayTypeName(Ty->getTypeID(),Ty);
    return "valuetype '"+getArrayTypeName(Ty->getTypeID(),Ty)+"' ";
  case Type::VectorTyID:
    if (isNested)
      return getArrayTypeName(Ty->getTypeID(),Ty);
    return "valuetype '"+getArrayTypeName(Ty->getTypeID(),Ty)+"' ";
  default:
    cerr << "Type = " << *Ty << '\n';
    assert(0 && "Invalid type in getTypeName()");
  }
  return ""; // Not reached
}


MSILWriter::ValueType MSILWriter::getValueLocation(const Value* V) {
  // Function argument
  if (isa<Argument>(V))
    return ArgumentVT;
  // Function
  else if (const Function* F = dyn_cast<Function>(V))
    return F->hasLocalLinkage() ? InternalVT : GlobalVT;
  // Variable
  else if (const GlobalVariable* G = dyn_cast<GlobalVariable>(V))
    return G->hasLocalLinkage() ? InternalVT : GlobalVT;
  // Constant
  else if (isa<Constant>(V))
    return isa<ConstantExpr>(V) ? ConstExprVT : ConstVT;
  // Local variable
  return LocalVT;
}


std::string MSILWriter::getTypePostfix(const Type* Ty, bool Expand,
                                       bool isSigned) {
  unsigned NumBits = 0;
  switch (Ty->getTypeID()) {
  // Integer constant, expanding for stack operations.
  case Type::IntegerTyID:
    NumBits = getBitWidth(Ty);
    // Expand integer value to "int32" or "int64".
    if (Expand) return (NumBits<=32 ? "i4" : "i8");
    if (NumBits==1) return "i1";
    return (isSigned ? "i" : "u")+utostr(NumBits/8);
  // Float constant.
  case Type::FloatTyID:
    return "r4";
  case Type::DoubleTyID:
    return "r8";
  case Type::PointerTyID:
    return "i"+utostr(TD->getTypePaddedSize(Ty));
  default:
    cerr << "TypeID = " << Ty->getTypeID() << '\n';
    assert(0 && "Invalid type in TypeToPostfix()");
  }
  return ""; // Not reached
}


void MSILWriter::printConvToPtr() {
  switch (ModulePtr->getPointerSize()) {
  case Module::Pointer32:
    printSimpleInstruction("conv.u4");
    break;
  case Module::Pointer64:
    printSimpleInstruction("conv.u8");
    break;
  default:
    assert(0 && "Module use not supporting pointer size");
  }
}


void MSILWriter::printPtrLoad(uint64_t N) {
  switch (ModulePtr->getPointerSize()) {
  case Module::Pointer32:
    printSimpleInstruction("ldc.i4",utostr(N).c_str());
    // FIXME: Need overflow test?
    if (!isUInt32(N)) {
      cerr << "Value = " << utostr(N) << '\n';
      assert(0 && "32-bit pointer overflowed");
    }
    break;
  case Module::Pointer64:
    printSimpleInstruction("ldc.i8",utostr(N).c_str());
    break;
  default:
    assert(0 && "Module use not supporting pointer size");
  }
}


void MSILWriter::printValuePtrLoad(const Value* V) {
  printValueLoad(V);
  printConvToPtr();
}


void MSILWriter::printConstLoad(const Constant* C) {
  if (const ConstantInt* CInt = dyn_cast<ConstantInt>(C)) {
    // Integer constant
    Out << "\tldc." << getTypePostfix(C->getType(),true) << '\t';
    if (CInt->isMinValue(true))
      Out << CInt->getSExtValue();
    else
      Out << CInt->getZExtValue();
  } else if (const ConstantFP* FP = dyn_cast<ConstantFP>(C)) {
    // Float constant
    uint64_t X;
    unsigned Size;
    if (FP->getType()->getTypeID()==Type::FloatTyID) {
      X = (uint32_t)FP->getValueAPF().bitcastToAPInt().getZExtValue();
      Size = 4;  
    } else {
      X = FP->getValueAPF().bitcastToAPInt().getZExtValue();
      Size = 8;  
    }
    Out << "\tldc.r" << Size << "\t( " << utohexstr(X) << ')';
  } else if (isa<UndefValue>(C)) {
    // Undefined constant value = NULL.
    printPtrLoad(0);
  } else {
    cerr << "Constant = " << *C << '\n';
    assert(0 && "Invalid constant value");
  }
  Out << '\n';
}


void MSILWriter::printValueLoad(const Value* V) {
  MSILWriter::ValueType Location = getValueLocation(V);
  switch (Location) {
  // Global variable or function address.
  case GlobalVT:
  case InternalVT:
    if (const Function* F = dyn_cast<Function>(V)) {
      std::string Name = getConvModopt(F->getCallingConv())+getValueName(F);
      printSimpleInstruction("ldftn",
        getCallSignature(F->getFunctionType(),NULL,Name).c_str());
    } else {
      std::string Tmp;
      const Type* ElemTy = cast<PointerType>(V->getType())->getElementType();
      if (Location==GlobalVT && cast<GlobalVariable>(V)->hasDLLImportLinkage()) {
        Tmp = "void* "+getValueName(V);
        printSimpleInstruction("ldsfld",Tmp.c_str());
      } else {
        Tmp = getTypeName(ElemTy)+getValueName(V);
        printSimpleInstruction("ldsflda",Tmp.c_str());
      }
    }
    break;
  // Function argument.
  case ArgumentVT:
    printSimpleInstruction("ldarg",getValueName(V).c_str());
    break;
  // Local function variable.
  case LocalVT:
    printSimpleInstruction("ldloc",getValueName(V).c_str());
    break;
  // Constant value.
  case ConstVT:
    if (isa<ConstantPointerNull>(V))
      printPtrLoad(0);
    else
      printConstLoad(cast<Constant>(V));
    break;
  // Constant expression.
  case ConstExprVT:
    printConstantExpr(cast<ConstantExpr>(V));
    break;
  default:
    cerr << "Value = " << *V << '\n';
    assert(0 && "Invalid value location");
  }
}


void MSILWriter::printValueSave(const Value* V) {
  switch (getValueLocation(V)) {
  case ArgumentVT:
    printSimpleInstruction("starg",getValueName(V).c_str());
    break;
  case LocalVT:
    printSimpleInstruction("stloc",getValueName(V).c_str());
    break;
  default:
    cerr << "Value  = " << *V << '\n';
    assert(0 && "Invalid value location");
  }
}


void MSILWriter::printBinaryInstruction(const char* Name, const Value* Left,
                                        const Value* Right) {
  printValueLoad(Left);
  printValueLoad(Right);
  Out << '\t' << Name << '\n';
}


void MSILWriter::printSimpleInstruction(const char* Inst, const char* Operand) {
  if(Operand) 
    Out << '\t' << Inst << '\t' << Operand << '\n';
  else
    Out << '\t' << Inst << '\n';
}


void MSILWriter::printPHICopy(const BasicBlock* Src, const BasicBlock* Dst) {
  for (BasicBlock::const_iterator I = Dst->begin(), E = Dst->end();
       isa<PHINode>(I); ++I) {
    const PHINode* Phi = cast<PHINode>(I);
    const Value* Val = Phi->getIncomingValueForBlock(Src);
    if (isa<UndefValue>(Val)) continue;
    printValueLoad(Val);
    printValueSave(Phi);
  }
}


void MSILWriter::printBranchToBlock(const BasicBlock* CurrBB,
                                    const BasicBlock* TrueBB,
                                    const BasicBlock* FalseBB) {
  if (TrueBB==FalseBB) {
    // "TrueBB" and "FalseBB" destination equals
    printPHICopy(CurrBB,TrueBB);
    printSimpleInstruction("pop");
    printSimpleInstruction("br",getLabelName(TrueBB).c_str());
  } else if (FalseBB==NULL) {
    // If "FalseBB" not used the jump have condition
    printPHICopy(CurrBB,TrueBB);
    printSimpleInstruction("brtrue",getLabelName(TrueBB).c_str());
  } else if (TrueBB==NULL) {
    // If "TrueBB" not used the jump is unconditional
    printPHICopy(CurrBB,FalseBB);
    printSimpleInstruction("br",getLabelName(FalseBB).c_str());
  } else {
    // Copy PHI instructions for each block
    std::string TmpLabel;
    // Print PHI instructions for "TrueBB"
    if (isa<PHINode>(TrueBB->begin())) {
      TmpLabel = getLabelName(TrueBB)+"$phi_"+utostr(getUniqID());
      printSimpleInstruction("brtrue",TmpLabel.c_str());
    } else {
      printSimpleInstruction("brtrue",getLabelName(TrueBB).c_str());
    }
    // Print PHI instructions for "FalseBB"
    if (isa<PHINode>(FalseBB->begin())) {
      printPHICopy(CurrBB,FalseBB);
      printSimpleInstruction("br",getLabelName(FalseBB).c_str());
    } else {
      printSimpleInstruction("br",getLabelName(FalseBB).c_str());
    }
    if (isa<PHINode>(TrueBB->begin())) {
      // Handle "TrueBB" PHI Copy
      Out << TmpLabel << ":\n";
      printPHICopy(CurrBB,TrueBB);
      printSimpleInstruction("br",getLabelName(TrueBB).c_str());
    }
  }
}


void MSILWriter::printBranchInstruction(const BranchInst* Inst) {
  if (Inst->isUnconditional()) {
    printBranchToBlock(Inst->getParent(),NULL,Inst->getSuccessor(0));
  } else {
    printValueLoad(Inst->getCondition());
    printBranchToBlock(Inst->getParent(),Inst->getSuccessor(0),
                       Inst->getSuccessor(1));
  }
}


void MSILWriter::printSelectInstruction(const Value* Cond, const Value* VTrue,
                                        const Value* VFalse) {
  std::string TmpLabel = std::string("select$true_")+utostr(getUniqID());
  printValueLoad(VTrue);
  printValueLoad(Cond);
  printSimpleInstruction("brtrue",TmpLabel.c_str());
  printSimpleInstruction("pop");
  printValueLoad(VFalse);
  Out << TmpLabel << ":\n";
}


void MSILWriter::printIndirectLoad(const Value* V) {
  const Type* Ty = V->getType();
  printValueLoad(V);
  if (const PointerType* P = dyn_cast<PointerType>(Ty))
    Ty = P->getElementType();
  std::string Tmp = "ldind."+getTypePostfix(Ty, false);
  printSimpleInstruction(Tmp.c_str());
}


void MSILWriter::printIndirectSave(const Value* Ptr, const Value* Val) {
  printValueLoad(Ptr);
  printValueLoad(Val);
  printIndirectSave(Val->getType());
}


void MSILWriter::printIndirectSave(const Type* Ty) {
  // Instruction need signed postfix for any type.
  std::string postfix = getTypePostfix(Ty, false);
  if (*postfix.begin()=='u') *postfix.begin() = 'i';
  postfix = "stind."+postfix;
  printSimpleInstruction(postfix.c_str());
}


void MSILWriter::printCastInstruction(unsigned int Op, const Value* V,
                                      const Type* Ty) {
  std::string Tmp("");
  printValueLoad(V);
  switch (Op) {
  // Signed
  case Instruction::SExt:
  case Instruction::SIToFP:
  case Instruction::FPToSI:
    Tmp = "conv."+getTypePostfix(Ty,false,true);
    printSimpleInstruction(Tmp.c_str());
    break;
  // Unsigned
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::FPToUI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
    Tmp = "conv."+getTypePostfix(Ty,false);
    printSimpleInstruction(Tmp.c_str());
    break;
  // Do nothing
  case Instruction::BitCast:
    // FIXME: meaning that ld*/st* instruction do not change data format.
    break;
  default:
    cerr << "Opcode = " << Op << '\n';
    assert(0 && "Invalid conversion instruction");
  }
}


void MSILWriter::printGepInstruction(const Value* V, gep_type_iterator I,
                                     gep_type_iterator E) {
  unsigned Size;
  // Load address
  printValuePtrLoad(V);
  // Calculate element offset.
  for (; I!=E; ++I){
    Size = 0;
    const Value* IndexValue = I.getOperand();
    if (const StructType* StrucTy = dyn_cast<StructType>(*I)) {
      uint64_t FieldIndex = cast<ConstantInt>(IndexValue)->getZExtValue();
      // Offset is the sum of all previous structure fields.
      for (uint64_t F = 0; F<FieldIndex; ++F)
        Size += TD->getTypePaddedSize(StrucTy->getContainedType((unsigned)F));
      printPtrLoad(Size);
      printSimpleInstruction("add");
      continue;
    } else if (const SequentialType* SeqTy = dyn_cast<SequentialType>(*I)) {
      Size = TD->getTypePaddedSize(SeqTy->getElementType());
    } else {
      Size = TD->getTypePaddedSize(*I);
    }
    // Add offset of current element to stack top.
    if (!isZeroValue(IndexValue)) {
      // Constant optimization.
      if (const ConstantInt* C = dyn_cast<ConstantInt>(IndexValue)) {
        if (C->getValue().isNegative()) {
          printPtrLoad(C->getValue().abs().getZExtValue()*Size);
          printSimpleInstruction("sub");
          continue;
        } else
          printPtrLoad(C->getZExtValue()*Size);
      } else {
        printPtrLoad(Size);
        printValuePtrLoad(IndexValue);
        printSimpleInstruction("mul");
      }
      printSimpleInstruction("add");
    }
  }
}


std::string MSILWriter::getCallSignature(const FunctionType* Ty,
                                         const Instruction* Inst,
                                         std::string Name) {
  std::string Tmp("");
  if (Ty->isVarArg()) Tmp += "vararg ";
  // Name and return type.
  Tmp += getTypeName(Ty->getReturnType())+Name+"(";
  // Function argument type list.
  unsigned NumParams = Ty->getNumParams();
  for (unsigned I = 0; I!=NumParams; ++I) {
    if (I!=0) Tmp += ",";
    Tmp += getTypeName(Ty->getParamType(I));
  }
  // CLR needs to know the exact amount of parameters received by vararg
  // function, because caller cleans the stack.
  if (Ty->isVarArg() && Inst) {
    // Origin to function arguments in "CallInst" or "InvokeInst".
    unsigned Org = isa<InvokeInst>(Inst) ? 3 : 1;
    // Print variable argument types.
    unsigned NumOperands = Inst->getNumOperands()-Org;
    if (NumParams<NumOperands) {
      if (NumParams!=0) Tmp += ", ";
      Tmp += "... , ";
      for (unsigned J = NumParams; J!=NumOperands; ++J) {
        if (J!=NumParams) Tmp += ", ";
        Tmp += getTypeName(Inst->getOperand(J+Org)->getType());
      }
    }
  }
  return Tmp+")";
}


void MSILWriter::printFunctionCall(const Value* FnVal,
                                   const Instruction* Inst) {
  // Get function calling convention.
  std::string Name = "";
  if (const CallInst* Call = dyn_cast<CallInst>(Inst))
    Name = getConvModopt(Call->getCallingConv());
  else if (const InvokeInst* Invoke = dyn_cast<InvokeInst>(Inst))
    Name = getConvModopt(Invoke->getCallingConv());
  else {
    cerr << "Instruction = " << Inst->getName() << '\n';
    assert(0 && "Need \"Invoke\" or \"Call\" instruction only");
  }
  if (const Function* F = dyn_cast<Function>(FnVal)) {
    // Direct call.
    Name += getValueName(F);
    printSimpleInstruction("call",
      getCallSignature(F->getFunctionType(),Inst,Name).c_str());
  } else {
    // Indirect function call.
    const PointerType* PTy = cast<PointerType>(FnVal->getType());
    const FunctionType* FTy = cast<FunctionType>(PTy->getElementType());
    // Load function address.
    printValueLoad(FnVal);
    printSimpleInstruction("calli",getCallSignature(FTy,Inst,Name).c_str());
  }
}


void MSILWriter::printIntrinsicCall(const IntrinsicInst* Inst) {
  std::string Name;
  switch (Inst->getIntrinsicID()) {
  case Intrinsic::vastart:
    Name = getValueName(Inst->getOperand(1));
    Name.insert(Name.length()-1,"$valist");
    // Obtain the argument handle.
    printSimpleInstruction("ldloca",Name.c_str());
    printSimpleInstruction("arglist");
    printSimpleInstruction("call",
      "instance void [mscorlib]System.ArgIterator::.ctor"
      "(valuetype [mscorlib]System.RuntimeArgumentHandle)");
    // Save as pointer type "void*"
    printValueLoad(Inst->getOperand(1));
    printSimpleInstruction("ldloca",Name.c_str());
    printIndirectSave(PointerType::getUnqual(IntegerType::get(8)));
    break;
  case Intrinsic::vaend:
    // Close argument list handle.
    printIndirectLoad(Inst->getOperand(1));
    printSimpleInstruction("call","instance void [mscorlib]System.ArgIterator::End()");
    break;
  case Intrinsic::vacopy:
    // Copy "ArgIterator" valuetype.
    printIndirectLoad(Inst->getOperand(1));
    printIndirectLoad(Inst->getOperand(2));
    printSimpleInstruction("cpobj","[mscorlib]System.ArgIterator");
    break;        
  default:
    cerr << "Intrinsic ID = " << Inst->getIntrinsicID() << '\n';
    assert(0 && "Invalid intrinsic function");
  }
}


void MSILWriter::printCallInstruction(const Instruction* Inst) {
  if (isa<IntrinsicInst>(Inst)) {
    // Handle intrinsic function.
    printIntrinsicCall(cast<IntrinsicInst>(Inst));
  } else {
    // Load arguments to stack and call function.
    for (int I = 1, E = Inst->getNumOperands(); I!=E; ++I)
      printValueLoad(Inst->getOperand(I));
    printFunctionCall(Inst->getOperand(0),Inst);
  }
}


void MSILWriter::printICmpInstruction(unsigned Predicate, const Value* Left,
                                      const Value* Right) {
  switch (Predicate) {
  case ICmpInst::ICMP_EQ:
    printBinaryInstruction("ceq",Left,Right);
    break;
  case ICmpInst::ICMP_NE:
    // Emulate = not neg (Op1 eq Op2)
    printBinaryInstruction("ceq",Left,Right);
    printSimpleInstruction("neg");
    printSimpleInstruction("not");
    break;
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE:
    // Emulate = (Op1 eq Op2) or (Op1 lt Op2)
    printBinaryInstruction("ceq",Left,Right);
    if (Predicate==ICmpInst::ICMP_ULE)
      printBinaryInstruction("clt.un",Left,Right);
    else
      printBinaryInstruction("clt",Left,Right);
    printSimpleInstruction("or");
    break;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
    // Emulate = (Op1 eq Op2) or (Op1 gt Op2)
    printBinaryInstruction("ceq",Left,Right);
    if (Predicate==ICmpInst::ICMP_UGE)
      printBinaryInstruction("cgt.un",Left,Right);
    else
      printBinaryInstruction("cgt",Left,Right);
    printSimpleInstruction("or");
    break;
  case ICmpInst::ICMP_ULT:
    printBinaryInstruction("clt.un",Left,Right);
    break;
  case ICmpInst::ICMP_SLT:
    printBinaryInstruction("clt",Left,Right);
    break;
  case ICmpInst::ICMP_UGT:
    printBinaryInstruction("cgt.un",Left,Right);
  case ICmpInst::ICMP_SGT:
    printBinaryInstruction("cgt",Left,Right);
    break;
  default:
    cerr << "Predicate = " << Predicate << '\n';
    assert(0 && "Invalid icmp predicate");
  }
}


void MSILWriter::printFCmpInstruction(unsigned Predicate, const Value* Left,
                                      const Value* Right) {
  // FIXME: Correct comparison
  std::string NanFunc = "bool [mscorlib]System.Double::IsNaN(float64)";
  switch (Predicate) {
  case FCmpInst::FCMP_UGT:
    // X >  Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("cgt",Left,Right);
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OGT:
    // X >  Y
    printBinaryInstruction("cgt",Left,Right);
    break;
  case FCmpInst::FCMP_UGE:
    // X >= Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("cgt",Left,Right);
    printSimpleInstruction("or");
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OGE:
    // X >= Y
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("cgt",Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_ULT:
    // X <  Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("clt",Left,Right);
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OLT:
    // X <  Y
    printBinaryInstruction("clt",Left,Right);
    break;
  case FCmpInst::FCMP_ULE:
    // X <= Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("clt",Left,Right);
    printSimpleInstruction("or");
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OLE:
    // X <= Y
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("clt",Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_UEQ:
    // X == Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OEQ:
    // X == Y
    printBinaryInstruction("ceq",Left,Right);
    break;
  case FCmpInst::FCMP_UNE:
    // X != Y
    printBinaryInstruction("ceq",Left,Right);
    printSimpleInstruction("neg");
    printSimpleInstruction("not");
    break;
  case FCmpInst::FCMP_ONE:
    // X != Y && llvm_fcmp_ord(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printSimpleInstruction("not");
    break;
  case FCmpInst::FCMP_ORD:
    // return X == X && Y == Y
    printBinaryInstruction("ceq",Left,Left);
    printBinaryInstruction("ceq",Right,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_UNO:
    // X != X || Y != Y
    printBinaryInstruction("ceq",Left,Left);
    printSimpleInstruction("not");
    printBinaryInstruction("ceq",Right,Right);
    printSimpleInstruction("not");
    printSimpleInstruction("or");
    break;
  default:
    assert(0 && "Illegal FCmp predicate");
  }
}


void MSILWriter::printInvokeInstruction(const InvokeInst* Inst) {
  std::string Label = "leave$normal_"+utostr(getUniqID());
  Out << ".try {\n";
  // Load arguments
  for (int I = 3, E = Inst->getNumOperands(); I!=E; ++I)
    printValueLoad(Inst->getOperand(I));
  // Print call instruction
  printFunctionCall(Inst->getOperand(0),Inst);
  // Save function result and leave "try" block
  printValueSave(Inst);
  printSimpleInstruction("leave",Label.c_str());
  Out << "}\n";
  Out << "catch [mscorlib]System.Exception {\n";
  // Redirect to unwind block
  printSimpleInstruction("pop");
  printBranchToBlock(Inst->getParent(),NULL,Inst->getUnwindDest());
  Out << "}\n" << Label << ":\n";
  // Redirect to continue block
  printBranchToBlock(Inst->getParent(),NULL,Inst->getNormalDest());
}


void MSILWriter::printSwitchInstruction(const SwitchInst* Inst) {
  // FIXME: Emulate with IL "switch" instruction
  // Emulate = if () else if () else if () else ...
  for (unsigned int I = 1, E = Inst->getNumCases(); I!=E; ++I) {
    printValueLoad(Inst->getCondition());
    printValueLoad(Inst->getCaseValue(I));
    printSimpleInstruction("ceq");
    // Condition jump to successor block
    printBranchToBlock(Inst->getParent(),Inst->getSuccessor(I),NULL);
  }
  // Jump to default block
  printBranchToBlock(Inst->getParent(),NULL,Inst->getDefaultDest());
}


void MSILWriter::printVAArgInstruction(const VAArgInst* Inst) {
  printIndirectLoad(Inst->getOperand(0));
  printSimpleInstruction("call",
    "instance typedref [mscorlib]System.ArgIterator::GetNextArg()");
  printSimpleInstruction("refanyval","void*");
  std::string Name = 
    "ldind."+getTypePostfix(PointerType::getUnqual(IntegerType::get(8)),false);
  printSimpleInstruction(Name.c_str());
}


void MSILWriter::printAllocaInstruction(const AllocaInst* Inst) {
  uint64_t Size = TD->getTypePaddedSize(Inst->getAllocatedType());
  // Constant optimization.
  if (const ConstantInt* CInt = dyn_cast<ConstantInt>(Inst->getOperand(0))) {
    printPtrLoad(CInt->getZExtValue()*Size);
  } else {
    printPtrLoad(Size);
    printValueLoad(Inst->getOperand(0));
    printSimpleInstruction("mul");
  }
  printSimpleInstruction("localloc");
}


void MSILWriter::printInstruction(const Instruction* Inst) {
  const Value *Left = 0, *Right = 0;
  if (Inst->getNumOperands()>=1) Left = Inst->getOperand(0);
  if (Inst->getNumOperands()>=2) Right = Inst->getOperand(1);
  // Print instruction
  // FIXME: "ShuffleVector","ExtractElement","InsertElement" support.
  switch (Inst->getOpcode()) {
  // Terminator
  case Instruction::Ret:
    if (Inst->getNumOperands()) {
      printValueLoad(Left);
      printSimpleInstruction("ret");
    } else
      printSimpleInstruction("ret");
    break;
  case Instruction::Br:
    printBranchInstruction(cast<BranchInst>(Inst));
    break;
  // Binary
  case Instruction::Add:
    printBinaryInstruction("add",Left,Right);
    break;
  case Instruction::Sub:
    printBinaryInstruction("sub",Left,Right);
    break;
  case Instruction::Mul:  
    printBinaryInstruction("mul",Left,Right);
    break;
  case Instruction::UDiv:
    printBinaryInstruction("div.un",Left,Right);
    break;
  case Instruction::SDiv:
  case Instruction::FDiv:
    printBinaryInstruction("div",Left,Right);
    break;
  case Instruction::URem:
    printBinaryInstruction("rem.un",Left,Right);
    break;
  case Instruction::SRem:
  case Instruction::FRem:
    printBinaryInstruction("rem",Left,Right);
    break;
  // Binary Condition
  case Instruction::ICmp:
    printICmpInstruction(cast<ICmpInst>(Inst)->getPredicate(),Left,Right);
    break;
  case Instruction::FCmp:
    printFCmpInstruction(cast<FCmpInst>(Inst)->getPredicate(),Left,Right);
    break;
  // Bitwise Binary
  case Instruction::And:
    printBinaryInstruction("and",Left,Right);
    break;
  case Instruction::Or:
    printBinaryInstruction("or",Left,Right);
    break;
  case Instruction::Xor:
    printBinaryInstruction("xor",Left,Right);
    break;
  case Instruction::Shl:
    printValueLoad(Left);
    printValueLoad(Right);
    printSimpleInstruction("conv.i4");
    printSimpleInstruction("shl");
    break;
  case Instruction::LShr:
    printValueLoad(Left);
    printValueLoad(Right);
    printSimpleInstruction("conv.i4");
    printSimpleInstruction("shr.un");
    break;
  case Instruction::AShr:
    printValueLoad(Left);
    printValueLoad(Right);
    printSimpleInstruction("conv.i4");
    printSimpleInstruction("shr");
    break;
  case Instruction::Select:
    printSelectInstruction(Inst->getOperand(0),Inst->getOperand(1),Inst->getOperand(2));
    break;
  case Instruction::Load:
    printIndirectLoad(Inst->getOperand(0));
    break;
  case Instruction::Store:
    printIndirectSave(Inst->getOperand(1), Inst->getOperand(0));
    break;
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    printCastInstruction(Inst->getOpcode(),Left,
                         cast<CastInst>(Inst)->getDestTy());
    break;
  case Instruction::GetElementPtr:
    printGepInstruction(Inst->getOperand(0),gep_type_begin(Inst),
                        gep_type_end(Inst));
    break;
  case Instruction::Call:
    printCallInstruction(cast<CallInst>(Inst));
    break;
  case Instruction::Invoke:
    printInvokeInstruction(cast<InvokeInst>(Inst));
    break;
  case Instruction::Unwind:
    printSimpleInstruction("newobj",
      "instance void [mscorlib]System.Exception::.ctor()");
    printSimpleInstruction("throw");
    break;
  case Instruction::Switch:
    printSwitchInstruction(cast<SwitchInst>(Inst));
    break;
  case Instruction::Alloca:
    printAllocaInstruction(cast<AllocaInst>(Inst));
    break;
  case Instruction::Malloc:
    assert(0 && "LowerAllocationsPass used");
    break;
  case Instruction::Free:
    assert(0 && "LowerAllocationsPass used");
    break;
  case Instruction::Unreachable:
    printSimpleInstruction("ldstr", "\"Unreachable instruction\"");
    printSimpleInstruction("newobj",
      "instance void [mscorlib]System.Exception::.ctor(string)");
    printSimpleInstruction("throw");
    break;
  case Instruction::VAArg:
    printVAArgInstruction(cast<VAArgInst>(Inst));
    break;
  default:
    cerr << "Instruction = " << Inst->getName() << '\n';
    assert(0 && "Unsupported instruction");
  }
}


void MSILWriter::printLoop(const Loop* L) {
  Out << getLabelName(L->getHeader()->getName()) << ":\n";
  const std::vector<BasicBlock*>& blocks = L->getBlocks();
  for (unsigned I = 0, E = blocks.size(); I!=E; I++) {
    BasicBlock* BB = blocks[I];
    Loop* BBLoop = LInfo->getLoopFor(BB);
    if (BBLoop == L)
      printBasicBlock(BB);
    else if (BB==BBLoop->getHeader() && BBLoop->getParentLoop()==L)
      printLoop(BBLoop);
  }
  printSimpleInstruction("br",getLabelName(L->getHeader()->getName()).c_str());
}


void MSILWriter::printBasicBlock(const BasicBlock* BB) {
  Out << getLabelName(BB) << ":\n";
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
    const Instruction* Inst = I;
    // Comment llvm original instruction
    // Out << "\n//" << *Inst << "\n";
    // Do not handle PHI instruction in current block
    if (Inst->getOpcode()==Instruction::PHI) continue;
    // Print instruction
    printInstruction(Inst);
    // Save result
    if (Inst->getType()!=Type::VoidTy) {
      // Do not save value after invoke, it done in "try" block
      if (Inst->getOpcode()==Instruction::Invoke) continue;
      printValueSave(Inst);
    }
  }
}


void MSILWriter::printLocalVariables(const Function& F) {
  std::string Name;
  const Type* Ty = NULL;
  std::set<const Value*> Printed;
  const Value* VaList = NULL;
  unsigned StackDepth = 8;
  // Find local variables
  for (const_inst_iterator I = inst_begin(&F), E = inst_end(&F); I!=E; ++I) {
    if (I->getOpcode()==Instruction::Call ||
        I->getOpcode()==Instruction::Invoke) {
      // Test stack depth.
      if (StackDepth<I->getNumOperands())
        StackDepth = I->getNumOperands();
    }
    const AllocaInst* AI = dyn_cast<AllocaInst>(&*I);
    if (AI && !isa<GlobalVariable>(AI)) {
      // Local variable allocation.
      Ty = PointerType::getUnqual(AI->getAllocatedType());
      Name = getValueName(AI);
      Out << "\t.locals (" << getTypeName(Ty) << Name << ")\n";
    } else if (I->getType()!=Type::VoidTy) {
      // Operation result.
      Ty = I->getType();
      Name = getValueName(&*I);
      Out << "\t.locals (" << getTypeName(Ty) << Name << ")\n";
    }
    // Test on 'va_list' variable    
    bool isVaList = false;     
    if (const VAArgInst* VaInst = dyn_cast<VAArgInst>(&*I)) {
      // "va_list" as "va_arg" instruction operand.
      isVaList = true;
      VaList = VaInst->getOperand(0);
    } else if (const IntrinsicInst* Inst = dyn_cast<IntrinsicInst>(&*I)) {
      // "va_list" as intrinsic function operand. 
      switch (Inst->getIntrinsicID()) {
      case Intrinsic::vastart:
      case Intrinsic::vaend:
      case Intrinsic::vacopy:
        isVaList = true;
        VaList = Inst->getOperand(1);
        break;
      default:
        isVaList = false;
      }
    }
    // Print "va_list" variable.
    if (isVaList && Printed.insert(VaList).second) {
      Name = getValueName(VaList);
      Name.insert(Name.length()-1,"$valist");
      Out << "\t.locals (valuetype [mscorlib]System.ArgIterator "
          << Name << ")\n";
    }
  }
  printSimpleInstruction(".maxstack",utostr(StackDepth*2).c_str());
}


void MSILWriter::printFunctionBody(const Function& F) {
  // Print body
  for (Function::const_iterator I = F.begin(), E = F.end(); I!=E; ++I) {
    if (Loop *L = LInfo->getLoopFor(I)) {
      if (L->getHeader()==I && L->getParentLoop()==0)
        printLoop(L);
    } else {
      printBasicBlock(I);
    }
  }
}


void MSILWriter::printConstantExpr(const ConstantExpr* CE) {
  const Value *left = 0, *right = 0;
  if (CE->getNumOperands()>=1) left = CE->getOperand(0);
  if (CE->getNumOperands()>=2) right = CE->getOperand(1);
  // Print instruction
  switch (CE->getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    printCastInstruction(CE->getOpcode(),left,CE->getType());
    break;
  case Instruction::GetElementPtr:
    printGepInstruction(CE->getOperand(0),gep_type_begin(CE),gep_type_end(CE));
    break;
  case Instruction::ICmp:
    printICmpInstruction(CE->getPredicate(),left,right);
    break;
  case Instruction::FCmp:
    printFCmpInstruction(CE->getPredicate(),left,right);
    break;
  case Instruction::Select:
    printSelectInstruction(CE->getOperand(0),CE->getOperand(1),CE->getOperand(2));
    break;
  case Instruction::Add:
    printBinaryInstruction("add",left,right);
    break;
  case Instruction::Sub:
    printBinaryInstruction("sub",left,right);
    break;
  case Instruction::Mul:
    printBinaryInstruction("mul",left,right);
    break;
  case Instruction::UDiv:
    printBinaryInstruction("div.un",left,right);
    break;
  case Instruction::SDiv:
  case Instruction::FDiv:
    printBinaryInstruction("div",left,right);
    break;
  case Instruction::URem:
    printBinaryInstruction("rem.un",left,right);
    break;
  case Instruction::SRem:
  case Instruction::FRem:
    printBinaryInstruction("rem",left,right);
    break;
  case Instruction::And:
    printBinaryInstruction("and",left,right);
    break;
  case Instruction::Or:
    printBinaryInstruction("or",left,right);
    break;
  case Instruction::Xor:
    printBinaryInstruction("xor",left,right);
    break;
  case Instruction::Shl:
    printBinaryInstruction("shl",left,right);
    break;
  case Instruction::LShr:
    printBinaryInstruction("shr.un",left,right);
    break;
  case Instruction::AShr:
    printBinaryInstruction("shr",left,right);
    break;
  default:
    cerr << "Expression = " << *CE << "\n";
    assert(0 && "Invalid constant expression");
  }
}


void MSILWriter::printStaticInitializerList() {
  // List of global variables with uninitialized fields.
  for (std::map<const GlobalVariable*,std::vector<StaticInitializer> >::iterator
       VarI = StaticInitList.begin(), VarE = StaticInitList.end(); VarI!=VarE;
       ++VarI) {
    const std::vector<StaticInitializer>& InitList = VarI->second;
    if (InitList.empty()) continue;
    // For each uninitialized field.
    for (std::vector<StaticInitializer>::const_iterator I = InitList.begin(),
         E = InitList.end(); I!=E; ++I) {
      if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(I->constant)) {
        // Out << "\n// Init " << getValueName(VarI->first) << ", offset " <<
        //  utostr(I->offset) << ", type "<< *I->constant->getType() << "\n\n";
        // Load variable address
        printValueLoad(VarI->first);
        // Add offset
        if (I->offset!=0) {
          printPtrLoad(I->offset);
          printSimpleInstruction("add");
        }
        // Load value
        printConstantExpr(CE);
        // Save result at offset
        std::string postfix = getTypePostfix(CE->getType(),true);
        if (*postfix.begin()=='u') *postfix.begin() = 'i';
        postfix = "stind."+postfix;
        printSimpleInstruction(postfix.c_str());
      } else {
        cerr << "Constant = " << *I->constant << '\n';
        assert(0 && "Invalid static initializer");
      }
    }
  }
}


void MSILWriter::printFunction(const Function& F) {
  bool isSigned = F.paramHasAttr(0, Attribute::SExt);
  Out << "\n.method static ";
  Out << (F.hasLocalLinkage() ? "private " : "public ");
  if (F.isVarArg()) Out << "vararg ";
  Out << getTypeName(F.getReturnType(),isSigned) << 
    getConvModopt(F.getCallingConv()) << getValueName(&F) << '\n';
  // Arguments
  Out << "\t(";
  unsigned ArgIdx = 1;
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end(); I!=E;
       ++I, ++ArgIdx) {
    isSigned = F.paramHasAttr(ArgIdx, Attribute::SExt);
    if (I!=F.arg_begin()) Out << ", ";
    Out << getTypeName(I->getType(),isSigned) << getValueName(I);
  }
  Out << ") cil managed\n";
  // Body
  Out << "{\n";
  printLocalVariables(F);
  printFunctionBody(F);
  Out << "}\n";
}


void MSILWriter::printDeclarations(const TypeSymbolTable& ST) {
  std::string Name;
  std::set<const Type*> Printed;
  for (std::set<const Type*>::const_iterator
       UI = UsedTypes->begin(), UE = UsedTypes->end(); UI!=UE; ++UI) {
    const Type* Ty = *UI;
    if (isa<ArrayType>(Ty) || isa<VectorType>(Ty) || isa<StructType>(Ty))
      Name = getTypeName(Ty, false, true);
    // Type with no need to declare.
    else continue;
    // Print not duplicated type
    if (Printed.insert(Ty).second) {
      Out << ".class value explicit ansi sealed '" << Name << "'";
      Out << " { .pack " << 1 << " .size " << TD->getTypePaddedSize(Ty);
      Out << " }\n\n";
    }
  }
}


unsigned int MSILWriter::getBitWidth(const Type* Ty) {
  unsigned int N = Ty->getPrimitiveSizeInBits();
  assert(N!=0 && "Invalid type in getBitWidth()");
  switch (N) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return N;
  default:
    cerr << "Bits = " << N << '\n';
    assert(0 && "Unsupported integer width");
  }
  return 0; // Not reached
}


void MSILWriter::printStaticConstant(const Constant* C, uint64_t& Offset) {
  uint64_t TySize = 0;
  const Type* Ty = C->getType();
  // Print zero initialized constant.
  if (isa<ConstantAggregateZero>(C) || C->isNullValue()) {
    TySize = TD->getTypePaddedSize(C->getType());
    Offset += TySize;
    Out << "int8 (0) [" << TySize << "]";
    return;
  }
  // Print constant initializer
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    TySize = TD->getTypePaddedSize(Ty);
    const ConstantInt* Int = cast<ConstantInt>(C);
    Out << getPrimitiveTypeName(Ty,true) << "(" << Int->getSExtValue() << ")";
    break;
  }
  case Type::FloatTyID:
  case Type::DoubleTyID: {
    TySize = TD->getTypePaddedSize(Ty);
    const ConstantFP* FP = cast<ConstantFP>(C);
    if (Ty->getTypeID() == Type::FloatTyID)
      Out << "int32 (" << 
        (uint32_t)FP->getValueAPF().bitcastToAPInt().getZExtValue() << ')';
    else
      Out << "int64 (" << 
        FP->getValueAPF().bitcastToAPInt().getZExtValue() << ')';
    break;
  }
  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::StructTyID:
    for (unsigned I = 0, E = C->getNumOperands(); I<E; I++) {
      if (I!=0) Out << ",\n";
      printStaticConstant(C->getOperand(I),Offset);
    }
    break;
  case Type::PointerTyID:
    TySize = TD->getTypePaddedSize(C->getType());
    // Initialize with global variable address
    if (const GlobalVariable *G = dyn_cast<GlobalVariable>(C)) {
      std::string name = getValueName(G);
      Out << "&(" << name.insert(name.length()-1,"$data") << ")";
    } else {
      // Dynamic initialization
      if (!isa<ConstantPointerNull>(C) && !C->isNullValue())
        InitListPtr->push_back(StaticInitializer(C,Offset));
      // Null pointer initialization
      if (TySize==4) Out << "int32 (0)";
      else if (TySize==8) Out << "int64 (0)";
      else assert(0 && "Invalid pointer size");
    }
    break;
  default:
    cerr << "TypeID = " << Ty->getTypeID() << '\n';
    assert(0 && "Invalid type in printStaticConstant()");
  }
  // Increase offset.
  Offset += TySize;
}


void MSILWriter::printStaticInitializer(const Constant* C,
                                        const std::string& Name) {
  switch (C->getType()->getTypeID()) {
  case Type::IntegerTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID: 
    Out << getPrimitiveTypeName(C->getType(), false);
    break;
  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::StructTyID:
  case Type::PointerTyID:
    Out << getTypeName(C->getType());
    break;
  default:
    cerr << "Type = " << *C << "\n";
    assert(0 && "Invalid constant type");
  }
  // Print initializer
  std::string label = Name;
  label.insert(label.length()-1,"$data");
  Out << Name << " at " << label << '\n';
  Out << ".data " << label << " = {\n";
  uint64_t offset = 0;
  printStaticConstant(C,offset);
  Out << "\n}\n\n";
}


void MSILWriter::printVariableDefinition(const GlobalVariable* G) {
  const Constant* C = G->getInitializer();
  if (C->isNullValue() || isa<ConstantAggregateZero>(C) || isa<UndefValue>(C))
    InitListPtr = 0;
  else
    InitListPtr = &StaticInitList[G];
  printStaticInitializer(C,getValueName(G));
}


void MSILWriter::printGlobalVariables() {
  if (ModulePtr->global_empty()) return;
  Module::global_iterator I,E;
  for (I = ModulePtr->global_begin(), E = ModulePtr->global_end(); I!=E; ++I) {
    // Variable definition
    Out << ".field static " << (I->isDeclaration() ? "public " :
                                                     "private ");
    if (I->isDeclaration()) {
      Out << getTypeName(I->getType()) << getValueName(&*I) << "\n\n";
    } else
      printVariableDefinition(&*I);
  }
}


const char* MSILWriter::getLibraryName(const Function* F) {
  return getLibraryForSymbol(F->getName().c_str(), true, F->getCallingConv());
}


const char* MSILWriter::getLibraryName(const GlobalVariable* GV) {
  return getLibraryForSymbol(Mang->getValueName(GV).c_str(), false, 0);
}


const char* MSILWriter::getLibraryForSymbol(const char* Name, bool isFunction,
                                           unsigned CallingConv) {
  // TODO: Read *.def file with function and libraries definitions.
  return "MSVCRT.DLL";  
}


void MSILWriter::printExternals() {
  Module::const_iterator I,E;
  // Functions.
  for (I=ModulePtr->begin(),E=ModulePtr->end(); I!=E; ++I) {
    // Skip intrisics
    if (I->isIntrinsic()) continue;
    if (I->isDeclaration()) {
      const Function* F = I; 
      std::string Name = getConvModopt(F->getCallingConv())+getValueName(F);
      std::string Sig = 
        getCallSignature(cast<FunctionType>(F->getFunctionType()), NULL, Name);
      Out << ".method static hidebysig pinvokeimpl(\""
          << getLibraryName(F) << "\")\n\t" << Sig << " preservesig {}\n\n";
    }
  }
  // External variables and static initialization.
  Out <<
  ".method public hidebysig static pinvokeimpl(\"KERNEL32.DLL\" ansi winapi)"
  "  native int LoadLibrary(string) preservesig {}\n"
  ".method public hidebysig static pinvokeimpl(\"KERNEL32.DLL\" ansi winapi)"
  "  native int GetProcAddress(native int, string) preservesig {}\n";
  Out <<
  ".method private static void* $MSIL_Import(string lib,string sym)\n"
  " managed cil\n{\n"
  "\tldarg\tlib\n"
  "\tcall\tnative int LoadLibrary(string)\n"
  "\tldarg\tsym\n"
  "\tcall\tnative int GetProcAddress(native int,string)\n"
  "\tdup\n"
  "\tbrtrue\tL_01\n"
  "\tldstr\t\"Can no import variable\"\n"
  "\tnewobj\tinstance void [mscorlib]System.Exception::.ctor(string)\n"
  "\tthrow\n"
  "L_01:\n"
  "\tret\n"
  "}\n\n"
  ".method static private void $MSIL_Init() managed cil\n{\n";
  printStaticInitializerList();
  // Foreach global variable.
  for (Module::global_iterator I = ModulePtr->global_begin(),
       E = ModulePtr->global_end(); I!=E; ++I) {
    if (!I->isDeclaration() || !I->hasDLLImportLinkage()) continue;
    // Use "LoadLibrary"/"GetProcAddress" to recive variable address.
    std::string Label = "not_null$_"+utostr(getUniqID());
    std::string Tmp = getTypeName(I->getType())+getValueName(&*I);
    printSimpleInstruction("ldsflda",Tmp.c_str());
    Out << "\tldstr\t\"" << getLibraryName(&*I) << "\"\n";
    Out << "\tldstr\t\"" << Mang->getValueName(&*I) << "\"\n";
    printSimpleInstruction("call","void* $MSIL_Import(string,string)");
    printIndirectSave(I->getType());
  }
  printSimpleInstruction("ret");
  Out << "}\n\n";
}


//===----------------------------------------------------------------------===//
//                      External Interface declaration
//===----------------------------------------------------------------------===//

bool MSILTarget::addPassesToEmitWholeFile(PassManager &PM, raw_ostream &o,
                                          CodeGenFileType FileType, bool Fast)
{
  if (FileType != TargetMachine::AssemblyFile) return true;
  MSILWriter* Writer = new MSILWriter(o);
  PM.add(createGCLoweringPass());
  PM.add(createLowerAllocationsPass(true));
  // FIXME: Handle switch trougth native IL instruction "switch"
  PM.add(createLowerSwitchPass());
  PM.add(createCFGSimplificationPass());
  PM.add(new MSILModule(Writer->UsedTypes,Writer->TD));
  PM.add(Writer);
  PM.add(createGCInfoDeleter());
  return false;
}
