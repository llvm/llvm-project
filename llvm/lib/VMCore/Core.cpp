//===-- Core.cpp ----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C bindings for libLLVMCore.a, which implements
// the LLVM intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ModuleProvider.h"
#include "llvm/InlineAsm.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CallSite.h"
#include <cassert>
#include <cstdlib>
#include <cstring>

using namespace llvm;


/*===-- Error handling ----------------------------------------------------===*/

void LLVMDisposeMessage(char *Message) {
  free(Message);
}


/*===-- Operations on modules ---------------------------------------------===*/

LLVMModuleRef LLVMModuleCreateWithName(const char *ModuleID) {
  return wrap(new Module(ModuleID));
}

void LLVMDisposeModule(LLVMModuleRef M) {
  delete unwrap(M);
}

/*--.. Data layout .........................................................--*/
const char * LLVMGetDataLayout(LLVMModuleRef M) {
  return unwrap(M)->getDataLayout().c_str();
}

void LLVMSetDataLayout(LLVMModuleRef M, const char *Triple) {
  unwrap(M)->setDataLayout(Triple);
}

/*--.. Target triple .......................................................--*/
const char * LLVMGetTarget(LLVMModuleRef M) {
  return unwrap(M)->getTargetTriple().c_str();
}

void LLVMSetTarget(LLVMModuleRef M, const char *Triple) {
  unwrap(M)->setTargetTriple(Triple);
}

/*--.. Type names ..........................................................--*/
int LLVMAddTypeName(LLVMModuleRef M, const char *Name, LLVMTypeRef Ty) {
  return unwrap(M)->addTypeName(Name, unwrap(Ty));
}

void LLVMDeleteTypeName(LLVMModuleRef M, const char *Name) {
  std::string N(Name);
  
  TypeSymbolTable &TST = unwrap(M)->getTypeSymbolTable();
  for (TypeSymbolTable::iterator I = TST.begin(), E = TST.end(); I != E; ++I)
    if (I->first == N)
      TST.remove(I);
}

void LLVMDumpModule(LLVMModuleRef M) {
  unwrap(M)->dump();
}


/*===-- Operations on types -----------------------------------------------===*/

/*--.. Operations on all types (mostly) ....................................--*/

LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty) {
  return static_cast<LLVMTypeKind>(unwrap(Ty)->getTypeID());
}

/*--.. Operations on integer types .........................................--*/

LLVMTypeRef LLVMInt1Type(void)  { return (LLVMTypeRef) Type::Int1Ty;  }
LLVMTypeRef LLVMInt8Type(void)  { return (LLVMTypeRef) Type::Int8Ty;  }
LLVMTypeRef LLVMInt16Type(void) { return (LLVMTypeRef) Type::Int16Ty; }
LLVMTypeRef LLVMInt32Type(void) { return (LLVMTypeRef) Type::Int32Ty; }
LLVMTypeRef LLVMInt64Type(void) { return (LLVMTypeRef) Type::Int64Ty; }

LLVMTypeRef LLVMIntType(unsigned NumBits) {
  return wrap(IntegerType::get(NumBits));
}

unsigned LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy) {
  return unwrap<IntegerType>(IntegerTy)->getBitWidth();
}

/*--.. Operations on real types ............................................--*/

LLVMTypeRef LLVMFloatType(void)    { return (LLVMTypeRef) Type::FloatTy;     }
LLVMTypeRef LLVMDoubleType(void)   { return (LLVMTypeRef) Type::DoubleTy;    }
LLVMTypeRef LLVMX86FP80Type(void)  { return (LLVMTypeRef) Type::X86_FP80Ty;  }
LLVMTypeRef LLVMFP128Type(void)    { return (LLVMTypeRef) Type::FP128Ty;     }
LLVMTypeRef LLVMPPCFP128Type(void) { return (LLVMTypeRef) Type::PPC_FP128Ty; }

/*--.. Operations on function types ........................................--*/

LLVMTypeRef LLVMFunctionType(LLVMTypeRef ReturnType,
                             LLVMTypeRef *ParamTypes, unsigned ParamCount,
                             int IsVarArg) {
  std::vector<const Type*> Tys;
  for (LLVMTypeRef *I = ParamTypes, *E = ParamTypes + ParamCount; I != E; ++I)
    Tys.push_back(unwrap(*I));
  
  return wrap(FunctionType::get(unwrap(ReturnType), Tys, IsVarArg != 0));
}

int LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->isVarArg();
}

LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy) {
  return wrap(unwrap<FunctionType>(FunctionTy)->getReturnType());
}

unsigned LLVMCountParamTypes(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->getNumParams();
}

void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest) {
  FunctionType *Ty = unwrap<FunctionType>(FunctionTy);
  for (FunctionType::param_iterator I = Ty->param_begin(),
                                    E = Ty->param_end(); I != E; ++I)
    *Dest++ = wrap(*I);
}

/*--.. Operations on struct types ..........................................--*/

LLVMTypeRef LLVMStructType(LLVMTypeRef *ElementTypes,
                           unsigned ElementCount, int Packed) {
  std::vector<const Type*> Tys;
  for (LLVMTypeRef *I = ElementTypes,
                   *E = ElementTypes + ElementCount; I != E; ++I)
    Tys.push_back(unwrap(*I));
  
  return wrap(StructType::get(Tys, Packed != 0));
}

unsigned LLVMCountStructElementTypes(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->getNumElements();
}

void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest) {
  StructType *Ty = unwrap<StructType>(StructTy);
  for (FunctionType::param_iterator I = Ty->element_begin(),
                                    E = Ty->element_end(); I != E; ++I)
    *Dest++ = wrap(*I);
}

int LLVMIsPackedStruct(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->isPacked();
}

/*--.. Operations on array, pointer, and vector types (sequence types) .....--*/

LLVMTypeRef LLVMArrayType(LLVMTypeRef ElementType, unsigned ElementCount) {
  return wrap(ArrayType::get(unwrap(ElementType), ElementCount));
}

LLVMTypeRef LLVMPointerType(LLVMTypeRef ElementType, unsigned AddressSpace) {
  return wrap(PointerType::get(unwrap(ElementType), AddressSpace));
}

LLVMTypeRef LLVMVectorType(LLVMTypeRef ElementType, unsigned ElementCount) {
  return wrap(VectorType::get(unwrap(ElementType), ElementCount));
}

LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty) {
  return wrap(unwrap<SequentialType>(Ty)->getElementType());
}

unsigned LLVMGetArrayLength(LLVMTypeRef ArrayTy) {
  return unwrap<ArrayType>(ArrayTy)->getNumElements();
}

unsigned LLVMGetPointerAddressSpace(LLVMTypeRef PointerTy) {
  return unwrap<PointerType>(PointerTy)->getAddressSpace();
}

unsigned LLVMGetVectorSize(LLVMTypeRef VectorTy) {
  return unwrap<VectorType>(VectorTy)->getNumElements();
}

/*--.. Operations on other types ...........................................--*/

LLVMTypeRef LLVMVoidType(void)  { return (LLVMTypeRef) Type::VoidTy;  }
LLVMTypeRef LLVMLabelType(void) { return (LLVMTypeRef) Type::LabelTy; }

LLVMTypeRef LLVMOpaqueType(void) {
  return wrap(llvm::OpaqueType::get());
}

/*--.. Operations on type handles ..........................................--*/

LLVMTypeHandleRef LLVMCreateTypeHandle(LLVMTypeRef PotentiallyAbstractTy) {
  return wrap(new PATypeHolder(unwrap(PotentiallyAbstractTy)));
}

void LLVMDisposeTypeHandle(LLVMTypeHandleRef TypeHandle) {
  delete unwrap(TypeHandle);
}

LLVMTypeRef LLVMResolveTypeHandle(LLVMTypeHandleRef TypeHandle) {
  return wrap(unwrap(TypeHandle)->get());
}

void LLVMRefineType(LLVMTypeRef AbstractTy, LLVMTypeRef ConcreteTy) {
  unwrap<DerivedType>(AbstractTy)->refineAbstractTypeTo(unwrap(ConcreteTy));
}


/*===-- Operations on values ----------------------------------------------===*/

/*--.. Operations on all values ............................................--*/

LLVMTypeRef LLVMTypeOf(LLVMValueRef Val) {
  return wrap(unwrap(Val)->getType());
}

const char *LLVMGetValueName(LLVMValueRef Val) {
  return unwrap(Val)->getNameStart();
}

void LLVMSetValueName(LLVMValueRef Val, const char *Name) {
  unwrap(Val)->setName(Name);
}

void LLVMDumpValue(LLVMValueRef Val) {
  unwrap(Val)->dump();
}


/*--.. Conversion functions ................................................--*/

#define LLVM_DEFINE_VALUE_CAST(name)                                       \
  LLVMValueRef LLVMIsA##name(LLVMValueRef Val) {                           \
    return wrap(static_cast<Value*>(dyn_cast_or_null<name>(unwrap(Val)))); \
  }

LLVM_FOR_EACH_VALUE_SUBCLASS(LLVM_DEFINE_VALUE_CAST)


/*--.. Operations on constants of any type .................................--*/

LLVMValueRef LLVMConstNull(LLVMTypeRef Ty) {
  return wrap(Constant::getNullValue(unwrap(Ty)));
}

LLVMValueRef LLVMConstAllOnes(LLVMTypeRef Ty) {
  return wrap(Constant::getAllOnesValue(unwrap(Ty)));
}

LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty) {
  return wrap(UndefValue::get(unwrap(Ty)));
}

int LLVMIsConstant(LLVMValueRef Ty) {
  return isa<Constant>(unwrap(Ty));
}

int LLVMIsNull(LLVMValueRef Val) {
  if (Constant *C = dyn_cast<Constant>(unwrap(Val)))
    return C->isNullValue();
  return false;
}

int LLVMIsUndef(LLVMValueRef Val) {
  return isa<UndefValue>(unwrap(Val));
}

/*--.. Operations on scalar constants ......................................--*/

LLVMValueRef LLVMConstInt(LLVMTypeRef IntTy, unsigned long long N,
                          int SignExtend) {
  return wrap(ConstantInt::get(unwrap<IntegerType>(IntTy), N, SignExtend != 0));
}

static const fltSemantics &SemanticsForType(Type *Ty) {
  assert(Ty->isFloatingPoint() && "Type is not floating point!");
  if (Ty == Type::FloatTy)
    return APFloat::IEEEsingle;
  if (Ty == Type::DoubleTy)
    return APFloat::IEEEdouble;
  if (Ty == Type::X86_FP80Ty)
    return APFloat::x87DoubleExtended;
  if (Ty == Type::FP128Ty)
    return APFloat::IEEEquad;
  if (Ty == Type::PPC_FP128Ty)
    return APFloat::PPCDoubleDouble;
  return APFloat::Bogus;
}

LLVMValueRef LLVMConstReal(LLVMTypeRef RealTy, double N) {
  APFloat APN(N);
  bool ignored;
  APN.convert(SemanticsForType(unwrap(RealTy)), APFloat::rmNearestTiesToEven,
              &ignored);
  return wrap(ConstantFP::get(APN));
}

LLVMValueRef LLVMConstRealOfString(LLVMTypeRef RealTy, const char *Text) {
  return wrap(ConstantFP::get(APFloat(SemanticsForType(unwrap(RealTy)), Text)));
}

/*--.. Operations on composite constants ...................................--*/

LLVMValueRef LLVMConstString(const char *Str, unsigned Length,
                             int DontNullTerminate) {
  /* Inverted the sense of AddNull because ', 0)' is a
     better mnemonic for null termination than ', 1)'. */
  return wrap(ConstantArray::get(std::string(Str, Length),
                                 DontNullTerminate == 0));
}

LLVMValueRef LLVMConstArray(LLVMTypeRef ElementTy,
                            LLVMValueRef *ConstantVals, unsigned Length) {
  return wrap(ConstantArray::get(ArrayType::get(unwrap(ElementTy), Length),
                                 unwrap<Constant>(ConstantVals, Length),
                                 Length));
}

LLVMValueRef LLVMConstStruct(LLVMValueRef *ConstantVals, unsigned Count,
                             int Packed) {
  return wrap(ConstantStruct::get(unwrap<Constant>(ConstantVals, Count),
                                  Count, Packed != 0));
}

LLVMValueRef LLVMConstVector(LLVMValueRef *ScalarConstantVals, unsigned Size) {
  return wrap(ConstantVector::get(unwrap<Constant>(ScalarConstantVals, Size),
                                  Size));
}

/*--.. Constant expressions ................................................--*/

LLVMValueRef LLVMSizeOf(LLVMTypeRef Ty) {
  return wrap(ConstantExpr::getSizeOf(unwrap(Ty)));
}

LLVMValueRef LLVMConstNeg(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getNeg(unwrap<Constant>(ConstantVal)));
}

LLVMValueRef LLVMConstNot(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getNot(unwrap<Constant>(ConstantVal)));
}

LLVMValueRef LLVMConstAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getAdd(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getSub(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getMul(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstUDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getUDiv(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstSDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getSDiv(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFDiv(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstURem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getURem(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstSRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getSRem(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFRem(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstAnd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getAnd(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstOr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getOr(unwrap<Constant>(LHSConstant),
                                  unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstXor(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getXor(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstICmp(LLVMIntPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getICmp(Predicate,
                                    unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFCmp(LLVMRealPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFCmp(Predicate,
                                    unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstShl(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getShl(unwrap<Constant>(LHSConstant),
                                  unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstLShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getLShr(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstAShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getAShr(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstGEP(LLVMValueRef ConstantVal,
                          LLVMValueRef *ConstantIndices, unsigned NumIndices) {
  return wrap(ConstantExpr::getGetElementPtr(unwrap<Constant>(ConstantVal),
                                             unwrap<Constant>(ConstantIndices, 
                                                              NumIndices),
                                             NumIndices));
}

LLVMValueRef LLVMConstTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getTrunc(unwrap<Constant>(ConstantVal),
                                     unwrap(ToType)));
}

LLVMValueRef LLVMConstSExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getSExt(unwrap<Constant>(ConstantVal),
                                    unwrap(ToType)));
}

LLVMValueRef LLVMConstZExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getZExt(unwrap<Constant>(ConstantVal),
                                    unwrap(ToType)));
}

LLVMValueRef LLVMConstFPTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPTrunc(unwrap<Constant>(ConstantVal),
                                       unwrap(ToType)));
}

LLVMValueRef LLVMConstFPExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPExtend(unwrap<Constant>(ConstantVal),
                                        unwrap(ToType)));
}

LLVMValueRef LLVMConstUIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getUIToFP(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstSIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getSIToFP(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstFPToUI(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPToUI(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstFPToSI(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPToSI(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstPtrToInt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getPtrToInt(unwrap<Constant>(ConstantVal),
                                        unwrap(ToType)));
}

LLVMValueRef LLVMConstIntToPtr(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getIntToPtr(unwrap<Constant>(ConstantVal),
                                        unwrap(ToType)));
}

LLVMValueRef LLVMConstBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getBitCast(unwrap<Constant>(ConstantVal),
                                       unwrap(ToType)));
}

LLVMValueRef LLVMConstSelect(LLVMValueRef ConstantCondition,
                             LLVMValueRef ConstantIfTrue,
                             LLVMValueRef ConstantIfFalse) {
  return wrap(ConstantExpr::getSelect(unwrap<Constant>(ConstantCondition),
                                      unwrap<Constant>(ConstantIfTrue),
                                      unwrap<Constant>(ConstantIfFalse)));
}

LLVMValueRef LLVMConstExtractElement(LLVMValueRef VectorConstant,
                                     LLVMValueRef IndexConstant) {
  return wrap(ConstantExpr::getExtractElement(unwrap<Constant>(VectorConstant),
                                              unwrap<Constant>(IndexConstant)));
}

LLVMValueRef LLVMConstInsertElement(LLVMValueRef VectorConstant,
                                    LLVMValueRef ElementValueConstant,
                                    LLVMValueRef IndexConstant) {
  return wrap(ConstantExpr::getInsertElement(unwrap<Constant>(VectorConstant),
                                         unwrap<Constant>(ElementValueConstant),
                                             unwrap<Constant>(IndexConstant)));
}

LLVMValueRef LLVMConstShuffleVector(LLVMValueRef VectorAConstant,
                                    LLVMValueRef VectorBConstant,
                                    LLVMValueRef MaskConstant) {
  return wrap(ConstantExpr::getShuffleVector(unwrap<Constant>(VectorAConstant),
                                             unwrap<Constant>(VectorBConstant),
                                             unwrap<Constant>(MaskConstant)));
}

LLVMValueRef LLVMConstExtractValue(LLVMValueRef AggConstant, unsigned *IdxList,
                                   unsigned NumIdx) {
  return wrap(ConstantExpr::getExtractValue(unwrap<Constant>(AggConstant),
                                            IdxList, NumIdx));
}

LLVMValueRef LLVMConstInsertValue(LLVMValueRef AggConstant,
                                  LLVMValueRef ElementValueConstant,
                                  unsigned *IdxList, unsigned NumIdx) {
  return wrap(ConstantExpr::getInsertValue(unwrap<Constant>(AggConstant),
                                         unwrap<Constant>(ElementValueConstant),
                                           IdxList, NumIdx));
}

LLVMValueRef LLVMConstInlineAsm(LLVMTypeRef Ty, const char *AsmString, 
                                const char *Constraints, int HasSideEffects) {
  return wrap(InlineAsm::get(dyn_cast<FunctionType>(unwrap(Ty)), AsmString, 
                             Constraints, HasSideEffects));
}

/*--.. Operations on global variables, functions, and aliases (globals) ....--*/

LLVMModuleRef LLVMGetGlobalParent(LLVMValueRef Global) {
  return wrap(unwrap<GlobalValue>(Global)->getParent());
}

int LLVMIsDeclaration(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->isDeclaration();
}

LLVMLinkage LLVMGetLinkage(LLVMValueRef Global) {
  return static_cast<LLVMLinkage>(unwrap<GlobalValue>(Global)->getLinkage());
}

void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage) {
  unwrap<GlobalValue>(Global)
    ->setLinkage(static_cast<GlobalValue::LinkageTypes>(Linkage));
}

const char *LLVMGetSection(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->getSection().c_str();
}

void LLVMSetSection(LLVMValueRef Global, const char *Section) {
  unwrap<GlobalValue>(Global)->setSection(Section);
}

LLVMVisibility LLVMGetVisibility(LLVMValueRef Global) {
  return static_cast<LLVMVisibility>(
    unwrap<GlobalValue>(Global)->getVisibility());
}

void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz) {
  unwrap<GlobalValue>(Global)
    ->setVisibility(static_cast<GlobalValue::VisibilityTypes>(Viz));
}

unsigned LLVMGetAlignment(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->getAlignment();
}

void LLVMSetAlignment(LLVMValueRef Global, unsigned Bytes) {
  unwrap<GlobalValue>(Global)->setAlignment(Bytes);
}

/*--.. Operations on global variables ......................................--*/

LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name) {
  return wrap(new GlobalVariable(unwrap(Ty), false,
                                 GlobalValue::ExternalLinkage, 0, Name,
                                 unwrap(M)));
}

LLVMValueRef LLVMGetNamedGlobal(LLVMModuleRef M, const char *Name) {
  return wrap(unwrap(M)->getNamedGlobal(Name));
}

LLVMValueRef LLVMGetFirstGlobal(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::global_iterator I = Mod->global_begin();
  if (I == Mod->global_end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetLastGlobal(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::global_iterator I = Mod->global_end();
  if (I == Mod->global_begin())
    return 0;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextGlobal(LLVMValueRef GlobalVar) {
  GlobalVariable *GV = unwrap<GlobalVariable>(GlobalVar);
  Module::global_iterator I = GV;
  if (++I == GV->getParent()->global_end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousGlobal(LLVMValueRef GlobalVar) {
  GlobalVariable *GV = unwrap<GlobalVariable>(GlobalVar);
  Module::global_iterator I = GV;
  if (I == GV->getParent()->global_begin())
    return 0;
  return wrap(--I);
}

void LLVMDeleteGlobal(LLVMValueRef GlobalVar) {
  unwrap<GlobalVariable>(GlobalVar)->eraseFromParent();
}

LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar) {
  return wrap(unwrap<GlobalVariable>(GlobalVar)->getInitializer());
}

void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal) {
  unwrap<GlobalVariable>(GlobalVar)
    ->setInitializer(unwrap<Constant>(ConstantVal));
}

int LLVMIsThreadLocal(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->isThreadLocal();
}

void LLVMSetThreadLocal(LLVMValueRef GlobalVar, int IsThreadLocal) {
  unwrap<GlobalVariable>(GlobalVar)->setThreadLocal(IsThreadLocal != 0);
}

int LLVMIsGlobalConstant(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->isConstant();
}

void LLVMSetGlobalConstant(LLVMValueRef GlobalVar, int IsConstant) {
  unwrap<GlobalVariable>(GlobalVar)->setConstant(IsConstant != 0);
}

/*--.. Operations on aliases ......................................--*/

LLVMValueRef LLVMAddAlias(LLVMModuleRef M, LLVMTypeRef Ty, LLVMValueRef Aliasee,
                          const char *Name) {
  return wrap(new GlobalAlias(unwrap(Ty), GlobalValue::ExternalLinkage, Name,
                              unwrap<Constant>(Aliasee), unwrap (M)));
}

/*--.. Operations on functions .............................................--*/

LLVMValueRef LLVMAddFunction(LLVMModuleRef M, const char *Name,
                             LLVMTypeRef FunctionTy) {
  return wrap(Function::Create(unwrap<FunctionType>(FunctionTy),
                               GlobalValue::ExternalLinkage, Name, unwrap(M)));
}

LLVMValueRef LLVMGetNamedFunction(LLVMModuleRef M, const char *Name) {
  return wrap(unwrap(M)->getFunction(Name));
}

LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::iterator I = Mod->begin();
  if (I == Mod->end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetLastFunction(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::iterator I = Mod->end();
  if (I == Mod->begin())
    return 0;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextFunction(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Module::iterator I = Func;
  if (++I == Func->getParent()->end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousFunction(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Module::iterator I = Func;
  if (I == Func->getParent()->begin())
    return 0;
  return wrap(--I);
}

void LLVMDeleteFunction(LLVMValueRef Fn) {
  unwrap<Function>(Fn)->eraseFromParent();
}

unsigned LLVMGetIntrinsicID(LLVMValueRef Fn) {
  if (Function *F = dyn_cast<Function>(unwrap(Fn)))
    return F->getIntrinsicID();
  return 0;
}

unsigned LLVMGetFunctionCallConv(LLVMValueRef Fn) {
  return unwrap<Function>(Fn)->getCallingConv();
}

void LLVMSetFunctionCallConv(LLVMValueRef Fn, unsigned CC) {
  return unwrap<Function>(Fn)->setCallingConv(CC);
}

const char *LLVMGetGC(LLVMValueRef Fn) {
  Function *F = unwrap<Function>(Fn);
  return F->hasGC()? F->getGC() : 0;
}

void LLVMSetGC(LLVMValueRef Fn, const char *GC) {
  Function *F = unwrap<Function>(Fn);
  if (GC)
    F->setGC(GC);
  else
    F->clearGC();
}

/*--.. Operations on parameters ............................................--*/

unsigned LLVMCountParams(LLVMValueRef FnRef) {
  // This function is strictly redundant to
  //   LLVMCountParamTypes(LLVMGetElementType(LLVMTypeOf(FnRef)))
  return unwrap<Function>(FnRef)->arg_size();
}

void LLVMGetParams(LLVMValueRef FnRef, LLVMValueRef *ParamRefs) {
  Function *Fn = unwrap<Function>(FnRef);
  for (Function::arg_iterator I = Fn->arg_begin(),
                              E = Fn->arg_end(); I != E; I++)
    *ParamRefs++ = wrap(I);
}

LLVMValueRef LLVMGetParam(LLVMValueRef FnRef, unsigned index) {
  Function::arg_iterator AI = unwrap<Function>(FnRef)->arg_begin();
  while (index --> 0)
    AI++;
  return wrap(AI);
}

LLVMValueRef LLVMGetParamParent(LLVMValueRef V) {
  return wrap(unwrap<Argument>(V)->getParent());
}

LLVMValueRef LLVMGetFirstParam(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::arg_iterator I = Func->arg_begin();
  if (I == Func->arg_end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetLastParam(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::arg_iterator I = Func->arg_end();
  if (I == Func->arg_begin())
    return 0;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextParam(LLVMValueRef Arg) {
  Argument *A = unwrap<Argument>(Arg);
  Function::arg_iterator I = A;
  if (++I == A->getParent()->arg_end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousParam(LLVMValueRef Arg) {
  Argument *A = unwrap<Argument>(Arg);
  Function::arg_iterator I = A;
  if (I == A->getParent()->arg_begin())
    return 0;
  return wrap(--I);
}

void LLVMAddAttribute(LLVMValueRef Arg, LLVMAttribute PA) {
  unwrap<Argument>(Arg)->addAttr(PA);
}

void LLVMRemoveAttribute(LLVMValueRef Arg, LLVMAttribute PA) {
  unwrap<Argument>(Arg)->removeAttr(PA);
}

void LLVMSetParamAlignment(LLVMValueRef Arg, unsigned align) {
  unwrap<Argument>(Arg)->addAttr(
          Attribute::constructAlignmentFromInt(align));
}

/*--.. Operations on basic blocks ..........................................--*/

LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef BB) {
  return wrap(static_cast<Value*>(unwrap(BB)));
}

int LLVMValueIsBasicBlock(LLVMValueRef Val) {
  return isa<BasicBlock>(unwrap(Val));
}

LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val) {
  return wrap(unwrap<BasicBlock>(Val));
}

LLVMValueRef LLVMGetBasicBlockParent(LLVMBasicBlockRef BB) {
  return wrap(unwrap(BB)->getParent());
}

unsigned LLVMCountBasicBlocks(LLVMValueRef FnRef) {
  return unwrap<Function>(FnRef)->size();
}

void LLVMGetBasicBlocks(LLVMValueRef FnRef, LLVMBasicBlockRef *BasicBlocksRefs){
  Function *Fn = unwrap<Function>(FnRef);
  for (Function::iterator I = Fn->begin(), E = Fn->end(); I != E; I++)
    *BasicBlocksRefs++ = wrap(I);
}

LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn) {
  return wrap(&unwrap<Function>(Fn)->getEntryBlock());
}

LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::iterator I = Func->begin();
  if (I == Func->end())
    return 0;
  return wrap(I);
}

LLVMBasicBlockRef LLVMGetLastBasicBlock(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::iterator I = Func->end();
  if (I == Func->begin())
    return 0;
  return wrap(--I);
}

LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  Function::iterator I = Block;
  if (++I == Block->getParent()->end())
    return 0;
  return wrap(I);
}

LLVMBasicBlockRef LLVMGetPreviousBasicBlock(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  Function::iterator I = Block;
  if (I == Block->getParent()->begin())
    return 0;
  return wrap(--I);
}

LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef FnRef, const char *Name) {
  return wrap(BasicBlock::Create(Name, unwrap<Function>(FnRef)));
}

LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef InsertBeforeBBRef,
                                       const char *Name) {
  BasicBlock *InsertBeforeBB = unwrap(InsertBeforeBBRef);
  return wrap(BasicBlock::Create(Name, InsertBeforeBB->getParent(),
                                 InsertBeforeBB));
}

void LLVMDeleteBasicBlock(LLVMBasicBlockRef BBRef) {
  unwrap(BBRef)->eraseFromParent();
}

/*--.. Operations on instructions ..........................................--*/

LLVMBasicBlockRef LLVMGetInstructionParent(LLVMValueRef Inst) {
  return wrap(unwrap<Instruction>(Inst)->getParent());
}

LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  BasicBlock::iterator I = Block->begin();
  if (I == Block->end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetLastInstruction(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  BasicBlock::iterator I = Block->end();
  if (I == Block->begin())
    return 0;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextInstruction(LLVMValueRef Inst) {
  Instruction *Instr = unwrap<Instruction>(Inst);
  BasicBlock::iterator I = Instr;
  if (++I == Instr->getParent()->end())
    return 0;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousInstruction(LLVMValueRef Inst) {
  Instruction *Instr = unwrap<Instruction>(Inst);
  BasicBlock::iterator I = Instr;
  if (I == Instr->getParent()->begin())
    return 0;
  return wrap(--I);
}

/*--.. Call and invoke instructions ........................................--*/

unsigned LLVMGetInstructionCallConv(LLVMValueRef Instr) {
  Value *V = unwrap(Instr);
  if (CallInst *CI = dyn_cast<CallInst>(V))
    return CI->getCallingConv();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(V))
    return II->getCallingConv();
  assert(0 && "LLVMGetInstructionCallConv applies only to call and invoke!");
  return 0;
}

void LLVMSetInstructionCallConv(LLVMValueRef Instr, unsigned CC) {
  Value *V = unwrap(Instr);
  if (CallInst *CI = dyn_cast<CallInst>(V))
    return CI->setCallingConv(CC);
  else if (InvokeInst *II = dyn_cast<InvokeInst>(V))
    return II->setCallingConv(CC);
  assert(0 && "LLVMSetInstructionCallConv applies only to call and invoke!");
}

void LLVMAddInstrAttribute(LLVMValueRef Instr, unsigned index, 
                           LLVMAttribute PA) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  Call.setAttributes(
    Call.getAttributes().addAttr(index, PA));
}

void LLVMRemoveInstrAttribute(LLVMValueRef Instr, unsigned index, 
                              LLVMAttribute PA) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  Call.setAttributes(
    Call.getAttributes().removeAttr(index, PA));
}

void LLVMSetInstrParamAlignment(LLVMValueRef Instr, unsigned index, 
                                unsigned align) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  Call.setAttributes(
    Call.getAttributes().addAttr(index, 
        Attribute::constructAlignmentFromInt(align)));
}

/*--.. Operations on call instructions (only) ..............................--*/

int LLVMIsTailCall(LLVMValueRef Call) {
  return unwrap<CallInst>(Call)->isTailCall();
}

void LLVMSetTailCall(LLVMValueRef Call, int isTailCall) {
  unwrap<CallInst>(Call)->setTailCall(isTailCall);
}

/*--.. Operations on phi nodes .............................................--*/

void LLVMAddIncoming(LLVMValueRef PhiNode, LLVMValueRef *IncomingValues,
                     LLVMBasicBlockRef *IncomingBlocks, unsigned Count) {
  PHINode *PhiVal = unwrap<PHINode>(PhiNode);
  for (unsigned I = 0; I != Count; ++I)
    PhiVal->addIncoming(unwrap(IncomingValues[I]), unwrap(IncomingBlocks[I]));
}

unsigned LLVMCountIncoming(LLVMValueRef PhiNode) {
  return unwrap<PHINode>(PhiNode)->getNumIncomingValues();
}

LLVMValueRef LLVMGetIncomingValue(LLVMValueRef PhiNode, unsigned Index) {
  return wrap(unwrap<PHINode>(PhiNode)->getIncomingValue(Index));
}

LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef PhiNode, unsigned Index) {
  return wrap(unwrap<PHINode>(PhiNode)->getIncomingBlock(Index));
}


/*===-- Instruction builders ----------------------------------------------===*/

LLVMBuilderRef LLVMCreateBuilder(void) {
  return wrap(new IRBuilder<>());
}

void LLVMPositionBuilder(LLVMBuilderRef Builder, LLVMBasicBlockRef Block,
                         LLVMValueRef Instr) {
  BasicBlock *BB = unwrap(Block);
  Instruction *I = Instr? unwrap<Instruction>(Instr) : (Instruction*) BB->end();
  unwrap(Builder)->SetInsertPoint(BB, I);
}

void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr) {
  Instruction *I = unwrap<Instruction>(Instr);
  unwrap(Builder)->SetInsertPoint(I->getParent(), I);
}

void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block) {
  BasicBlock *BB = unwrap(Block);
  unwrap(Builder)->SetInsertPoint(BB);
}

LLVMBasicBlockRef LLVMGetInsertBlock(LLVMBuilderRef Builder) {
   return wrap(unwrap(Builder)->GetInsertBlock());
}

void LLVMClearInsertionPosition(LLVMBuilderRef Builder) {
  unwrap(Builder)->ClearInsertionPoint ();
}

void LLVMInsertIntoBuilder(LLVMBuilderRef Builder, LLVMValueRef Instr) {
  unwrap(Builder)->Insert(unwrap<Instruction>(Instr));
}

void LLVMDisposeBuilder(LLVMBuilderRef Builder) {
  delete unwrap(Builder);
}

/*--.. Instruction builders ................................................--*/

LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateRetVoid());
}

LLVMValueRef LLVMBuildRet(LLVMBuilderRef B, LLVMValueRef V) {
  return wrap(unwrap(B)->CreateRet(unwrap(V)));
}

LLVMValueRef LLVMBuildBr(LLVMBuilderRef B, LLVMBasicBlockRef Dest) {
  return wrap(unwrap(B)->CreateBr(unwrap(Dest)));
}

LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef B, LLVMValueRef If,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Else) {
  return wrap(unwrap(B)->CreateCondBr(unwrap(If), unwrap(Then), unwrap(Else)));
}

LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef B, LLVMValueRef V,
                             LLVMBasicBlockRef Else, unsigned NumCases) {
  return wrap(unwrap(B)->CreateSwitch(unwrap(V), unwrap(Else), NumCases));
}

LLVMValueRef LLVMBuildInvoke(LLVMBuilderRef B, LLVMValueRef Fn,
                             LLVMValueRef *Args, unsigned NumArgs,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch,
                             const char *Name) {
  return wrap(unwrap(B)->CreateInvoke(unwrap(Fn), unwrap(Then), unwrap(Catch),
                                      unwrap(Args), unwrap(Args) + NumArgs,
                                      Name));
}

LLVMValueRef LLVMBuildUnwind(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateUnwind());
}

LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateUnreachable());
}

void LLVMAddCase(LLVMValueRef Switch, LLVMValueRef OnVal,
                 LLVMBasicBlockRef Dest) {
  unwrap<SwitchInst>(Switch)->addCase(unwrap<ConstantInt>(OnVal), unwrap(Dest));
}

/*--.. Arithmetic ..........................................................--*/

LLVMValueRef LLVMBuildAdd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateAdd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSub(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateSub(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildMul(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateMul(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateUDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateSDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildURem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateURem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSRem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateSRem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFRem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFRem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildShl(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateShl(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildLShr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateLShr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAShr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateAShr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAnd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateAnd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildOr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                         const char *Name) {
  return wrap(unwrap(B)->CreateOr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildXor(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateXor(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNeg(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateNeg(unwrap(V), Name));
}

LLVMValueRef LLVMBuildNot(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateNot(unwrap(V), Name));
}

/*--.. Memory ..............................................................--*/

LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef B, LLVMTypeRef Ty,
                             const char *Name) {
  return wrap(unwrap(B)->CreateMalloc(unwrap(Ty), 0, Name));
}

LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name) {
  return wrap(unwrap(B)->CreateMalloc(unwrap(Ty), unwrap(Val), Name));
}

LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                             const char *Name) {
  return wrap(unwrap(B)->CreateAlloca(unwrap(Ty), 0, Name));
}

LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name) {
  return wrap(unwrap(B)->CreateAlloca(unwrap(Ty), unwrap(Val), Name));
}

LLVMValueRef LLVMBuildFree(LLVMBuilderRef B, LLVMValueRef PointerVal) {
  return wrap(unwrap(B)->CreateFree(unwrap(PointerVal)));
}


LLVMValueRef LLVMBuildLoad(LLVMBuilderRef B, LLVMValueRef PointerVal,
                           const char *Name) {
  return wrap(unwrap(B)->CreateLoad(unwrap(PointerVal), Name));
}

LLVMValueRef LLVMBuildStore(LLVMBuilderRef B, LLVMValueRef Val, 
                            LLVMValueRef PointerVal) {
  return wrap(unwrap(B)->CreateStore(unwrap(Val), unwrap(PointerVal)));
}

LLVMValueRef LLVMBuildGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                          LLVMValueRef *Indices, unsigned NumIndices,
                          const char *Name) {
  return wrap(unwrap(B)->CreateGEP(unwrap(Pointer), unwrap(Indices),
                                   unwrap(Indices) + NumIndices, Name));
}

/*--.. Casts ...............................................................--*/

LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateTrunc(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildZExt(LLVMBuilderRef B, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateZExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildSExt(LLVMBuilderRef B, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPToUI(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPToSI(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateUIToFP(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSIToFP(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPTrunc(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef B, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreatePtrToInt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef B, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateIntToPtr(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateBitCast(unwrap(Val), unwrap(DestTy), Name));
}

/*--.. Comparisons .........................................................--*/

LLVMValueRef LLVMBuildICmp(LLVMBuilderRef B, LLVMIntPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateICmp(static_cast<ICmpInst::Predicate>(Op),
                                    unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef B, LLVMRealPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFCmp(static_cast<FCmpInst::Predicate>(Op),
                                    unwrap(LHS), unwrap(RHS), Name));
}

/*--.. Miscellaneous instructions ..........................................--*/

LLVMValueRef LLVMBuildPhi(LLVMBuilderRef B, LLVMTypeRef Ty, const char *Name) {
  return wrap(unwrap(B)->CreatePHI(unwrap(Ty), Name));
}

LLVMValueRef LLVMBuildCall(LLVMBuilderRef B, LLVMValueRef Fn,
                           LLVMValueRef *Args, unsigned NumArgs,
                           const char *Name) {
  return wrap(unwrap(B)->CreateCall(unwrap(Fn), unwrap(Args),
                                    unwrap(Args) + NumArgs, Name));
}

LLVMValueRef LLVMBuildSelect(LLVMBuilderRef B, LLVMValueRef If,
                             LLVMValueRef Then, LLVMValueRef Else,
                             const char *Name) {
  return wrap(unwrap(B)->CreateSelect(unwrap(If), unwrap(Then), unwrap(Else),
                                      Name));
}

LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef B, LLVMValueRef List,
                            LLVMTypeRef Ty, const char *Name) {
  return wrap(unwrap(B)->CreateVAArg(unwrap(List), unwrap(Ty), Name));
}

LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef B, LLVMValueRef VecVal,
                                      LLVMValueRef Index, const char *Name) {
  return wrap(unwrap(B)->CreateExtractElement(unwrap(VecVal), unwrap(Index),
                                              Name));
}

LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef B, LLVMValueRef VecVal,
                                    LLVMValueRef EltVal, LLVMValueRef Index,
                                    const char *Name) {
  return wrap(unwrap(B)->CreateInsertElement(unwrap(VecVal), unwrap(EltVal),
                                             unwrap(Index), Name));
}

LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef B, LLVMValueRef V1,
                                    LLVMValueRef V2, LLVMValueRef Mask,
                                    const char *Name) {
  return wrap(unwrap(B)->CreateShuffleVector(unwrap(V1), unwrap(V2),
                                             unwrap(Mask), Name));
}

LLVMValueRef LLVMBuildExtractValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                   unsigned Index, const char *Name) {
  return wrap(unwrap(B)->CreateExtractValue(unwrap(AggVal), Index, Name));
}

LLVMValueRef LLVMBuildInsertValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                  LLVMValueRef EltVal, unsigned Index,
                                  const char *Name) {
  return wrap(unwrap(B)->CreateInsertValue(unwrap(AggVal), unwrap(EltVal),
                                           Index, Name));
}


/*===-- Module providers --------------------------------------------------===*/

LLVMModuleProviderRef
LLVMCreateModuleProviderForExistingModule(LLVMModuleRef M) {
  return wrap(new ExistingModuleProvider(unwrap(M)));
}

void LLVMDisposeModuleProvider(LLVMModuleProviderRef MP) {
  delete unwrap(MP);
}


/*===-- Memory buffers ----------------------------------------------------===*/

int LLVMCreateMemoryBufferWithContentsOfFile(const char *Path,
                                             LLVMMemoryBufferRef *OutMemBuf,
                                             char **OutMessage) {
  std::string Error;
  if (MemoryBuffer *MB = MemoryBuffer::getFile(Path, &Error)) {
    *OutMemBuf = wrap(MB);
    return 0;
  }
  
  *OutMessage = strdup(Error.c_str());
  return 1;
}

int LLVMCreateMemoryBufferWithSTDIN(LLVMMemoryBufferRef *OutMemBuf,
                                    char **OutMessage) {
  if (MemoryBuffer *MB = MemoryBuffer::getSTDIN()) {
    *OutMemBuf = wrap(MB);
    return 0;
  }
  
  *OutMessage = strdup("stdin is empty.");
  return 1;
}

void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef MemBuf) {
  delete unwrap(MemBuf);
}
