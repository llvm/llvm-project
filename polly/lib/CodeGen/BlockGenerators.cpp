//===--- BlockGenerators.cpp - Generate code for statements -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BlockGenerator and VectorBlockGenerator classes,
// which generate sequential code and vectorized code for a polyhedral
// statement, respectively.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopInfo.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/Support/GICHelper.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"

#include "isl/aff.h"
#include "isl/set.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool>
Aligned("enable-polly-aligned",
       cl::desc("Assumed aligned memory accesses."), cl::Hidden,
       cl::value_desc("OpenMP code generation enabled if true"),
       cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
SCEVCodegen("polly-codegen-scev",
            cl::desc("Use SCEV based code generation."), cl::Hidden,
            cl::init(false), cl::ZeroOrMore);

/// The SCEVRewriter takes a scalar evolution expression and updates the
/// following components:
///
/// - SCEVUnknown
///
///   Values referenced in SCEVUnknown subexpressions are looked up in
///   two Value to Value maps (GlobalMap and BBMap). If they are found they are
///   replaced by a reference to the value they map to.
///
/// - SCEVAddRecExpr
///
///   Based on a Loop -> Value map {Loop_1: %Value}, an expression
///   {%Base, +, %Step}<Loop_1> is rewritten to %Base + %Value * %Step.
///   AddRecExpr's with more than two operands can not be translated.
///
///   FIXME: The comment above is not yet reality. At the moment we derive
///   %Value by looking up the canonical IV of the loop and by defining
///   %Value = GlobalMap[%IV]. This needs to be changed to remove the need for
///   canonical induction variables.
///
///
/// How can this be used?
/// ====================
///
/// SCEVRewrite based code generation works on virtually independent blocks.
/// This means we do not run the independent blocks pass to rewrite scalar
/// instructions, but just ignore instructions that we can analyze with scalar
/// evolution. Virtually independent blocks are blocks that only reference the
/// following values:
///
/// o Values calculated within a basic block
/// o Values representable by SCEV
///
/// During code generation we can ignore all instructions:
///
/// - Ignore all instructions except:
///   - Load instructions
///   - Instructions that reference operands already calculated within the
///     basic block.
///   - Store instructions
struct SCEVRewriter : public SCEVVisitor<SCEVRewriter, const SCEV*> {
public:
  static const SCEV *rewrite(const SCEV *scev, Scop &S, ScalarEvolution &SE,
                             ValueMapT &GlobalMap,  ValueMapT &BBMap) {
    SCEVRewriter Rewriter(S, SE, GlobalMap, BBMap);
    return Rewriter.visit(scev);
  }

  SCEVRewriter(Scop &S, ScalarEvolution &SE, ValueMapT &GlobalMap,
               ValueMapT &BBMap) : S(S), SE(SE), GlobalMap(GlobalMap),
               BBMap(BBMap) {}

  const SCEV *visit(const SCEV *Expr) {
    // FIXME: The parameter handling is incorrect.
    //
    // Polly does only detect parameters in Access function and loop iteration
    // counters, but it does not get parameters that are just used by
    // instructions within the basic block.
    //
    // There are two options to solve this:
    //  o Iterate over all instructions of the SCoP and find the actual
    //    parameters.
    //  o Just check within the SCEVRewriter if Values lay outside of the SCoP
    //    and detect parameters on the fly.
    //
    // This is especially important for OpenMP and GPGPU code generation, as
    // they require us to detect and possibly rewrite the corresponding
    // parameters.
    if (isl_id *Id = S.getIdForParam(Expr)) {
      isl_id_free(Id);
      return Expr;
    }


    return SCEVVisitor<SCEVRewriter, const SCEV*>::visit(Expr);
  }

  const SCEV *visitConstant(const SCEVConstant *Constant) {
    return Constant;
  }

  const SCEV *visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    const SCEV *Operand = visit(Expr->getOperand());
    return SE.getTruncateExpr(Operand, Expr->getType());
  }

  const SCEV *visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    const SCEV *Operand = visit(Expr->getOperand());
    return SE.getZeroExtendExpr(Operand, Expr->getType());
  }

  const SCEV *visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    const SCEV *Operand = visit(Expr->getOperand());
    return SE.getSignExtendExpr(Operand, Expr->getType());
  }

  const SCEV *visitAddExpr(const SCEVAddExpr *Expr) {
    SmallVector<const SCEV *, 2> Operands;
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      const SCEV *Operand = visit(Expr->getOperand(i));
      Operands.push_back(Operand);
    }

    return SE.getAddExpr(Operands);
  }

  const SCEV *visitMulExpr(const SCEVMulExpr *Expr) {
    SmallVector<const SCEV *, 2> Operands;
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      const SCEV *Operand = visit(Expr->getOperand(i));
      Operands.push_back(Operand);
    }

    return SE.getMulExpr(Operands);
  }

  const SCEV *visitUDivExpr(const SCEVUDivExpr *Expr) {
    return SE.getUDivExpr(visit(Expr->getLHS()), visit(Expr->getRHS()));
  }

  // Return a new induction variable if the loop is within the original SCoP
  // or NULL otherwise.
  Value *getNewIV(const Loop *L) {
    Value *IV = L->getCanonicalInductionVariable();
    if (!IV)
      return NULL;

    ValueMapT::iterator NewIV = GlobalMap.find(IV);

    if (NewIV == GlobalMap.end())
      return NULL;

    return NewIV->second;
  }

  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    Value *IV;

    IV = getNewIV(Expr->getLoop());

    // The IV is not within the GlobalMaps. So do not rewrite it and also do
    // not rewrite any descendants.
    if (!IV)
      return Expr;

    assert(Expr->getNumOperands() == 2
          && "An AddRecExpr with more than two operands can not be rewritten.");

    const SCEV *Base, *Step, *IVExpr, *Product;

    Base = visit(Expr->getStart());
    Step = visit(Expr->getOperand(1));
    IVExpr = SE.getUnknown(IV);
    IVExpr = SE.getTruncateOrSignExtend(IVExpr, Step->getType());
    Product = SE.getMulExpr(Step, IVExpr);

    return SE.getAddExpr(Base, Product);
  }

  const SCEV *visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    SmallVector<const SCEV *, 2> Operands;
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      const SCEV *Operand = visit(Expr->getOperand(i));
      Operands.push_back(Operand);
    }

    return SE.getSMaxExpr(Operands);
  }

  const SCEV *visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    SmallVector<const SCEV *, 2> Operands;
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      const SCEV *Operand = visit(Expr->getOperand(i));
      Operands.push_back(Operand);
    }

    return SE.getUMaxExpr(Operands);
  }

  const SCEV *visitUnknown(const SCEVUnknown *Expr) {
    Value *V = Expr->getValue();

    if (GlobalMap.count(V))
      return SE.getUnknown(GlobalMap[V]);

    if (BBMap.count(V))
      return SE.getUnknown(BBMap[V]);

    return Expr;
  }

private:
  Scop &S;
  ScalarEvolution &SE;
  ValueMapT &GlobalMap;
  ValueMapT &BBMap;
};

// Helper class to generate memory location.
namespace {
class IslGenerator {
public:
  IslGenerator(IRBuilder<> &Builder, std::vector<Value *> &IVS) :
    Builder(Builder), IVS(IVS) {}
  Value *generateIslInt(__isl_take isl_int Int);
  Value *generateIslAff(__isl_take isl_aff *Aff);
  Value *generateIslPwAff(__isl_take isl_pw_aff *PwAff);

private:
  typedef struct {
    Value *Result;
    class IslGenerator *Generator;
  } IslGenInfo;

  IRBuilder<> &Builder;
  std::vector<Value *> &IVS;
  static int mergeIslAffValues(__isl_take isl_set *Set,
                               __isl_take isl_aff *Aff, void *User);
};
}


Value *IslGenerator::generateIslInt(isl_int Int) {
  mpz_t IntMPZ;
  mpz_init(IntMPZ);
  isl_int_get_gmp(Int, IntMPZ);
  Value *IntValue = Builder.getInt(APInt_from_MPZ(IntMPZ));
  mpz_clear(IntMPZ);
  return IntValue;
}

Value *IslGenerator::generateIslAff(__isl_take isl_aff *Aff) {
  Value *Result;
  Value *ConstValue;
  isl_int ConstIsl;

  isl_int_init(ConstIsl);
  isl_aff_get_constant(Aff, &ConstIsl);
  ConstValue = generateIslInt(ConstIsl);
  Type *Ty = Builder.getInt64Ty();

  // FIXME: We should give the constant and coefficients the right type. Here
  // we force it into i64.
  Result = Builder.CreateSExtOrBitCast(ConstValue, Ty);

  unsigned int NbInputDims = isl_aff_dim(Aff, isl_dim_in);

  assert((IVS.size() == NbInputDims) && "The Dimension of Induction Variables"
         "must match the dimension of the affine space.");

  isl_int CoefficientIsl;
  isl_int_init(CoefficientIsl);

  for (unsigned int i = 0; i < NbInputDims; ++i) {
    Value *CoefficientValue;
    isl_aff_get_coefficient(Aff, isl_dim_in, i, &CoefficientIsl);

    if (isl_int_is_zero(CoefficientIsl))
      continue;

    CoefficientValue = generateIslInt(CoefficientIsl);
    CoefficientValue = Builder.CreateIntCast(CoefficientValue, Ty, true);
    Value *IV = Builder.CreateIntCast(IVS[i], Ty, true);
    Value *PAdd = Builder.CreateMul(CoefficientValue, IV, "p_mul_coeff");
    Result = Builder.CreateAdd(Result, PAdd, "p_sum_coeff");
  }

  isl_int_clear(CoefficientIsl);
  isl_int_clear(ConstIsl);
  isl_aff_free(Aff);

  return Result;
}

int IslGenerator::mergeIslAffValues(__isl_take isl_set *Set,
                                    __isl_take isl_aff *Aff, void *User) {
  IslGenInfo *GenInfo = (IslGenInfo *)User;

  assert((GenInfo->Result == NULL) && "Result is already set."
         "Currently only single isl_aff is supported");
  assert(isl_set_plain_is_universe(Set)
         && "Code generation failed because the set is not universe");

  GenInfo->Result = GenInfo->Generator->generateIslAff(Aff);

  isl_set_free(Set);
  return 0;
}

Value *IslGenerator::generateIslPwAff(__isl_take isl_pw_aff *PwAff) {
  IslGenInfo User;
  User.Result = NULL;
  User.Generator = this;
  isl_pw_aff_foreach_piece(PwAff, mergeIslAffValues, &User);
  assert(User.Result && "Code generation for isl_pw_aff failed");

  isl_pw_aff_free(PwAff);
  return User.Result;
}


BlockGenerator::BlockGenerator(IRBuilder<> &B, ScopStmt &Stmt, Pass *P):
  Builder(B), Statement(Stmt), P(P), SE(P->getAnalysis<ScalarEvolution>()) {}

bool BlockGenerator::isSCEVIgnore(const Instruction *Inst) {
  if (SCEVCodegen && SE.isSCEVable(Inst->getType()))
    if (const SCEV *Scev = SE.getSCEV(const_cast<Instruction*>(Inst)))
      if (!isa<SCEVCouldNotCompute>(Scev)) {
        if (const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(Scev)) {
          if (Unknown->getValue() != Inst)
            return true;
        } else {
          return true;
        }
      }

  return false;
}

Value *BlockGenerator::getNewValue(const Value *Old, ValueMapT &BBMap,
                                   ValueMapT &GlobalMap) {
  // We assume constants never change.
  // This avoids map lookups for many calls to this function.
  if (isa<Constant>(Old))
    return const_cast<Value*>(Old);

  if (GlobalMap.count(Old)) {
    Value *New = GlobalMap[Old];

    if (Old->getType()->getScalarSizeInBits()
        < New->getType()->getScalarSizeInBits())
      New = Builder.CreateTruncOrBitCast(New, Old->getType());

    return New;
  }

  if (BBMap.count(Old)) {
    return BBMap[Old];
  }

  if (SCEVCodegen && SE.isSCEVable(Old->getType()))
    if (const SCEV *Scev = SE.getSCEV(const_cast<Value*>(Old)))
      if (!isa<SCEVCouldNotCompute>(Scev)) {
        const SCEV *NewScev = SCEVRewriter::rewrite(Scev,
                                                    *Statement.getParent(), SE,
                                                    GlobalMap, BBMap);
        SCEVExpander Expander(SE, "polly");
        Value *Expanded = Expander.expandCodeFor(NewScev, Old->getType(),
                                                 Builder.GetInsertPoint());

        BBMap[Old] = Expanded;
        return Expanded;
      }

  // 'Old' is within the original SCoP, but was not rewritten.
  //
  // Such values appear, if they only calculate information already available in
  // the polyhedral description (e.g.  an induction variable increment). They
  // can be safely ignored.
  if (const Instruction *Inst = dyn_cast<Instruction>(Old))
    if (Statement.getParent()->getRegion().contains(Inst->getParent()))
      return NULL;

  // Everything else is probably a scop-constant value defined as global,
  // function parameter or an instruction not within the scop.
  return const_cast<Value*>(Old);
}

void BlockGenerator::copyInstScalar(const Instruction *Inst, ValueMapT &BBMap,
                                    ValueMapT &GlobalMap) {
  Instruction *NewInst = Inst->clone();

  // Replace old operands with the new ones.
  for (Instruction::const_op_iterator OI = Inst->op_begin(),
       OE = Inst->op_end(); OI != OE; ++OI) {
    Value *OldOperand = *OI;
    Value *NewOperand = getNewValue(OldOperand, BBMap, GlobalMap);

    if (!NewOperand) {
      assert(!isa<StoreInst>(NewInst)
             && "Store instructions are always needed!");
      delete NewInst;
      return;
    }

    NewInst->replaceUsesOfWith(OldOperand, NewOperand);
  }

  Builder.Insert(NewInst);
  BBMap[Inst] = NewInst;

  if (!NewInst->getType()->isVoidTy())
    NewInst->setName("p_" + Inst->getName());
}

std::vector<Value*> BlockGenerator::getMemoryAccessIndex(
  __isl_keep isl_map *AccessRelation, Value *BaseAddress,
  ValueMapT &BBMap, ValueMapT &GlobalMap) {

  assert((isl_map_dim(AccessRelation, isl_dim_out) == 1)
         && "Only single dimensional access functions supported");

  std::vector<Value *> IVS;
  for (unsigned i = 0; i < Statement.getNumIterators(); ++i) {
    const Value *OriginalIV = Statement.getInductionVariableForDimension(i);
    Value *NewIV = getNewValue(OriginalIV, BBMap, GlobalMap);
    IVS.push_back(NewIV);
  }

  isl_pw_aff *PwAff = isl_map_dim_max(isl_map_copy(AccessRelation), 0);
  IslGenerator IslGen(Builder, IVS);
  Value *OffsetValue = IslGen.generateIslPwAff(PwAff);

  Type *Ty = Builder.getInt64Ty();
  OffsetValue = Builder.CreateIntCast(OffsetValue, Ty, true);

  std::vector<Value*> IndexArray;
  Value *NullValue = Constant::getNullValue(Ty);
  IndexArray.push_back(NullValue);
  IndexArray.push_back(OffsetValue);
  return IndexArray;
}

Value *BlockGenerator::getNewAccessOperand(
  __isl_keep isl_map *NewAccessRelation, Value *BaseAddress,
  ValueMapT &BBMap, ValueMapT &GlobalMap) {
  std::vector<Value*> IndexArray = getMemoryAccessIndex(NewAccessRelation,
                                                        BaseAddress,
                                                        BBMap, GlobalMap);
  Value *NewOperand = Builder.CreateGEP(BaseAddress, IndexArray,
                                        "p_newarrayidx_");
  return NewOperand;
}

Value *BlockGenerator::generateLocationAccessed(const Instruction *Inst,
                                                const Value *Pointer,
                                                ValueMapT &BBMap,
                                                ValueMapT &GlobalMap) {
  MemoryAccess &Access = Statement.getAccessFor(Inst);
  isl_map *CurrentAccessRelation = Access.getAccessRelation();
  isl_map *NewAccessRelation = Access.getNewAccessRelation();

  assert(isl_map_has_equal_space(CurrentAccessRelation, NewAccessRelation)
         && "Current and new access function use different spaces");

  Value *NewPointer;

  if (!NewAccessRelation) {
    NewPointer = getNewValue(Pointer, BBMap, GlobalMap);
  } else {
    Value *BaseAddress = const_cast<Value*>(Access.getBaseAddr());
    NewPointer = getNewAccessOperand(NewAccessRelation, BaseAddress,
                                     BBMap, GlobalMap);
  }

  isl_map_free(CurrentAccessRelation);
  isl_map_free(NewAccessRelation);
  return NewPointer;
}

Value *BlockGenerator::generateScalarLoad(const LoadInst *Load,
                                          ValueMapT &BBMap,
                                          ValueMapT &GlobalMap) {
  const Value *Pointer = Load->getPointerOperand();
  const Instruction *Inst = dyn_cast<Instruction>(Load);
  Value *NewPointer = generateLocationAccessed(Inst, Pointer, BBMap, GlobalMap);
  Value *ScalarLoad = Builder.CreateLoad(NewPointer,
                                         Load->getName() + "_p_scalar_");
  return ScalarLoad;
}

Value *BlockGenerator::generateScalarStore(const StoreInst *Store,
                                           ValueMapT &BBMap,
                                           ValueMapT &GlobalMap) {
  const Value *Pointer = Store->getPointerOperand();
  Value *NewPointer = generateLocationAccessed(Store, Pointer, BBMap,
                                               GlobalMap);
  Value *ValueOperand = getNewValue(Store->getValueOperand(), BBMap, GlobalMap);

  return Builder.CreateStore(ValueOperand, NewPointer);
}

void BlockGenerator::copyInstruction(const Instruction *Inst,
                                     ValueMapT &BBMap, ValueMapT &GlobalMap) {
  // Terminator instructions control the control flow. They are explicitly
  // expressed in the clast and do not need to be copied.
  if (Inst->isTerminator())
    return;

  if (isSCEVIgnore(Inst))
    return;

  if (const LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
    BBMap[Load] = generateScalarLoad(Load, BBMap, GlobalMap);
    return;
  }

  if (const StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
    BBMap[Store] = generateScalarStore(Store, BBMap, GlobalMap);
    return;
  }

  copyInstScalar(Inst, BBMap, GlobalMap);
}


void BlockGenerator::copyBB(ValueMapT &GlobalMap) {
  BasicBlock *BB = Statement.getBasicBlock();
  BasicBlock *CopyBB = SplitBlock(Builder.GetInsertBlock(),
                                  Builder.GetInsertPoint(), P);
  CopyBB->setName("polly.stmt." + BB->getName());
  Builder.SetInsertPoint(CopyBB->begin());

  ValueMapT BBMap;

  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end(); II != IE;
       ++II)
      copyInstruction(II, BBMap, GlobalMap);
}

VectorBlockGenerator::VectorBlockGenerator(IRBuilder<> &B,
  VectorValueMapT &GlobalMaps, ScopStmt &Stmt, __isl_keep isl_set *Domain,
  Pass *P) : BlockGenerator(B, Stmt, P), GlobalMaps(GlobalMaps),
  Domain(Domain) {
    assert(GlobalMaps.size() > 1 && "Only one vector lane found");
    assert(Domain && "No statement domain provided");
  }

Value *VectorBlockGenerator::getVectorValue(const Value *Old,
                                            ValueMapT &VectorMap,
                                            VectorValueMapT &ScalarMaps) {
  if (VectorMap.count(Old))
    return VectorMap[Old];

  int Width = getVectorWidth();

  Value *Vector = UndefValue::get(VectorType::get(Old->getType(), Width));

  for (int Lane = 0; Lane < Width; Lane++)
    Vector = Builder.CreateInsertElement(Vector,
                                         getNewValue(Old,
                                                     ScalarMaps[Lane],
                                                     GlobalMaps[Lane]),
                                         Builder.getInt32(Lane));

  VectorMap[Old] = Vector;

  return Vector;
}

Type *VectorBlockGenerator::getVectorPtrTy(const Value *Val, int Width) {
  PointerType *PointerTy = dyn_cast<PointerType>(Val->getType());
  assert(PointerTy && "PointerType expected");

  Type *ScalarType = PointerTy->getElementType();
  VectorType *VectorType = VectorType::get(ScalarType, Width);

  return PointerType::getUnqual(VectorType);
}

Value *VectorBlockGenerator::generateStrideOneLoad(const LoadInst *Load,
                                                   ValueMapT &BBMap) {
  const Value *Pointer = Load->getPointerOperand();
  Type *VectorPtrType = getVectorPtrTy(Pointer, getVectorWidth());
  Value *NewPointer = getNewValue(Pointer, BBMap, GlobalMaps[0]);
  Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                           "vector_ptr");
  LoadInst *VecLoad = Builder.CreateLoad(VectorPtr,
                                         Load->getName() + "_p_vec_full");
  if (!Aligned)
    VecLoad->setAlignment(8);

  return VecLoad;
}

Value *VectorBlockGenerator::generateStrideZeroLoad(const LoadInst *Load,
                                                    ValueMapT &BBMap) {
  const Value *Pointer = Load->getPointerOperand();
  Type *VectorPtrType = getVectorPtrTy(Pointer, 1);
  Value *NewPointer = getNewValue(Pointer, BBMap, GlobalMaps[0]);
  Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                           Load->getName() + "_p_vec_p");
  LoadInst *ScalarLoad= Builder.CreateLoad(VectorPtr,
                                           Load->getName() + "_p_splat_one");

  if (!Aligned)
    ScalarLoad->setAlignment(8);

  Constant *SplatVector =
    Constant::getNullValue(VectorType::get(Builder.getInt32Ty(),
                                           getVectorWidth()));

  Value *VectorLoad = Builder.CreateShuffleVector(ScalarLoad, ScalarLoad,
                                                  SplatVector,
                                                  Load->getName()
                                                  + "_p_splat");
  return VectorLoad;
}

Value *VectorBlockGenerator::generateUnknownStrideLoad(const LoadInst *Load,
  VectorValueMapT &ScalarMaps) {
  int VectorWidth = getVectorWidth();
  const Value *Pointer = Load->getPointerOperand();
  VectorType *VectorType = VectorType::get(
    dyn_cast<PointerType>(Pointer->getType())->getElementType(), VectorWidth);

  Value *Vector = UndefValue::get(VectorType);

  for (int i = 0; i < VectorWidth; i++) {
    Value *NewPointer = getNewValue(Pointer, ScalarMaps[i], GlobalMaps[i]);
    Value *ScalarLoad = Builder.CreateLoad(NewPointer,
                                           Load->getName() + "_p_scalar_");
    Vector = Builder.CreateInsertElement(Vector, ScalarLoad,
                                         Builder.getInt32(i),
                                         Load->getName() + "_p_vec_");
  }

  return Vector;
}

void VectorBlockGenerator::generateLoad(const LoadInst *Load,
                                        ValueMapT &VectorMap,
                                        VectorValueMapT &ScalarMaps) {
  if (PollyVectorizerChoice >= VECTORIZER_FIRST_NEED_GROUPED_UNROLL ||
      !VectorType::isValidElementType(Load->getType())) {
    for (int i = 0; i < getVectorWidth(); i++)
      ScalarMaps[i][Load] = generateScalarLoad(Load, ScalarMaps[i],
                                               GlobalMaps[i]);
    return;
  }

  MemoryAccess &Access = Statement.getAccessFor(Load);

  Value *NewLoad;
  if (Access.isStrideZero(isl_set_copy(Domain)))
    NewLoad = generateStrideZeroLoad(Load, ScalarMaps[0]);
  else if (Access.isStrideOne(isl_set_copy(Domain)))
    NewLoad = generateStrideOneLoad(Load, ScalarMaps[0]);
  else
    NewLoad = generateUnknownStrideLoad(Load, ScalarMaps);

  VectorMap[Load] = NewLoad;
}

void VectorBlockGenerator::copyUnaryInst(const UnaryInstruction *Inst,
                                         ValueMapT &VectorMap,
                                         VectorValueMapT &ScalarMaps) {
  int VectorWidth = getVectorWidth();
  Value *NewOperand = getVectorValue(Inst->getOperand(0), VectorMap,
                                     ScalarMaps);

  assert(isa<CastInst>(Inst) && "Can not generate vector code for instruction");

  const CastInst *Cast = dyn_cast<CastInst>(Inst);
  VectorType *DestType = VectorType::get(Inst->getType(), VectorWidth);
  VectorMap[Inst] = Builder.CreateCast(Cast->getOpcode(), NewOperand, DestType);
}

void VectorBlockGenerator::copyBinaryInst(const BinaryOperator *Inst,
                                          ValueMapT &VectorMap,
                                          VectorValueMapT &ScalarMaps) {
  Value *OpZero = Inst->getOperand(0);
  Value *OpOne = Inst->getOperand(1);

  Value *NewOpZero, *NewOpOne;
  NewOpZero = getVectorValue(OpZero, VectorMap, ScalarMaps);
  NewOpOne = getVectorValue(OpOne, VectorMap, ScalarMaps);

  Value *NewInst = Builder.CreateBinOp(Inst->getOpcode(), NewOpZero,
                                       NewOpOne,
                                       Inst->getName() + "p_vec");
  VectorMap[Inst] = NewInst;
}

void VectorBlockGenerator::copyStore(const StoreInst *Store,
                                     ValueMapT &VectorMap,
                                     VectorValueMapT &ScalarMaps) {
  int VectorWidth = getVectorWidth();

  MemoryAccess &Access = Statement.getAccessFor(Store);

  const Value *Pointer = Store->getPointerOperand();
  Value *Vector = getVectorValue(Store->getValueOperand(), VectorMap,
                                   ScalarMaps);

  if (Access.isStrideOne(isl_set_copy(Domain))) {
    Type *VectorPtrType = getVectorPtrTy(Pointer, VectorWidth);
    Value *NewPointer = getNewValue(Pointer, ScalarMaps[0], GlobalMaps[0]);

    Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                             "vector_ptr");
    StoreInst *Store = Builder.CreateStore(Vector, VectorPtr);

    if (!Aligned)
      Store->setAlignment(8);
  } else {
    for (unsigned i = 0; i < ScalarMaps.size(); i++) {
      Value *Scalar = Builder.CreateExtractElement(Vector,
                                                   Builder.getInt32(i));
      Value *NewPointer = getNewValue(Pointer, ScalarMaps[i], GlobalMaps[i]);
      Builder.CreateStore(Scalar, NewPointer);
    }
  }
}

bool VectorBlockGenerator::hasVectorOperands(const Instruction *Inst,
                                             ValueMapT &VectorMap) {
  for (Instruction::const_op_iterator OI = Inst->op_begin(),
       OE = Inst->op_end(); OI != OE; ++OI)
    if (VectorMap.count(*OI))
      return true;
  return false;
}

bool VectorBlockGenerator::extractScalarValues(const Instruction *Inst,
                                               ValueMapT &VectorMap,
                                               VectorValueMapT &ScalarMaps) {
  bool HasVectorOperand = false;
  int VectorWidth = getVectorWidth();

  for (Instruction::const_op_iterator OI = Inst->op_begin(),
       OE = Inst->op_end(); OI != OE; ++OI) {
    ValueMapT::iterator VecOp = VectorMap.find(*OI);

    if (VecOp == VectorMap.end())
      continue;

    HasVectorOperand = true;
    Value *NewVector = VecOp->second;

    for (int i = 0; i < VectorWidth; ++i) {
      ValueMapT &SM = ScalarMaps[i];

      // If there is one scalar extracted, all scalar elements should have
      // already been extracted by the code here. So no need to check for the
      // existance of all of them.
      if (SM.count(*OI))
        break;

      SM[*OI] = Builder.CreateExtractElement(NewVector, Builder.getInt32(i));
    }
  }

  return HasVectorOperand;
}

void VectorBlockGenerator::copyInstScalarized(const Instruction *Inst,
                                              ValueMapT &VectorMap,
                                              VectorValueMapT &ScalarMaps) {
  bool HasVectorOperand;
  int VectorWidth = getVectorWidth();

  HasVectorOperand = extractScalarValues(Inst, VectorMap, ScalarMaps);

  for (int VectorLane = 0; VectorLane < getVectorWidth(); VectorLane++)
    copyInstScalar(Inst, ScalarMaps[VectorLane], GlobalMaps[VectorLane]);

  if (!VectorType::isValidElementType(Inst->getType()) || !HasVectorOperand)
    return;

  // Make the result available as vector value.
  VectorType *VectorType = VectorType::get(Inst->getType(), VectorWidth);
  Value *Vector = UndefValue::get(VectorType);

  for (int i = 0; i < VectorWidth; i++)
    Vector = Builder.CreateInsertElement(Vector, ScalarMaps[i][Inst],
                                         Builder.getInt32(i));

  VectorMap[Inst] = Vector;
}

int VectorBlockGenerator::getVectorWidth() {
  return GlobalMaps.size();
}

void VectorBlockGenerator::copyInstruction(const Instruction *Inst,
                                           ValueMapT &VectorMap,
                                           VectorValueMapT &ScalarMaps) {
  // Terminator instructions control the control flow. They are explicitly
  // expressed in the clast and do not need to be copied.
  if (Inst->isTerminator())
    return;

  if (isSCEVIgnore(Inst))
    return;

  if (const LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
    generateLoad(Load, VectorMap, ScalarMaps);
    return;
  }

  if (hasVectorOperands(Inst, VectorMap)) {
    if (const StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
      copyStore(Store, VectorMap, ScalarMaps);
      return;
    }

    if (const UnaryInstruction *Unary = dyn_cast<UnaryInstruction>(Inst)) {
      copyUnaryInst(Unary, VectorMap, ScalarMaps);
      return;
    }

    if (const BinaryOperator *Binary = dyn_cast<BinaryOperator>(Inst)) {
      copyBinaryInst(Binary, VectorMap, ScalarMaps);
      return;
    }

    // Falltrough: We generate scalar instructions, if we don't know how to
    // generate vector code.
  }

  copyInstScalarized(Inst, VectorMap, ScalarMaps);
}

void VectorBlockGenerator::copyBB() {
  BasicBlock *BB = Statement.getBasicBlock();
  BasicBlock *CopyBB = SplitBlock(Builder.GetInsertBlock(),
                                  Builder.GetInsertPoint(), P);
  CopyBB->setName("polly.stmt." + BB->getName());
  Builder.SetInsertPoint(CopyBB->begin());

  // Create two maps that store the mapping from the original instructions of
  // the old basic block to their copies in the new basic block. Those maps
  // are basic block local.
  //
  // As vector code generation is supported there is one map for scalar values
  // and one for vector values.
  //
  // In case we just do scalar code generation, the vectorMap is not used and
  // the scalarMap has just one dimension, which contains the mapping.
  //
  // In case vector code generation is done, an instruction may either appear
  // in the vector map once (as it is calculating >vectorwidth< values at a
  // time. Or (if the values are calculated using scalar operations), it
  // appears once in every dimension of the scalarMap.
  VectorValueMapT ScalarBlockMap(getVectorWidth());
  ValueMapT VectorBlockMap;

  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
       II != IE; ++II)
      copyInstruction(II, VectorBlockMap, ScalarBlockMap);
}
