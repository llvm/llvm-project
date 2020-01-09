//===- LowerMatrixIntrinsics.cpp -  Lower matrix intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower matrix intrinsics to vector operations.
//
// TODO:
//  * Implement multiply & add fusion
//  * Add remark, summarizing the available matrix optimization opportunities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "lower-matrix-intrinsics"

static cl::opt<bool> EnableShapePropagation("matrix-propagate-shape",
                                            cl::init(true));

static cl::opt<bool> AllowContractEnabled(
    "matrix-allow-contract", cl::init(false), cl::Hidden,
    cl::desc("Allow the use of FMAs if available and profitable. This may "
             "result in different results, due to less rounding error."));

namespace {

// Given an element poitner \p BasePtr to the start of a (sub) matrix, compute
// the start address of column \p Col with type (\p EltType x \p NumRows)
// assuming \p Stride elements between start two consecutive columns.
// \p Stride must be >= \p NumRows.
//
// Consider a 4x4 matrix like below
//
//      0       1      2      3
// 0   v_0_0  v_0_1  v_0_2  v_0_3
// 1   v_1_0  v_1_1  v_1_2  v_1_3
// 2   v_2_0  v_2_1  v_2_2  v_2_3
// 3   v_3_0  v_3_1  v_3_2  v_3_3

// To compute the column addresses for a 2x3 sub-matrix at row 1 and column 1,
// we need a pointer to the first element of the submatrix as base pointer.
// Then we can use computeColumnAddr to compute the addresses for the columns
// of the sub-matrix.
//
// Column 0: computeColumnAddr(Base, 0 (column), 4 (stride), 2 (num rows), ..)
//           -> just returns Base
// Column 1: computeColumnAddr(Base, 1 (column), 4 (stride), 2 (num rows), ..)
//           -> returns Base + (1 * 4)
// Column 2: computeColumnAddr(Base, 2 (column), 4 (stride), 2 (num rows), ..)
//           -> returns Base + (2 * 4)
//
// The graphic below illustrates the number of elements in a column (marked
// with |) and the number of skipped elements (marked with }).
//
//         v_0_0  v_0_1 {v_0_2 {v_0_3
//                Base   Col 1  Col 2
//                  |     |      |
//         v_1_0 |v_1_1 |v_1_2 |v_1_3
//         v_2_0 |v_2_1 |v_2_2 |v_2_3
//         v_3_0 {v_3_1 {v_3_2  v_3_3
//
Value *computeColumnAddr(Value *BasePtr, Value *Col, Value *Stride,
                         unsigned NumRows, Type *EltType,
                         IRBuilder<> &Builder) {

  assert((!isa<ConstantInt>(Stride) ||
          cast<ConstantInt>(Stride)->getZExtValue() >= NumRows) &&
         "Stride must be >= the number of rows.");
  unsigned AS = cast<PointerType>(BasePtr->getType())->getAddressSpace();

  // Compute the start of the column with index Col as Col * Stride.
  Value *ColumnStart = Builder.CreateMul(Col, Stride, "col.start");

  // Get pointer to the start of the selected column. Skip GEP creation,
  // if we select column 0.
  if (isa<ConstantInt>(ColumnStart) && cast<ConstantInt>(ColumnStart)->isZero())
    ColumnStart = BasePtr;
  else
    ColumnStart = Builder.CreateGEP(EltType, BasePtr, ColumnStart, "col.gep");

  // Cast elementwise column start pointer to a pointer to a column
  // (EltType x NumRows)*.
  Type *ColumnType = VectorType::get(EltType, NumRows);
  Type *ColumnPtrType = PointerType::get(ColumnType, AS);
  return Builder.CreatePointerCast(ColumnStart, ColumnPtrType, "col.cast");
}

/// LowerMatrixIntrinsics contains the methods used to lower matrix intrinsics.
///
/// Currently, the lowering for each matrix intrinsic is done as follows:
/// 1. Propagate the shape information from intrinsics to connected
/// instructions.
/// 2. Lower instructions with shape information.
///  2.1. Get column vectors for each argument. If we already lowered the
///       definition of an argument, use the produced column vectors directly.
///       If not, split the operand vector containing an embedded matrix into
///       a set of column vectors,
///  2.2. Lower the instruction in terms of columnwise operations, which yields
///       a set of column vectors containing result matrix. Note that we lower
///       all instructions that have shape information. Besides the intrinsics,
///       this includes stores for example.
///  2.3. Update uses of the lowered instruction. If we have shape information
///       for a user, there is nothing to do, as we will look up the result
///       column matrix when lowering the user. For other uses, we embed the
///       result matrix in a flat vector and update the use.
///  2.4. Cache the result column matrix for the instruction we lowered
/// 3. After we lowered all instructions in a function, remove the now
///    obsolete instructions.
///
class LowerMatrixIntrinsics {
  Function &Func;
  const DataLayout &DL;
  const TargetTransformInfo &TTI;

  /// Wrapper class representing a matrix as a set of column vectors.
  /// All column vectors must have the same vector type.
  class ColumnMatrixTy {
    SmallVector<Value *, 16> Columns;

  public:
    ColumnMatrixTy() : Columns() {}
    ColumnMatrixTy(ArrayRef<Value *> Cols)
        : Columns(Cols.begin(), Cols.end()) {}

    Value *getColumn(unsigned i) const { return Columns[i]; }

    void setColumn(unsigned i, Value *V) { Columns[i] = V; }

    size_t getNumColumns() const { return Columns.size(); }
    size_t getNumRows() const {
      assert(Columns.size() > 0 && "Cannot call getNumRows without columns");
      return cast<VectorType>(Columns[0]->getType())->getNumElements();
    }

    const SmallVectorImpl<Value *> &getColumnVectors() const { return Columns; }

    SmallVectorImpl<Value *> &getColumnVectors() { return Columns; }

    void addColumn(Value *V) { Columns.push_back(V); }

    iterator_range<SmallVector<Value *, 8>::iterator> columns() {
      return make_range(Columns.begin(), Columns.end());
    }

    /// Embed the columns of the matrix into a flat vector by concatenating
    /// them.
    Value *embedInVector(IRBuilder<> &Builder) const {
      return Columns.size() == 1 ? Columns[0]
                                 : concatenateVectors(Builder, Columns);
    }
  };

  struct ShapeInfo {
    unsigned NumRows;
    unsigned NumColumns;

    ShapeInfo(unsigned NumRows = 0, unsigned NumColumns = 0)
        : NumRows(NumRows), NumColumns(NumColumns) {}

    ShapeInfo(Value *NumRows, Value *NumColumns)
        : NumRows(cast<ConstantInt>(NumRows)->getZExtValue()),
          NumColumns(cast<ConstantInt>(NumColumns)->getZExtValue()) {}

    bool operator==(const ShapeInfo &other) {
      return NumRows == other.NumRows && NumColumns == other.NumColumns;
    }
    bool operator!=(const ShapeInfo &other) { return !(*this == other); }

    /// Returns true if shape-information is defined, meaning both dimensions
    /// are != 0.
    operator bool() const {
      assert(NumRows == 0 || NumColumns != 0);
      return NumRows != 0;
    }
  };

  /// Maps instructions to their shape information. The shape information
  /// describes the shape to be used while lowering. This matches the shape of
  /// the result value of the instruction, with the only exceptions being store
  /// instructions and the matrix_columnwise_store intrinsics. For those, the
  /// shape information indicates that those instructions should be lowered
  /// using shape information as well.
  DenseMap<Value *, ShapeInfo> ShapeMap;

  /// List of instructions to remove. While lowering, we are not replacing all
  /// users of a lowered instruction, if shape information is available and
  /// those need to be removed after we finished lowering.
  SmallVector<Instruction *, 16> ToRemove;

  /// Map from instructions to their produced column matrix.
  DenseMap<Value *, ColumnMatrixTy> Inst2ColumnMatrix;

public:
  LowerMatrixIntrinsics(Function &F, TargetTransformInfo &TTI)
      : Func(F), DL(F.getParent()->getDataLayout()), TTI(TTI) {}

  /// Return the set of column vectors that a matrix value is lowered to.
  ///
  /// If we lowered \p MatrixVal, just return the cache result column matrix.
  /// Otherwie split the flat vector \p MatrixVal containing a matrix with
  /// shape \p SI into column vectors.
  ColumnMatrixTy getMatrix(Value *MatrixVal, const ShapeInfo &SI,
                           IRBuilder<> Builder) {
    VectorType *VType = dyn_cast<VectorType>(MatrixVal->getType());
    assert(VType && "MatrixVal must be a vector type");
    assert(VType->getNumElements() == SI.NumRows * SI.NumColumns &&
           "The vector size must match the number of matrix elements");

    // Check if we lowered MatrixVal using shape information. In that case,
    // return the existing column matrix, if it matches the requested shape
    // information. If there is a mis-match, embed the result in a flat
    // vector and split it later.
    auto Found = Inst2ColumnMatrix.find(MatrixVal);
    if (Found != Inst2ColumnMatrix.end()) {
      ColumnMatrixTy &M = Found->second;
      // Return the found matrix, if its shape matches the requested shape
      // information
      if (SI.NumRows == M.getNumRows() && SI.NumColumns == M.getNumColumns())
        return M;

      MatrixVal = M.embedInVector(Builder);
    }

    // Otherwise split MatrixVal.
    SmallVector<Value *, 16> SplitVecs;
    Value *Undef = UndefValue::get(VType);
    for (unsigned MaskStart = 0; MaskStart < VType->getNumElements();
         MaskStart += SI.NumRows) {
      Constant *Mask = createSequentialMask(Builder, MaskStart, SI.NumRows, 0);
      Value *V = Builder.CreateShuffleVector(MatrixVal, Undef, Mask, "split");
      SplitVecs.push_back(V);
    }

    return {SplitVecs};
  }

  /// If \p V already has a known shape return false.  Otherwise set the shape
  /// for instructions that support it.
  bool setShapeInfo(Value *V, ShapeInfo Shape) {
    assert(Shape && "Shape not set");
    if (isa<UndefValue>(V) || !supportsShapeInfo(V))
      return false;

    auto SIter = ShapeMap.find(V);
    if (SIter != ShapeMap.end()) {
      LLVM_DEBUG(dbgs() << "  not overriding existing shape: "
                        << SIter->second.NumRows << " "
                        << SIter->second.NumColumns << " for " << *V << "\n");
      return false;
    }

    ShapeMap.insert({V, Shape});
    LLVM_DEBUG(dbgs() << "  " << Shape.NumRows << " x " << Shape.NumColumns
                      << " for " << *V << "\n");
    return true;
  }

  bool isUniformShape(Value *V) {
    Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      return true;

    switch (I->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul: // Scalar multiply.
    case Instruction::Add:
    case Instruction::Mul:
    case Instruction::Sub:
      return true;
    default:
      return false;
    }
  }

  /// Returns true if shape information can be used for \p V. The supported
  /// instructions must match the instructions that can be lowered by this pass.
  bool supportsShapeInfo(Value *V) {
    Instruction *Inst = dyn_cast<Instruction>(V);
    if (!Inst)
      return false;

    IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst);
    if (II)
      switch (II->getIntrinsicID()) {
      case Intrinsic::matrix_multiply:
      case Intrinsic::matrix_transpose:
      case Intrinsic::matrix_columnwise_load:
      case Intrinsic::matrix_columnwise_store:
        return true;
      default:
        return false;
      }
    return isUniformShape(V) || isa<StoreInst>(V) || isa<LoadInst>(V);
  }

  /// Propagate the shape information of instructions to their users.
  /// The work list contains instructions for which we can compute the shape,
  /// either based on the information provided by matrix intrinsics or known
  /// shapes of operands.
  SmallVector<Instruction *, 32>
  propagateShapeForward(SmallVectorImpl<Instruction *> &WorkList) {
    SmallVector<Instruction *, 32> NewWorkList;
    // Pop an element for which we guaranteed to have at least one of the
    // operand shapes.  Add the shape for this and then add users to the work
    // list.
    LLVM_DEBUG(dbgs() << "Forward-propagate shapes:\n");
    while (!WorkList.empty()) {
      Instruction *Inst = WorkList.back();
      WorkList.pop_back();

      // New entry, set the value and insert operands
      bool Propagate = false;

      Value *MatrixA;
      Value *MatrixB;
      Value *M;
      Value *N;
      Value *K;
      if (match(Inst, m_Intrinsic<Intrinsic::matrix_multiply>(
                          m_Value(MatrixA), m_Value(MatrixB), m_Value(M),
                          m_Value(N), m_Value(K)))) {
        Propagate = setShapeInfo(Inst, {M, K});
      } else if (match(Inst, m_Intrinsic<Intrinsic::matrix_transpose>(
                                 m_Value(MatrixA), m_Value(M), m_Value(N)))) {
        // Flip dimensions.
        Propagate = setShapeInfo(Inst, {N, M});
      } else if (match(Inst, m_Intrinsic<Intrinsic::matrix_columnwise_store>(
                                 m_Value(MatrixA), m_Value(), m_Value(),
                                 m_Value(M), m_Value(N)))) {
        Propagate = setShapeInfo(Inst, {N, M});
      } else if (match(Inst,
                       m_Intrinsic<Intrinsic::matrix_columnwise_load>(
                           m_Value(), m_Value(), m_Value(M), m_Value(N)))) {
        Propagate = setShapeInfo(Inst, {M, N});
      } else if (match(Inst, m_Store(m_Value(MatrixA), m_Value()))) {
        auto OpShape = ShapeMap.find(MatrixA);
        if (OpShape != ShapeMap.end())
          setShapeInfo(Inst, OpShape->second);
        continue;
      } else if (isUniformShape(Inst)) {
        // Find the first operand that has a known shape and use that.
        for (auto &Op : Inst->operands()) {
          auto OpShape = ShapeMap.find(Op.get());
          if (OpShape != ShapeMap.end()) {
            Propagate |= setShapeInfo(Inst, OpShape->second);
            break;
          }
        }
      }

      if (Propagate) {
        NewWorkList.push_back(Inst);
        for (auto *User : Inst->users())
          if (ShapeMap.count(User) == 0)
            WorkList.push_back(cast<Instruction>(User));
      }
    }

    return NewWorkList;
  }

  /// Propagate the shape to operands of instructions with shape information.
  /// \p Worklist contains the instruction for which we already know the shape.
  SmallVector<Instruction *, 32>
  propagateShapeBackward(SmallVectorImpl<Instruction *> &WorkList) {
    SmallVector<Instruction *, 32> NewWorkList;

    auto pushInstruction = [](Value *V,
                              SmallVectorImpl<Instruction *> &WorkList) {
      Instruction *I = dyn_cast<Instruction>(V);
      if (I)
        WorkList.push_back(I);
    };
    // Pop an element with known shape.  Traverse the operands, if their shape
    // derives from the result shape and is unknown, add it and add them to the
    // worklist.
    LLVM_DEBUG(dbgs() << "Backward-propagate shapes:\n");
    while (!WorkList.empty()) {
      Value *V = WorkList.back();
      WorkList.pop_back();

      size_t BeforeProcessingV = WorkList.size();
      if (!isa<Instruction>(V))
        continue;

      Value *MatrixA;
      Value *MatrixB;
      Value *M;
      Value *N;
      Value *K;
      if (match(V, m_Intrinsic<Intrinsic::matrix_multiply>(
                       m_Value(MatrixA), m_Value(MatrixB), m_Value(M),
                       m_Value(N), m_Value(K)))) {
        if (setShapeInfo(MatrixA, {M, N}))
          pushInstruction(MatrixA, WorkList);

        if (setShapeInfo(MatrixB, {N, K}))
          pushInstruction(MatrixB, WorkList);

      } else if (match(V, m_Intrinsic<Intrinsic::matrix_transpose>(
                              m_Value(MatrixA), m_Value(M), m_Value(N)))) {
        // Flip dimensions.
        if (setShapeInfo(MatrixA, {M, N}))
          pushInstruction(MatrixA, WorkList);
      } else if (match(V, m_Intrinsic<Intrinsic::matrix_columnwise_store>(
                              m_Value(MatrixA), m_Value(), m_Value(),
                              m_Value(M), m_Value(N)))) {
        if (setShapeInfo(MatrixA, {M, N})) {
          pushInstruction(MatrixA, WorkList);
        }
      } else if (isa<LoadInst>(V) ||
                 match(V, m_Intrinsic<Intrinsic::matrix_columnwise_load>())) {
        // Nothing to do, no matrix input.
      } else if (isa<StoreInst>(V)) {
        // Nothing to do.  We forward-propagated to this so we would just
        // backward propagate to an instruction with an already known shape.
      } else if (isUniformShape(V)) {
        // Propagate to all operands.
        ShapeInfo Shape = ShapeMap[V];
        for (Use &U : cast<Instruction>(V)->operands()) {
          if (setShapeInfo(U.get(), Shape))
            pushInstruction(U.get(), WorkList);
        }
      }
      // After we discovered new shape info for new instructions in the
      // worklist, we use their users as seeds for the next round of forward
      // propagation.
      for (size_t I = BeforeProcessingV; I != WorkList.size(); I++)
        for (User *U : WorkList[I]->users())
          if (isa<Instruction>(U) && V != U)
            NewWorkList.push_back(cast<Instruction>(U));
    }
    return NewWorkList;
  }

  bool Visit() {
    if (EnableShapePropagation) {
      SmallVector<Instruction *, 32> WorkList;

      // Initially only the shape of matrix intrinsics is known.
      // Initialize the work list with ops carrying shape information.
      for (BasicBlock &BB : Func)
        for (Instruction &Inst : BB) {
          IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
          if (!II)
            continue;

          switch (II->getIntrinsicID()) {
          case Intrinsic::matrix_multiply:
          case Intrinsic::matrix_transpose:
          case Intrinsic::matrix_columnwise_load:
          case Intrinsic::matrix_columnwise_store:
            WorkList.push_back(&Inst);
            break;
          default:
            break;
          }
        }
      // Propagate shapes until nothing changes any longer.
      while (!WorkList.empty()) {
        WorkList = propagateShapeForward(WorkList);
        WorkList = propagateShapeBackward(WorkList);
      }
    }

    ReversePostOrderTraversal<Function *> RPOT(&Func);
    bool Changed = false;
    for (auto *BB : RPOT) {
      for (Instruction &Inst : make_early_inc_range(*BB)) {
        IRBuilder<> Builder(&Inst);

        if (CallInst *CInst = dyn_cast<CallInst>(&Inst))
          Changed |= VisitCallInst(CInst);

        Value *Op1;
        Value *Op2;
        if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst))
          Changed |= VisitBinaryOperator(BinOp);
        if (match(&Inst, m_Load(m_Value(Op1))))
          Changed |= VisitLoad(&Inst, Op1, Builder);
        else if (match(&Inst, m_Store(m_Value(Op1), m_Value(Op2))))
          Changed |= VisitStore(&Inst, Op1, Op2, Builder);
      }
    }

    for (Instruction *Inst : reverse(ToRemove))
      Inst->eraseFromParent();

    return Changed;
  }

  LoadInst *createColumnLoad(Value *ColumnPtr, Type *EltType,
                             IRBuilder<> Builder) {
    unsigned Align = DL.getABITypeAlignment(EltType);
    return Builder.CreateAlignedLoad(ColumnPtr, Align, "col.load");
  }

  StoreInst *createColumnStore(Value *ColumnValue, Value *ColumnPtr,
                               Type *EltType, IRBuilder<> Builder) {
    unsigned Align = DL.getABITypeAlignment(EltType);
    return Builder.CreateAlignedStore(ColumnValue, ColumnPtr, Align);
  }


  /// Turns \p BasePtr into an elementwise pointer to \p EltType.
  Value *createElementPtr(Value *BasePtr, Type *EltType, IRBuilder<> &Builder) {
    unsigned AS = cast<PointerType>(BasePtr->getType())->getAddressSpace();
    Type *EltPtrType = PointerType::get(EltType, AS);
    return Builder.CreatePointerCast(BasePtr, EltPtrType);
  }

  /// Replace intrinsic calls
  bool VisitCallInst(CallInst *Inst) {
    if (!Inst->getCalledFunction() || !Inst->getCalledFunction()->isIntrinsic())
      return false;

    switch (Inst->getCalledFunction()->getIntrinsicID()) {
    case Intrinsic::matrix_multiply:
      LowerMultiply(Inst);
      break;
    case Intrinsic::matrix_transpose:
      LowerTranspose(Inst);
      break;
    case Intrinsic::matrix_columnwise_load:
      LowerColumnwiseLoad(Inst);
      break;
    case Intrinsic::matrix_columnwise_store:
      LowerColumnwiseStore(Inst);
      break;
    default:
      return false;
    }
    return true;
  }

  void LowerLoad(Instruction *Inst, Value *Ptr, Value *Stride,
                 ShapeInfo Shape) {
    IRBuilder<> Builder(Inst);
    auto VType = cast<VectorType>(Inst->getType());
    Value *EltPtr = createElementPtr(Ptr, VType->getElementType(), Builder);
    ColumnMatrixTy Result;
    // Distance between start of one column and the start of the next
    for (unsigned C = 0, E = Shape.NumColumns; C < E; ++C) {
      Value *GEP =
          computeColumnAddr(EltPtr, Builder.getInt32(C), Stride, Shape.NumRows,
                            VType->getElementType(), Builder);
      Value *Column = createColumnLoad(GEP, VType->getElementType(), Builder);
      Result.addColumn(Column);
    }

    finalizeLowering(Inst, Result, Builder);
  }

  /// Lowers llvm.matrix.columnwise.load.
  ///
  /// The intrinsic loads a matrix from memory using a stride between columns.
  void LowerColumnwiseLoad(CallInst *Inst) {
    Value *Ptr = Inst->getArgOperand(0);
    Value *Stride = Inst->getArgOperand(1);
    LowerLoad(Inst, Ptr, Stride,
              {Inst->getArgOperand(2), Inst->getArgOperand(3)});
  }

  void LowerStore(Instruction *Inst, Value *Matrix, Value *Ptr, Value *Stride,
                  ShapeInfo Shape) {
    IRBuilder<> Builder(Inst);
    auto VType = cast<VectorType>(Matrix->getType());
    Value *EltPtr = createElementPtr(Ptr, VType->getElementType(), Builder);
    auto LM = getMatrix(Matrix, Shape, Builder);
    for (auto C : enumerate(LM.columns())) {
      Value *GEP =
          computeColumnAddr(EltPtr, Builder.getInt32(C.index()), Stride,
                            Shape.NumRows, VType->getElementType(), Builder);
      createColumnStore(C.value(), GEP, VType->getElementType(), Builder);
    }

    ToRemove.push_back(Inst);
  }

  /// Lowers llvm.matrix.columnwise.store.
  ///
  /// The intrinsic store a matrix back memory using a stride between columns.
  void LowerColumnwiseStore(CallInst *Inst) {
    Value *Matrix = Inst->getArgOperand(0);
    Value *Ptr = Inst->getArgOperand(1);
    Value *Stride = Inst->getArgOperand(2);
    LowerStore(Inst, Matrix, Ptr, Stride,
               {Inst->getArgOperand(3), Inst->getArgOperand(4)});
  }

  /// Extract a column vector of \p NumElts starting at index (\p I, \p J) from
  /// the matrix \p LM represented as a vector of column vectors.
  Value *extractVector(const ColumnMatrixTy &LM, unsigned I, unsigned J,
                       unsigned NumElts, IRBuilder<> Builder) {
    Value *Col = LM.getColumn(J);
    Value *Undef = UndefValue::get(Col->getType());
    Constant *Mask = createSequentialMask(Builder, I, NumElts, 0);
    return Builder.CreateShuffleVector(Col, Undef, Mask, "block");
  }

  // Set elements I..I+NumElts-1 to Block
  Value *insertVector(Value *Col, unsigned I, Value *Block,
                      IRBuilder<> Builder) {

    // First, bring Block to the same size as Col
    unsigned BlockNumElts =
        cast<VectorType>(Block->getType())->getNumElements();
    unsigned NumElts = cast<VectorType>(Col->getType())->getNumElements();
    assert(NumElts >= BlockNumElts && "Too few elements for current block");

    Value *ExtendMask =
        createSequentialMask(Builder, 0, BlockNumElts, NumElts - BlockNumElts);
    Value *Undef = UndefValue::get(Block->getType());
    Block = Builder.CreateShuffleVector(Block, Undef, ExtendMask);

    // If Col is 7 long and I is 2 and BlockNumElts is 2 the mask is: 0, 1, 7,
    // 8, 4, 5, 6
    SmallVector<Constant *, 16> Mask;
    unsigned i;
    for (i = 0; i < I; i++)
      Mask.push_back(Builder.getInt32(i));

    unsigned VecNumElts = cast<VectorType>(Col->getType())->getNumElements();
    for (; i < I + BlockNumElts; i++)
      Mask.push_back(Builder.getInt32(i - I + VecNumElts));

    for (; i < VecNumElts; i++)
      Mask.push_back(Builder.getInt32(i));

    Value *MaskVal = ConstantVector::get(Mask);

    return Builder.CreateShuffleVector(Col, Block, MaskVal);
  }

  Value *createMulAdd(Value *Sum, Value *A, Value *B, bool UseFPOp,
                      IRBuilder<> &Builder, bool AllowContraction) {

    if (!Sum)
      return UseFPOp ? Builder.CreateFMul(A, B) : Builder.CreateMul(A, B);

    if (UseFPOp) {
      if (AllowContraction) {
        // Use fmuladd for floating point operations and let the backend decide
        // if that's profitable.
        Value *FMulAdd = Intrinsic::getDeclaration(
            Func.getParent(), Intrinsic::fmuladd, A->getType());
        return Builder.CreateCall(FMulAdd, {A, B, Sum});
      }
      Value *Mul = Builder.CreateFMul(A, B);
      return Builder.CreateFAdd(Sum, Mul);
    }

    Value *Mul = Builder.CreateMul(A, B);
    return Builder.CreateAdd(Sum, Mul);
  }

  /// Cache \p Matrix as result of \p Inst and update the uses of \p Inst. For
  /// users with shape information, there's nothing to do: the will use the
  /// cached value when they are lowered. For other users, \p Matrix is
  /// flattened and the uses are updated to use it. Also marks \p Inst for
  /// deletion.
  void finalizeLowering(Instruction *Inst, ColumnMatrixTy Matrix,
                        IRBuilder<> &Builder) {
    Inst2ColumnMatrix.insert(std::make_pair(Inst, Matrix));

    ToRemove.push_back(Inst);
    Value *Flattened = nullptr;
    for (auto I = Inst->use_begin(), E = Inst->use_end(); I != E;) {
      Use &U = *I++;
      if (ShapeMap.find(U.getUser()) == ShapeMap.end()) {
        if (!Flattened)
          Flattened = Matrix.embedInVector(Builder);
        U.set(Flattened);
      }
    }
  }

  /// Lowers llvm.matrix.multiply.
  void LowerMultiply(CallInst *MatMul) {
    IRBuilder<> Builder(MatMul);
    auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();
    ShapeInfo LShape(MatMul->getArgOperand(2), MatMul->getArgOperand(3));
    ShapeInfo RShape(MatMul->getArgOperand(3), MatMul->getArgOperand(4));

    const ColumnMatrixTy &Lhs =
        getMatrix(MatMul->getArgOperand(0), LShape, Builder);
    const ColumnMatrixTy &Rhs =
        getMatrix(MatMul->getArgOperand(1), RShape, Builder);

    const unsigned R = LShape.NumRows;
    const unsigned M = LShape.NumColumns;
    const unsigned C = RShape.NumColumns;
    assert(M == RShape.NumRows);

    // Initialize the output
    ColumnMatrixTy Result;
    for (unsigned J = 0; J < C; ++J)
      Result.addColumn(UndefValue::get(VectorType::get(EltType, R)));

    const unsigned VF = std::max(TTI.getRegisterBitWidth(true) /
                                     EltType->getPrimitiveSizeInBits(),
                                 uint64_t(1));

    bool AllowContract = AllowContractEnabled || (isa<FPMathOperator>(MatMul) &&
                                                  MatMul->hasAllowContract());
    // Multiply columns from the first operand with scalars from the second
    // operand.  Then move along the K axes and accumulate the columns.  With
    // this the adds can be vectorized without reassociation.
    for (unsigned J = 0; J < C; ++J) {
      unsigned BlockSize = VF;
      for (unsigned I = 0; I < R; I += BlockSize) {
        // Gradually lower the vectorization factor to cover the remainder.
        while (I + BlockSize > R)
          BlockSize /= 2;

        Value *Sum = nullptr;
        for (unsigned K = 0; K < M; ++K) {
          Value *L = extractVector(Lhs, I, K, BlockSize, Builder);
          Value *RH = Builder.CreateExtractElement(Rhs.getColumn(J), K);
          Value *Splat = Builder.CreateVectorSplat(BlockSize, RH, "splat");
          Sum = createMulAdd(Sum, L, Splat, EltType->isFloatingPointTy(),
                             Builder, AllowContract);
        }
        Result.setColumn(J, insertVector(Result.getColumn(J), I, Sum, Builder));
      }
    }
    finalizeLowering(MatMul, Result, Builder);
  }

  /// Lowers llvm.matrix.transpose.
  void LowerTranspose(CallInst *Inst) {
    ColumnMatrixTy Result;
    IRBuilder<> Builder(Inst);
    Value *InputVal = Inst->getArgOperand(0);
    VectorType *VectorTy = cast<VectorType>(InputVal->getType());
    ShapeInfo ArgShape(Inst->getArgOperand(1), Inst->getArgOperand(2));
    ColumnMatrixTy InputMatrix = getMatrix(InputVal, ArgShape, Builder);

    for (unsigned Row = 0; Row < ArgShape.NumRows; ++Row) {
      // Build a single column vector for this row. First initialize it.
      Value *ResultColumn = UndefValue::get(
          VectorType::get(VectorTy->getElementType(), ArgShape.NumColumns));

      // Go through the elements of this row and insert it into the resulting
      // column vector.
      for (auto C : enumerate(InputMatrix.columns())) {
        Value *Elt = Builder.CreateExtractElement(C.value(), Row);
        // We insert at index Column since that is the row index after the
        // transpose.
        ResultColumn =
            Builder.CreateInsertElement(ResultColumn, Elt, C.index());
      }
      Result.addColumn(ResultColumn);
    }

    finalizeLowering(Inst, Result, Builder);
  }

  /// Lower load instructions, if shape information is available.
  bool VisitLoad(Instruction *Inst, Value *Ptr, IRBuilder<> &Builder) {
    auto I = ShapeMap.find(Inst);
    if (I == ShapeMap.end())
      return false;

    LowerLoad(Inst, Ptr, Builder.getInt32(I->second.NumRows), I->second);
    return true;
  }

  bool VisitStore(Instruction *Inst, Value *StoredVal, Value *Ptr,
                  IRBuilder<> &Builder) {
    auto I = ShapeMap.find(StoredVal);
    if (I == ShapeMap.end())
      return false;

    LowerStore(Inst, StoredVal, Ptr, Builder.getInt32(I->second.NumRows), I->second);
    return true;
  }

  /// Lower binary operators, if shape information is available.
  bool VisitBinaryOperator(BinaryOperator *Inst) {
    auto I = ShapeMap.find(Inst);
    if (I == ShapeMap.end())
      return false;

    Value *Lhs = Inst->getOperand(0);
    Value *Rhs = Inst->getOperand(1);

    IRBuilder<> Builder(Inst);
    ShapeInfo &Shape = I->second;

    ColumnMatrixTy LoweredLhs = getMatrix(Lhs, Shape, Builder);
    ColumnMatrixTy LoweredRhs = getMatrix(Rhs, Shape, Builder);

    // Add each column and store the result back into the opmapping
    ColumnMatrixTy Result;
    auto BuildColumnOp = [&Builder, Inst](Value *LHS, Value *RHS) {
      switch (Inst->getOpcode()) {
      case Instruction::Add:
        return Builder.CreateAdd(LHS, RHS);
      case Instruction::Mul:
        return Builder.CreateMul(LHS, RHS);
      case Instruction::Sub:
        return Builder.CreateSub(LHS, RHS);
      case Instruction::FAdd:
        return Builder.CreateFAdd(LHS, RHS);
      case Instruction::FMul:
        return Builder.CreateFMul(LHS, RHS);
      case Instruction::FSub:
        return Builder.CreateFSub(LHS, RHS);
      default:
        llvm_unreachable("Unsupported binary operator for matrix");
      }
    };
    for (unsigned C = 0; C < Shape.NumColumns; ++C)
      Result.addColumn(
          BuildColumnOp(LoweredLhs.getColumn(C), LoweredRhs.getColumn(C)));

    finalizeLowering(Inst, Result, Builder);
    return true;
  }
};
} // namespace

PreservedAnalyses LowerMatrixIntrinsicsPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  LowerMatrixIntrinsics LMT(F, TTI);
  if (LMT.Visit()) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
  return PreservedAnalyses::all();
}

namespace {

class LowerMatrixIntrinsicsLegacyPass : public FunctionPass {
public:
  static char ID;

  LowerMatrixIntrinsicsLegacyPass() : FunctionPass(ID) {
    initializeLowerMatrixIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    LowerMatrixIntrinsics LMT(F, *TTI);
    bool C = LMT.Visit();
    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // namespace

static const char pass_name[] = "Lower the matrix intrinsics";
char LowerMatrixIntrinsicsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LowerMatrixIntrinsicsLegacyPass, DEBUG_TYPE, pass_name,
                      false, false)
INITIALIZE_PASS_END(LowerMatrixIntrinsicsLegacyPass, DEBUG_TYPE, pass_name,
                    false, false)

Pass *llvm::createLowerMatrixIntrinsicsPass() {
  return new LowerMatrixIntrinsicsLegacyPass();
}
