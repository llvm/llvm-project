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
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
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

static cl::opt<bool> EnableShapePropagation(
    "matrix-propagate-shape", cl::init(true), cl::Hidden,
    cl::desc("Enable/disable shape propagation from matrix intrinsics to other "
             "instructions."));

static cl::opt<bool> AllowContractEnabled(
    "matrix-allow-contract", cl::init(false), cl::Hidden,
    cl::desc("Allow the use of FMAs if available and profitable. This may "
             "result in different results, due to less rounding error."));

/// Helper function to either return Scope, if it is a subprogram or the
/// attached subprogram for a local scope.
static DISubprogram *getSubprogram(DIScope *Scope) {
  if (auto *Subprogram = dyn_cast<DISubprogram>(Scope))
    return Subprogram;
  return cast<DILocalScope>(Scope)->getSubprogram();
}

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
  OptimizationRemarkEmitter &ORE;

  /// Contains estimates of the number of operations (loads, stores, compute) required to lower a matrix operation.
  struct OpInfoTy {
    /// Number of stores emitted to generate this matrix.
    unsigned NumStores = 0;
    /// Number of loads emitted to generate this matrix.
    unsigned NumLoads = 0;
    /// Number of compute operations emitted to generate this matrix.
    unsigned NumComputeOps = 0;

    OpInfoTy &operator+=(const OpInfoTy &RHS) {
      NumStores += RHS.NumStores;
      NumLoads += RHS.NumLoads;
      NumComputeOps += RHS.NumComputeOps;
      return *this;
    }
  };

  /// Wrapper class representing a matrix as a set of column vectors.
  /// All column vectors must have the same vector type.
  class ColumnMatrixTy {
    SmallVector<Value *, 16> Columns;

    OpInfoTy OpInfo;

  public:
    ColumnMatrixTy() : Columns() {}
    ColumnMatrixTy(ArrayRef<Value *> Cols)
        : Columns(Cols.begin(), Cols.end()) {}

    Value *getColumn(unsigned i) const { return Columns[i]; }

    void setColumn(unsigned i, Value *V) { Columns[i] = V; }

    Type *getElementType() {
      return cast<VectorType>(Columns[0]->getType())->getElementType();
    }

    unsigned getNumColumns() const { return Columns.size(); }
    unsigned getNumRows() const {
      assert(Columns.size() > 0 && "Cannot call getNumRows without columns");
      return cast<VectorType>(Columns[0]->getType())->getNumElements();
    }

    const SmallVectorImpl<Value *> &getColumnVectors() const { return Columns; }

    SmallVectorImpl<Value *> &getColumnVectors() { return Columns; }

    void addColumn(Value *V) { Columns.push_back(V); }

    VectorType *getColumnTy() {
      return cast<VectorType>(Columns[0]->getType());
    }

    iterator_range<SmallVector<Value *, 8>::iterator> columns() {
      return make_range(Columns.begin(), Columns.end());
    }

    /// Embed the columns of the matrix into a flat vector by concatenating
    /// them.
    Value *embedInVector(IRBuilder<> &Builder) const {
      return Columns.size() == 1 ? Columns[0]
                                 : concatenateVectors(Builder, Columns);
    }

    ColumnMatrixTy &addNumLoads(unsigned N) {
      OpInfo.NumLoads += N;
      return *this;
    }

    void setNumLoads(unsigned N) { OpInfo.NumLoads = N; }

    ColumnMatrixTy &addNumStores(unsigned N) {
      OpInfo.NumStores += N;
      return *this;
    }

    ColumnMatrixTy &addNumComputeOps(unsigned N) {
      OpInfo.NumComputeOps += N;
      return *this;
    }

    unsigned getNumStores() const { return OpInfo.NumStores; }
    unsigned getNumLoads() const { return OpInfo.NumLoads; }
    unsigned getNumComputeOps() const { return OpInfo.NumComputeOps; }

    const OpInfoTy &getOpInfo() const { return OpInfo; }
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
  MapVector<Value *, ColumnMatrixTy> Inst2ColumnMatrix;

public:
  LowerMatrixIntrinsics(Function &F, TargetTransformInfo &TTI,
                        OptimizationRemarkEmitter &ORE)
      : Func(F), DL(F.getParent()->getDataLayout()), TTI(TTI), ORE(ORE) {}

  unsigned getNumOps(Type *VT) {
    assert(isa<VectorType>(VT) && "Expected vector type");
    return getNumOps(VT->getScalarType(),
                     cast<VectorType>(VT)->getNumElements());
  }

  //
  /// Return the estimated number of vector ops required for an operation on
  /// \p VT * N.
  unsigned getNumOps(Type *ST, unsigned N) {
    return std::ceil((ST->getPrimitiveSizeInBits() * N).getFixedSize() /
                     double(TTI.getRegisterBitWidth(true)));
  }

  /// Return the set of column vectors that a matrix value is lowered to.
  ///
  /// If we lowered \p MatrixVal, just return the cache result column matrix.
  /// Otherwie split the flat vector \p MatrixVal containing a matrix with
  /// shape \p SI into column vectors.
  ColumnMatrixTy getMatrix(Value *MatrixVal, const ShapeInfo &SI,
                           IRBuilder<> &Builder) {
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

    RemarkGenerator RemarkGen(Inst2ColumnMatrix, ORE, Func);
    RemarkGen.emitRemarks();

    for (Instruction *Inst : reverse(ToRemove))
      Inst->eraseFromParent();

    return Changed;
  }

  LoadInst *createColumnLoad(Value *ColumnPtr, Type *EltType,
                             IRBuilder<> &Builder) {
    return Builder.CreateAlignedLoad(
        ColumnPtr, Align(DL.getABITypeAlignment(EltType)), "col.load");
  }

  StoreInst *createColumnStore(Value *ColumnValue, Value *ColumnPtr,
                               Type *EltType, IRBuilder<> &Builder) {
    return Builder.CreateAlignedStore(ColumnValue, ColumnPtr,
                                      DL.getABITypeAlign(EltType));
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

  /// Load a matrix with \p Shape starting at \p Ptr and using \p Stride between
  /// columns.
  ColumnMatrixTy loadMatrix(Type *Ty, Value *Ptr, Value *Stride,
                            ShapeInfo Shape, IRBuilder<> &Builder) {
    auto VType = cast<VectorType>(Ty);
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
    return Result.addNumLoads(getNumOps(Result.getColumnTy()) *
                              Result.getNumColumns());
  }

  /// Loads a sub-matrix with shape \p ResultShape from a \p R x \p C matrix,
  /// starting at \p MatrixPtr[I][J].
  ColumnMatrixTy loadMatrix(Value *MatrixPtr, ShapeInfo MatrixShape, unsigned I,
                            unsigned J, ShapeInfo ResultShape, Type *EltTy,
                            IRBuilder<> &Builder) {

    Value *Offset = Builder.CreateAdd(
        Builder.CreateMul(Builder.getInt32(J),
                          Builder.getInt32(MatrixShape.NumRows)),
        Builder.getInt32(I));

    unsigned AS = cast<PointerType>(MatrixPtr->getType())->getAddressSpace();
    Value *EltPtr =
        Builder.CreatePointerCast(MatrixPtr, PointerType::get(EltTy, AS));
    Value *TileStart = Builder.CreateGEP(EltTy, EltPtr, Offset);
    Type *TileTy =
        VectorType::get(EltTy, ResultShape.NumRows * ResultShape.NumColumns);
    Type *TilePtrTy = PointerType::get(TileTy, AS);
    Value *TilePtr =
        Builder.CreatePointerCast(TileStart, TilePtrTy, "col.cast");

    return loadMatrix(TileTy, TilePtr, Builder.getInt32(ResultShape.NumRows),
                      ResultShape, Builder);
  }

  /// Lower a load instruction with shape information.
  void LowerLoad(Instruction *Inst, Value *Ptr, Value *Stride,
                 ShapeInfo Shape) {
    IRBuilder<> Builder(Inst);
    finalizeLowering(Inst,
                     loadMatrix(Inst->getType(), Ptr, Stride, Shape, Builder),
                     Builder);
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

  /// Stores a sub-matrix \p StoreVal into the \p R x \p C matrix starting at \p
  /// MatrixPtr[I][J].
  void storeMatrix(const ColumnMatrixTy &StoreVal, Value *MatrixPtr,
                   ShapeInfo MatrixShape, unsigned I, unsigned J, Type *EltTy,
                   IRBuilder<> &Builder) {
    Value *Offset = Builder.CreateAdd(
        Builder.CreateMul(Builder.getInt32(J),
                          Builder.getInt32(MatrixShape.NumRows)),
        Builder.getInt32(I));

    unsigned AS = cast<PointerType>(MatrixPtr->getType())->getAddressSpace();
    Value *EltPtr =
        Builder.CreatePointerCast(MatrixPtr, PointerType::get(EltTy, AS));
    Value *TileStart = Builder.CreateGEP(EltTy, EltPtr, Offset);
    Type *TileTy = VectorType::get(EltTy, StoreVal.getNumRows() *
                                              StoreVal.getNumColumns());
    Type *TilePtrTy = PointerType::get(TileTy, AS);
    Value *TilePtr =
        Builder.CreatePointerCast(TileStart, TilePtrTy, "col.cast");

    storeMatrix(TileTy, StoreVal, TilePtr,
                Builder.getInt32(StoreVal.getNumRows()), Builder);
  }

  /// Store matrix \p StoreVal starting at \p Ptr and using \p Stride between
  /// columns.
  ColumnMatrixTy storeMatrix(Type *Ty, ColumnMatrixTy StoreVal, Value *Ptr,
                             Value *Stride, IRBuilder<> &Builder) {
    auto VType = cast<VectorType>(Ty);
    Value *EltPtr = createElementPtr(Ptr, VType->getElementType(), Builder);
    for (auto C : enumerate(StoreVal.columns())) {
      Value *GEP = computeColumnAddr(EltPtr, Builder.getInt32(C.index()),
                                     Stride, StoreVal.getNumRows(),
                                     VType->getElementType(), Builder);
      createColumnStore(C.value(), GEP, VType->getElementType(), Builder);
    }
    return ColumnMatrixTy().addNumStores(getNumOps(StoreVal.getColumnTy()) *
                                         StoreVal.getNumColumns());
  }

  /// Lower a store instruction with shape information.
  void LowerStore(Instruction *Inst, Value *Matrix, Value *Ptr, Value *Stride,
                  ShapeInfo Shape) {
    IRBuilder<> Builder(Inst);
    auto StoreVal = getMatrix(Matrix, Shape, Builder);
    finalizeLowering(
        Inst, storeMatrix(Matrix->getType(), StoreVal, Ptr, Stride, Builder),
        Builder);
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
                       unsigned NumElts, IRBuilder<> &Builder) {
    Value *Col = LM.getColumn(J);
    Value *Undef = UndefValue::get(Col->getType());
    Constant *Mask = createSequentialMask(Builder, I, NumElts, 0);
    return Builder.CreateShuffleVector(Col, Undef, Mask, "block");
  }

  // Set elements I..I+NumElts-1 to Block
  Value *insertVector(Value *Col, unsigned I, Value *Block,
                      IRBuilder<> &Builder) {

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
                      IRBuilder<> &Builder, bool AllowContraction,
                      unsigned &NumComputeOps) {
    NumComputeOps += getNumOps(A->getType());
    if (!Sum)
      return UseFPOp ? Builder.CreateFMul(A, B) : Builder.CreateMul(A, B);

    if (UseFPOp) {
      if (AllowContraction) {
        // Use fmuladd for floating point operations and let the backend decide
        // if that's profitable.
        Function *FMulAdd = Intrinsic::getDeclaration(
            Func.getParent(), Intrinsic::fmuladd, A->getType());
        return Builder.CreateCall(FMulAdd, {A, B, Sum});
      }
      NumComputeOps += getNumOps(A->getType());
      Value *Mul = Builder.CreateFMul(A, B);
      return Builder.CreateFAdd(Sum, Mul);
    }

    NumComputeOps += getNumOps(A->getType());
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

  /// Compute Res += A * B for tile-sized matrices with left-associating
  /// addition.
  void emitChainedMatrixMultiply(ColumnMatrixTy &Result,
                                 const ColumnMatrixTy &A,
                                 const ColumnMatrixTy &B, bool AllowContraction,
                                 IRBuilder<> &Builder, bool isTiled) {
    const unsigned VF = std::max<unsigned>(
        TTI.getRegisterBitWidth(true) /
            Result.getElementType()->getPrimitiveSizeInBits().getFixedSize(),
        1U);
    unsigned R = Result.getNumRows();
    unsigned C = Result.getNumColumns();
    unsigned M = A.getNumColumns();

    for (unsigned J = 0; J < C; ++J) {
      unsigned BlockSize = VF;

      // If Result is zero, we don't need to accumulate in the K==0 iteration.
      bool isSumZero = isa<ConstantAggregateZero>(Result.getColumn(J));

      unsigned NumOps = 0;
      for (unsigned I = 0; I < R; I += BlockSize) {
        // Gradually lower the vectorization factor to cover the remainder.
        while (I + BlockSize > R)
          BlockSize /= 2;

        Value *Sum =
            isTiled ? extractVector(Result, I, J, BlockSize, Builder) : nullptr;
        for (unsigned K = 0; K < M; ++K) {
          Value *L = extractVector(A, I, K, BlockSize, Builder);
          Value *RH = Builder.CreateExtractElement(B.getColumn(J), K);
          Value *Splat = Builder.CreateVectorSplat(BlockSize, RH, "splat");
          Sum = createMulAdd(isSumZero && K == 0 ? nullptr : Sum, L, Splat,
                             Result.getElementType()->isFloatingPointTy(),
                             Builder, AllowContraction, NumOps);
        }
        Result.setColumn(J, insertVector(Result.getColumn(J), I, Sum, Builder));
      }

      Result.addNumComputeOps(NumOps);
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
    const unsigned C = RShape.NumColumns;
    assert(LShape.NumColumns == RShape.NumRows);

    // Initialize the output
    ColumnMatrixTy Result;
    for (unsigned J = 0; J < C; ++J)
      Result.addColumn(UndefValue::get(VectorType::get(EltType, R)));

    bool AllowContract = AllowContractEnabled || (isa<FPMathOperator>(MatMul) &&
                                                  MatMul->hasAllowContract());
    emitChainedMatrixMultiply(Result, Lhs, Rhs, AllowContract, Builder, false);
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

    // TODO: Improve estimate of operations needed for transposes. Currently we
    // just count the insertelement/extractelement instructions, but do not
    // account for later simplifications/combines.
    finalizeLowering(
        Inst,
        Result.addNumComputeOps(2 * ArgShape.NumRows * ArgShape.NumColumns),
        Builder);
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

    finalizeLowering(Inst,
                     Result.addNumComputeOps(getNumOps(Result.getColumnTy()) *
                                             Result.getNumColumns()),
                     Builder);
    return true;
  }

  /// Helper to linearize a matrix expression tree into a string. Currently
  /// matrix expressions are linarized by starting at an expression leaf and
  /// linearizing bottom up.
  struct ExprLinearizer {
    unsigned LengthToBreak = 100;
    std::string Str;
    raw_string_ostream Stream;
    unsigned LineLength = 0;
    const DataLayout &DL;

    /// Mapping from instructions to column matrixes. It is used to identify
    /// matrix instructions.
    const MapVector<Value *, ColumnMatrixTy> &Inst2ColumnMatrix;

    /// Mapping from values to the leaves of all expressions that the value is
    /// part of.
    const DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared;

    /// Set of matrix expressions in the scope of a given DISubprogram.
    const SmallSetVector<Value *, 32> &ExprsInSubprogram;

    /// Leaf node of the expression to linearize.
    Value *Leaf;

    /// Used to keep track of sub-expressions that get reused while linearizing
    /// the expression. Re-used sub-expressions are marked as (reused).
    SmallPtrSet<Value *, 8> ReusedExprs;

    ExprLinearizer(const DataLayout &DL,
                   const MapVector<Value *, ColumnMatrixTy> &Inst2ColumnMatrix,
                   const DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared,
                   const SmallSetVector<Value *, 32> &ExprsInSubprogram,
                   Value *Leaf)
        : Str(), Stream(Str), DL(DL), Inst2ColumnMatrix(Inst2ColumnMatrix),
          Shared(Shared), ExprsInSubprogram(ExprsInSubprogram), Leaf(Leaf) {}

    void indent(unsigned N) {
      LineLength += N;
      for (unsigned i = 0; i < N; i++)
        Stream << " ";
    }

    void lineBreak() {
      Stream << "\n";
      LineLength = 0;
    }

    void maybeIndent(unsigned Indent) {
      if (LineLength >= LengthToBreak)
        lineBreak();

      if (LineLength == 0)
        indent(Indent);
    }

    void write(StringRef S) {
      LineLength += S.size();
      Stream << S;
    }

    Value *getUnderlyingObjectThroughLoads(Value *V) {
      if (Value *Ptr = getPointerOperand(V))
        return getUnderlyingObjectThroughLoads(Ptr);
      else if (V->getType()->isPointerTy())
        return GetUnderlyingObject(V, DL);
      return V;
    }

    /// Returns true if \p V is a matrix value in the given subprogram.
    bool isMatrix(Value *V) const { return ExprsInSubprogram.count(V); }

    /// If \p V is a matrix value, print its shape as as NumRows x NumColumns to
    /// \p SS.
    void prettyPrintMatrixType(Value *V, raw_string_ostream &SS) {
      auto M = Inst2ColumnMatrix.find(V);
      if (M == Inst2ColumnMatrix.end())
        SS << "unknown";
      else {
        SS << M->second.getNumRows();
        SS << "x";
        SS << M->second.getNumColumns();
      }
    }

    /// Write the called function name. Handles calls to llvm.matrix.*
    /// specially: we write the name, followed by the dimensions of the input
    /// matrixes, followed by the scalar type name.
    void writeFnName(CallInst *CI) {
      if (!CI->getCalledFunction())
        write("<no called fn>");
      else {
        StringRef Name = CI->getCalledFunction()->getName();
        if (!Name.startswith("llvm.matrix")) {
          write(Name);
          return;
        }
        IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI);
        write(StringRef(Intrinsic::getName(II->getIntrinsicID(), {}))
                  .drop_front(StringRef("llvm.matrix.").size()));
        write(".");
        std::string Tmp = "";
        raw_string_ostream SS(Tmp);

        switch (II->getIntrinsicID()) {
        case Intrinsic::matrix_multiply:
          prettyPrintMatrixType(II->getOperand(0), SS);
          SS << ".";
          prettyPrintMatrixType(II->getOperand(1), SS);
          SS << "." << *II->getType()->getScalarType();
          break;
        case Intrinsic::matrix_transpose:
          prettyPrintMatrixType(II->getOperand(0), SS);
          SS << "." << *II->getType()->getScalarType();
          break;
        case Intrinsic::matrix_columnwise_load:
          prettyPrintMatrixType(II, SS);
          SS << "." << *II->getType()->getScalarType();
          break;
        case Intrinsic::matrix_columnwise_store:
          prettyPrintMatrixType(II->getOperand(0), SS);
          SS << "." << *II->getOperand(0)->getType()->getScalarType();
          break;
        default:
          llvm_unreachable("Unhandled case");
        }
        SS.flush();
        write(Tmp);
      }
    }

    unsigned getNumShapeArgs(CallInst *CI) const {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::matrix_multiply:
          return 3;
        case Intrinsic::matrix_transpose:
        case Intrinsic::matrix_columnwise_load:
        case Intrinsic::matrix_columnwise_store:
          return 2;
        default:
          return 0;
        }
      }
      return 0;
    }

    /// Special printing for values: for pointers, we print if they refer to an
    /// (function) external address or a stack address, for other values we
    /// either print the constant or "scalar"/"matrix" for other values.
    void write(Value *V) {
      V = getUnderlyingObjectThroughLoads(V);
      if (V->getType()->isPointerTy()) {
        if (isa<AllocaInst>(V)) {
          Stream << "stack addr";
          LineLength += StringRef("stack addr").size();
        } else {
          Stream << "addr";
          LineLength += StringRef("addr").size();
        }
        if (!V->getName().empty()) {
          Stream << " %" << V->getName() << "";
          LineLength += V->getName().size() + 2;
        }
        return;
      }

      std::string Tmp;
      raw_string_ostream TmpStream(Tmp);

      if (auto *CI = dyn_cast<ConstantInt>(V))
        TmpStream << CI->getValue();
      else if (isa<Constant>(V))
        TmpStream << "constant";
      else {
        if (isMatrix(V))
          TmpStream << "matrix";
        else
          TmpStream << "scalar";
      }
      TmpStream.flush();
      Tmp = std::string(StringRef(Tmp).trim());
      LineLength += Tmp.size();
      Stream << Tmp;
    }

    /// Linearize expression \p Expr starting at an indentation of \p Indent.
    /// Expressions that are re-used multiple times are prefixed with (reused)
    /// at the re-used root instruction.
    void linearizeExpr(Value *Expr, unsigned Indent, bool ParentReused,
                       bool ParentShared) {
      auto *I = cast<Instruction>(Expr);
      maybeIndent(Indent);
      SmallVector<Value *, 8> Ops;

      // Is Expr shared with other expression leaves?
      bool ExprShared = false;

      // Deal with shared subtrees. Mark them as shared, if required.
      if (!ParentShared) {
        auto SI = Shared.find(Expr);
        assert(SI != Shared.end() && SI->second.find(Leaf) != SI->second.end());

        for (Value *S : SI->second) {
          if (S == Leaf)
            continue;
          DebugLoc DL = cast<Instruction>(S)->getDebugLoc();
          write("shared with remark at line " + std::to_string(DL.getLine()) +
                " column " + std::to_string(DL.getCol()) + " (");
        }
        ExprShared = SI->second.size() > 1;
      }

      bool Reused = !ReusedExprs.insert(Expr).second;
      if (Reused && !ParentReused)
        write("(reused) ");

      if (auto *CI = dyn_cast<CallInst>(I)) {
        writeFnName(CI);

        Ops.append(CallSite(CI).arg_begin(),
                   CallSite(CI).arg_end() - getNumShapeArgs(CI));
      } else if (isa<BitCastInst>(Expr)) {
        // Special case bitcasts, which are used to materialize matrixes from
        // non-matrix ops.
        write("matrix");
        return;
      } else {
        Ops.append(I->value_op_begin(), I->value_op_end());
        write(std::string(I->getOpcodeName()));
      }

      write(std::string("("));

      unsigned NumOpsToBreak = 1;
      if (match(Expr, m_Intrinsic<Intrinsic::matrix_columnwise_load>()))
        NumOpsToBreak = 2;

      for (Value *Op : Ops) {
        if (Ops.size() > NumOpsToBreak)
          lineBreak();

        maybeIndent(Indent + 1);
        if (isMatrix(Op))
          linearizeExpr(Op, Indent + 1, Reused, ExprShared);
        else
          write(Op);
        if (Op != Ops.back())
          write(", ");
      }

      write(")");
    }

    const std::string &getResult() {
      Stream.flush();
      return Str;
    }
  };

  /// Generate remarks for matrix operations in a function. To generate remarks
  /// for matrix expressions, the following approach is used:
  /// 1. Use the inlined-at debug information to group matrix operations to the
  ///    DISubprograms they are contained in.
  /// 2. Collect leaves of matrix expressions (done in
  ///    RemarkGenerator::getExpressionLeaves) for each subprogram - expression
  //     mapping.  Leaves are lowered matrix instructions without other matrix
  //     users (like stores) in the current subprogram.
  /// 3. For each leaf, create a remark containing a linearizied version of the
  ///    matrix expression. The expression is linearized by a recursive
  ///    bottom-up traversal of the matrix operands, starting at a leaf. Note
  ///    that multiple leaves can share sub-expressions. Shared subexpressions
  ///    are explicitly marked as shared().
  struct RemarkGenerator {
    const MapVector<Value *, ColumnMatrixTy> &Inst2ColumnMatrix;
    OptimizationRemarkEmitter &ORE;
    Function &Func;
    const DataLayout &DL;

    RemarkGenerator(const MapVector<Value *, ColumnMatrixTy> &Inst2ColumnMatrix,
                    OptimizationRemarkEmitter &ORE, Function &Func)
        : Inst2ColumnMatrix(Inst2ColumnMatrix), ORE(ORE), Func(Func),
          DL(Func.getParent()->getDataLayout()) {}

    /// Return all leaves of the expressions in \p ExprsInSubprogram. Those are
    /// instructions in Inst2ColumnMatrix returning void or without any users in
    /// \p ExprsInSubprogram. Currently that should only include stores.
    SmallVector<Value *, 4>
    getExpressionLeaves(const SmallSetVector<Value *, 32> &ExprsInSubprogram) {
      SmallVector<Value *, 4> Leaves;
      for (auto *Expr : ExprsInSubprogram)
        if (Expr->getType()->isVoidTy() ||
            !any_of(Expr->users(), [&ExprsInSubprogram](User *U) {
              return ExprsInSubprogram.count(U);
            }))
          Leaves.push_back(Expr);
      return Leaves;
    }

    /// Recursively traverse expression \p V starting at \p Leaf and add \p Leaf
    /// to all visited expressions in \p Shared. Limit the matrix operations to
    /// the ones in \p ExprsInSubprogram.
    void collectSharedInfo(Value *Leaf, Value *V,
                           const SmallSetVector<Value *, 32> &ExprsInSubprogram,
                           DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared) {

      if (!ExprsInSubprogram.count(V))
        return;

      auto I = Shared.insert({V, {}});
      I.first->second.insert(Leaf);

      for (Value *Op : cast<Instruction>(V)->operand_values())
        collectSharedInfo(Leaf, Op, ExprsInSubprogram, Shared);
      return;
    }

    /// Calculate the number of exclusive and shared op counts for expression
    /// starting at \p V. Expressions used multiple times are counted once.
    /// Limit the matrix operations to the ones in \p ExprsInSubprogram.
    std::pair<OpInfoTy, OpInfoTy>
    sumOpInfos(Value *Root, SmallPtrSetImpl<Value *> &ReusedExprs,
               const SmallSetVector<Value *, 32> &ExprsInSubprogram,
               DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared) const {
      if (!ExprsInSubprogram.count(Root))
        return {};

      // Already counted this expression. Stop.
      if (!ReusedExprs.insert(Root).second)
        return {};

      OpInfoTy SharedCount;
      OpInfoTy Count;

      auto I = Shared.find(Root);
      auto CM = Inst2ColumnMatrix.find(Root);
      if (I->second.size() == 1)
        Count = CM->second.getOpInfo();
      else
        SharedCount = CM->second.getOpInfo();

      for (Value *Op : cast<Instruction>(Root)->operand_values()) {
        auto C = sumOpInfos(Op, ReusedExprs, ExprsInSubprogram, Shared);
        Count += C.first;
        SharedCount += C.second;
      }
      return {Count, SharedCount};
    }

    void emitRemarks() {
      if (!ORE.allowExtraAnalysis(DEBUG_TYPE))
        return;

      // Map matrix operations to their containting subprograms, by traversing
      // the inlinedAt chain. If the function does not have a DISubprogram, we
      // only map them to the containing function.
      MapVector<DISubprogram *, SmallVector<Value *, 8>> Subprog2Exprs;
      for (auto &KV : Inst2ColumnMatrix) {
        if (Func.getSubprogram()) {
          auto *I = cast<Instruction>(KV.first);
          DILocation *Context = I->getDebugLoc();
          while (Context) {
            auto I =
                Subprog2Exprs.insert({getSubprogram(Context->getScope()), {}});
            I.first->second.push_back(KV.first);
            Context = DebugLoc(Context).getInlinedAt();
          }
        } else {
          auto I = Subprog2Exprs.insert({nullptr, {}});
          I.first->second.push_back(KV.first);
        }
      }
      for (auto &KV : Subprog2Exprs) {
        SmallSetVector<Value *, 32> ExprsInSubprogram(KV.second.begin(),
                                                      KV.second.end());
        auto Leaves = getExpressionLeaves(ExprsInSubprogram);

        DenseMap<Value *, SmallPtrSet<Value *, 2>> Shared;
        for (Value *Leaf : Leaves)
          collectSharedInfo(Leaf, Leaf, ExprsInSubprogram, Shared);

        // Generate remarks for each leaf.
        for (auto *L : Leaves) {

          DebugLoc Loc = cast<Instruction>(L)->getDebugLoc();
          DILocation *Context = cast<Instruction>(L)->getDebugLoc();
          while (Context) {
            if (getSubprogram(Context->getScope()) == KV.first) {
              Loc = Context;
              break;
            }
            Context = DebugLoc(Context).getInlinedAt();
          }

          SmallPtrSet<Value *, 8> ReusedExprs;
          OpInfoTy Counts, SharedCounts;
          std::tie(Counts, SharedCounts) =
              sumOpInfos(L, ReusedExprs, ExprsInSubprogram, Shared);

          OptimizationRemark Rem(DEBUG_TYPE, "matrix-lowered", Loc,
                                 cast<Instruction>(L)->getParent());

          Rem << "Lowered with ";
          Rem << ore::NV("NumStores", Counts.NumStores) << " stores, "
              << ore::NV("NumLoads", Counts.NumLoads) << " loads, "
              << ore::NV("NumComputeOps", Counts.NumComputeOps)
              << " compute ops";

          if (SharedCounts.NumStores > 0 || SharedCounts.NumLoads > 0 ||
              SharedCounts.NumComputeOps > 0) {
            Rem << ",\nadditionally "
                << ore::NV("NumStores", SharedCounts.NumStores) << " stores, "
                << ore::NV("NumLoads", SharedCounts.NumLoads) << " loads, "
                << ore::NV("NumFPOps", SharedCounts.NumComputeOps)
                << " compute ops"
                << " are shared with other expressions";
          }

          Rem << ("\n" + linearize(L, Shared, ExprsInSubprogram, DL));
          ORE.emit(Rem);
        }
      }
    }

    std::string
    linearize(Value *L,
              const DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared,
              const SmallSetVector<Value *, 32> &ExprsInSubprogram,
              const DataLayout &DL) {
      ExprLinearizer Lin(DL, Inst2ColumnMatrix, Shared, ExprsInSubprogram, L);
      Lin.linearizeExpr(L, 0, false, false);
      return Lin.getResult();
    }
  };
};
} // namespace

PreservedAnalyses LowerMatrixIntrinsicsPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  LowerMatrixIntrinsics LMT(F, TTI, ORE);
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
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    LowerMatrixIntrinsics LMT(F, TTI, ORE);
    bool C = LMT.Visit();
    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // namespace

static const char pass_name[] = "Lower the matrix intrinsics";
char LowerMatrixIntrinsicsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LowerMatrixIntrinsicsLegacyPass, DEBUG_TYPE, pass_name,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(LowerMatrixIntrinsicsLegacyPass, DEBUG_TYPE, pass_name,
                    false, false)

Pass *llvm::createLowerMatrixIntrinsicsPass() {
  return new LowerMatrixIntrinsicsLegacyPass();
}
