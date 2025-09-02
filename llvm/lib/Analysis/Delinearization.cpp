//===---- Delinearization.cpp - MultiDimensional Index Delinearization ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an analysis pass that tries to delinearize all GEP
// instructions in all loops using the SCEV analysis functionality. This pass is
// only used for testing purposes: if your pass needs delinearization, please
// use the on-demand SCEVAddRecExpr::delinearize() function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Delinearization.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionDivision.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DL_NAME "delinearize"
#define DEBUG_TYPE DL_NAME

static cl::opt<bool> UseFixedSizeArrayHeuristic(
    "delinearize-use-fixed-size-array-heuristic", cl::init(false), cl::Hidden,
    cl::desc("When printing analysis, use the heuristic for fixed-size arrays "
             "if the default delinearizetion fails."));

// Return true when S contains at least an undef value.
static inline bool containsUndefs(const SCEV *S) {
  return SCEVExprContains(S, [](const SCEV *S) {
    if (const auto *SU = dyn_cast<SCEVUnknown>(S))
      return isa<UndefValue>(SU->getValue());
    return false;
  });
}

namespace {

// Collect all steps of SCEV expressions.
struct SCEVCollectStrides {
  ScalarEvolution &SE;
  SmallVectorImpl<const SCEV *> &Strides;

  SCEVCollectStrides(ScalarEvolution &SE, SmallVectorImpl<const SCEV *> &S)
      : SE(SE), Strides(S) {}

  bool follow(const SCEV *S) {
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
      Strides.push_back(AR->getStepRecurrence(SE));
    return true;
  }

  bool isDone() const { return false; }
};

// Collect all SCEVUnknown and SCEVMulExpr expressions.
struct SCEVCollectTerms {
  SmallVectorImpl<const SCEV *> &Terms;

  SCEVCollectTerms(SmallVectorImpl<const SCEV *> &T) : Terms(T) {}

  bool follow(const SCEV *S) {
    if (isa<SCEVUnknown>(S) || isa<SCEVMulExpr>(S) ||
        isa<SCEVSignExtendExpr>(S)) {
      if (!containsUndefs(S))
        Terms.push_back(S);

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

// Check if a SCEV contains an AddRecExpr.
struct SCEVHasAddRec {
  bool &ContainsAddRec;

  SCEVHasAddRec(bool &ContainsAddRec) : ContainsAddRec(ContainsAddRec) {
    ContainsAddRec = false;
  }

  bool follow(const SCEV *S) {
    if (isa<SCEVAddRecExpr>(S)) {
      ContainsAddRec = true;

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

// Find factors that are multiplied with an expression that (possibly as a
// subexpression) contains an AddRecExpr. In the expression:
//
//  8 * (100 +  %p * %q * (%a + {0, +, 1}_loop))
//
// "%p * %q" are factors multiplied by the expression "(%a + {0, +, 1}_loop)"
// that contains the AddRec {0, +, 1}_loop. %p * %q are likely to be array size
// parameters as they form a product with an induction variable.
//
// This collector expects all array size parameters to be in the same MulExpr.
// It might be necessary to later add support for collecting parameters that are
// spread over different nested MulExpr.
struct SCEVCollectAddRecMultiplies {
  SmallVectorImpl<const SCEV *> &Terms;
  ScalarEvolution &SE;

  SCEVCollectAddRecMultiplies(SmallVectorImpl<const SCEV *> &T,
                              ScalarEvolution &SE)
      : Terms(T), SE(SE) {}

  bool follow(const SCEV *S) {
    if (auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
      bool HasAddRec = false;
      SmallVector<const SCEV *, 0> Operands;
      for (const SCEV *Op : Mul->operands()) {
        const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(Op);
        if (Unknown && !isa<CallInst>(Unknown->getValue())) {
          Operands.push_back(Op);
        } else if (Unknown) {
          HasAddRec = true;
        } else {
          bool ContainsAddRec = false;
          SCEVHasAddRec ContiansAddRec(ContainsAddRec);
          visitAll(Op, ContiansAddRec);
          HasAddRec |= ContainsAddRec;
        }
      }
      if (Operands.size() == 0)
        return true;

      if (!HasAddRec)
        return false;

      Terms.push_back(SE.getMulExpr(Operands));
      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

} // end anonymous namespace

/// Find parametric terms in this SCEVAddRecExpr. We first for parameters in
/// two places:
///   1) The strides of AddRec expressions.
///   2) Unknowns that are multiplied with AddRec expressions.
void llvm::collectParametricTerms(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Terms) {
  SmallVector<const SCEV *, 4> Strides;
  SCEVCollectStrides StrideCollector(SE, Strides);
  visitAll(Expr, StrideCollector);

  LLVM_DEBUG({
    dbgs() << "Strides:\n";
    for (const SCEV *S : Strides)
      dbgs() << "  " << *S << "\n";
  });

  for (const SCEV *S : Strides) {
    SCEVCollectTerms TermCollector(Terms);
    visitAll(S, TermCollector);
  }

  LLVM_DEBUG({
    dbgs() << "Terms:\n";
    for (const SCEV *T : Terms)
      dbgs() << "  " << *T << "\n";
  });

  SCEVCollectAddRecMultiplies MulCollector(Terms, SE);
  visitAll(Expr, MulCollector);
}

static bool findArrayDimensionsRec(ScalarEvolution &SE,
                                   SmallVectorImpl<const SCEV *> &Terms,
                                   SmallVectorImpl<const SCEV *> &Sizes) {
  int Last = Terms.size() - 1;
  const SCEV *Step = Terms[Last];

  // End of recursion.
  if (Last == 0) {
    if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Step)) {
      SmallVector<const SCEV *, 2> Qs;
      for (const SCEV *Op : M->operands())
        if (!isa<SCEVConstant>(Op))
          Qs.push_back(Op);

      Step = SE.getMulExpr(Qs);
    }

    Sizes.push_back(Step);
    return true;
  }

  for (const SCEV *&Term : Terms) {
    // Normalize the terms before the next call to findArrayDimensionsRec.
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, Step, &Q, &R);

    // Bail out when GCD does not evenly divide one of the terms.
    if (!R->isZero())
      return false;

    Term = Q;
  }

  // Remove all SCEVConstants.
  erase_if(Terms, [](const SCEV *E) { return isa<SCEVConstant>(E); });

  if (Terms.size() > 0)
    if (!findArrayDimensionsRec(SE, Terms, Sizes))
      return false;

  Sizes.push_back(Step);
  return true;
}

// Returns true when one of the SCEVs of Terms contains a SCEVUnknown parameter.
static inline bool containsParameters(SmallVectorImpl<const SCEV *> &Terms) {
  for (const SCEV *T : Terms)
    if (SCEVExprContains(T, [](const SCEV *S) { return isa<SCEVUnknown>(S); }))
      return true;

  return false;
}

// Return the number of product terms in S.
static inline int numberOfTerms(const SCEV *S) {
  if (const SCEVMulExpr *Expr = dyn_cast<SCEVMulExpr>(S))
    return Expr->getNumOperands();
  return 1;
}

static const SCEV *removeConstantFactors(ScalarEvolution &SE, const SCEV *T) {
  if (isa<SCEVConstant>(T))
    return nullptr;

  if (isa<SCEVUnknown>(T))
    return T;

  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(T)) {
    SmallVector<const SCEV *, 2> Factors;
    for (const SCEV *Op : M->operands())
      if (!isa<SCEVConstant>(Op))
        Factors.push_back(Op);

    return SE.getMulExpr(Factors);
  }

  return T;
}

void llvm::findArrayDimensions(ScalarEvolution &SE,
                               SmallVectorImpl<const SCEV *> &Terms,
                               SmallVectorImpl<const SCEV *> &Sizes,
                               const SCEV *ElementSize) {
  if (Terms.size() < 1 || !ElementSize)
    return;

  // Early return when Terms do not contain parameters: we do not delinearize
  // non parametric SCEVs.
  if (!containsParameters(Terms))
    return;

  LLVM_DEBUG({
    dbgs() << "Terms:\n";
    for (const SCEV *T : Terms)
      dbgs() << "  " << *T << "\n";
  });

  // Remove duplicates.
  array_pod_sort(Terms.begin(), Terms.end());
  Terms.erase(llvm::unique(Terms), Terms.end());

  // Put larger terms first.
  llvm::sort(Terms, [](const SCEV *LHS, const SCEV *RHS) {
    return numberOfTerms(LHS) > numberOfTerms(RHS);
  });

  // Try to divide all terms by the element size. If term is not divisible by
  // element size, proceed with the original term.
  for (const SCEV *&Term : Terms) {
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, ElementSize, &Q, &R);
    if (!Q->isZero())
      Term = Q;
  }

  SmallVector<const SCEV *, 4> NewTerms;

  // Remove constant factors.
  for (const SCEV *T : Terms)
    if (const SCEV *NewT = removeConstantFactors(SE, T))
      NewTerms.push_back(NewT);

  LLVM_DEBUG({
    dbgs() << "Terms after sorting:\n";
    for (const SCEV *T : NewTerms)
      dbgs() << "  " << *T << "\n";
  });

  if (NewTerms.empty() || !findArrayDimensionsRec(SE, NewTerms, Sizes)) {
    Sizes.clear();
    return;
  }

  // The last element to be pushed into Sizes is the size of an element.
  Sizes.push_back(ElementSize);

  LLVM_DEBUG({
    dbgs() << "Sizes:\n";
    for (const SCEV *S : Sizes)
      dbgs() << "  " << *S << "\n";
  });
}

void llvm::computeAccessFunctions(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Subscripts,
                                  SmallVectorImpl<const SCEV *> &Sizes,
                                  const SCEV *ElementSize) {
  // Early exit in case this SCEV is not an affine multivariate function.
  if (Sizes.empty())
    return;

  if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr))
    if (!AR->isAffine())
      return;

  if (ElementSize->isZero())
    return;

  // Clear output vector.
  Subscripts.clear();

  LLVM_DEBUG(dbgs() << "\ncomputeAccessFunctions\n"
                    << "Linearized Memory Access Function: " << *Expr << "\n");

  const SCEV *Res = Expr;
  int Last = Sizes.size() - 1;

  for (int i = Last; i >= 0; i--) {
    const SCEV *Size = Sizes[i];
    if (Size->isZero())
      continue;

    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Res, Size, &Q, &R);
    LLVM_DEBUG({
      dbgs() << "Computing 'MemAccFn / Sizes[" << i << "]':\n";
      dbgs() << "  MemAccFn: " << *Res << "\n";
      dbgs() << "  Sizes[" << i << "]: " << *Size << "\n";
      dbgs() << "  Quotient (Leftover): " << *Q << "\n";
      dbgs() << "  Remainder (Subscript Access Function): " << *R << "\n";
    });
    Res = Q;

    // Do not record the last subscript corresponding to the size of elements in
    // the array.
    if (i == Last) {

      // Bail out if the byte offset is non-zero.
      if (!R->isZero()) {
        Subscripts.clear();
        Sizes.clear();
        return;
      }

      continue;
    }

    // Record the access function for the current subscript.
    LLVM_DEBUG(dbgs() << "Subscripts push_back Remainder: " << *R << "\n");
    Subscripts.push_back(R);
  }

  // Also push in last position the quotient "Res = Q" of the last division: it
  // will be the access function of the outermost array dimension.
  if (!Res->isZero()) {
    // This is only needed when the outermost array size is not known.  Res = 0
    // when the outermost array dimension is known, as for example when reading
    // array sizes from a local or global declaration.
    Subscripts.push_back(Res);
    LLVM_DEBUG(dbgs() << "Subscripts push_back Res: " << *Res << "\n");
  }

  std::reverse(Subscripts.begin(), Subscripts.end());

  LLVM_DEBUG({
    dbgs() << "Subscripts:\n";
    for (const SCEV *S : Subscripts)
      dbgs() << "  " << *S << "\n";
    dbgs() << "\n";
  });
}

/// Backward compatibility wrapper for the old 4-parameter version.
void llvm::computeAccessFunctions(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Subscripts,
                                  SmallVectorImpl<const SCEV *> &Sizes) {
  // Use the element size from the last element in Sizes array (legacy behavior)
  if (Sizes.empty()) {
    Subscripts.clear();
    return;
  }
  const SCEV *ElementSize = Sizes.back();
  computeAccessFunctions(SE, Expr, Subscripts, Sizes, ElementSize);
}

/// Extract array dimensions from alloca or global variable declarations.
/// Returns true if array dimensions were successfully extracted.
static bool
extractArrayInfoFromAllocaOrGlobal(ScalarEvolution &SE, Value *BasePtr,
                                   SmallVectorImpl<const SCEV *> &Sizes,
                                   const SCEV *ElementSize) {
  // Clear output vector.
  Sizes.clear();

  LLVM_DEBUG(
      dbgs() << "extractArrayInfoFromAllocaOrGlobal called with BasePtr: "
             << *BasePtr << "\n");

  // Distinguish between simple array accesses and complex pointer arithmetic.
  // Only apply array_info extraction to direct array accesses to avoid
  // incorrect delinearization of complex pointer arithmetic patterns.
  if (auto *GEP = dyn_cast<GetElementPtrInst>(BasePtr)) {
    // Check if this is a simple array access pattern: GEP [N x T]* @array, 0,
    // idx This represents direct indexing like array[i], which should use array
    // dimensions.
    if (GEP->getNumIndices() == 2) {
      auto *FirstIdx = dyn_cast<ConstantInt>(GEP->getOperand(1));
      if (FirstIdx && FirstIdx->isZero()) {
        // Simple array access: extract dimensions from the underlying array
        // type
        Value *Source = GEP->getPointerOperand()->stripPointerCasts();
        return extractArrayInfoFromAllocaOrGlobal(SE, Source, Sizes,
                                                  ElementSize);
      }
    }
    // Complex GEPs like (&array[offset])[index] represent pointer arithmetic,
    // not simple array indexing. These should be handled by parametric
    // delinearization to preserve the linearized byte-offset semantics rather
    // than treating them as multidimensional array accesses.
    return false;
  }

  // Check if BasePtr is from an alloca instruction.
  Type *ElementType = nullptr;
  if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
    ElementType = AI->getAllocatedType();
    LLVM_DEBUG(dbgs() << "Found alloca with type: " << *ElementType << "\n");
  } else if (auto *GV = dyn_cast<GlobalVariable>(BasePtr)) {
    ElementType = GV->getValueType();
    LLVM_DEBUG(dbgs() << "Found global variable with type: " << *ElementType
                      << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "No alloca or global found for base pointer\n");
    return false;
  }

  // Extract dimensions from nested array types.
  Type *I64Ty = Type::getInt64Ty(SE.getContext());

  while (auto *ArrayTy = dyn_cast<ArrayType>(ElementType)) {
    uint64_t Size = ArrayTy->getNumElements();
    const SCEV *SizeSCEV = SE.getConstant(I64Ty, Size);
    Sizes.push_back(SizeSCEV);
    ElementType = ArrayTy->getElementType();
    LLVM_DEBUG(dbgs() << "  Found array dimension: " << Size << "\n");
  }

  if (Sizes.empty()) {
    LLVM_DEBUG(dbgs() << "No array dimensions found in type\n");
    return false;
  }

  // Add element size as the last element for computeAccessFunctions algorithm.
  Sizes.push_back(ElementSize);

  LLVM_DEBUG({
    dbgs() << "Extracted array info from alloca/global for base pointer "
           << *BasePtr << "\n";
    dbgs() << "Dimensions: ";
    for (const SCEV *Size : Sizes)
      dbgs() << *Size << " ";
    dbgs() << "\n";
  });

  return true;
}

bool llvm::delinearizeUsingArrayInfo(ScalarEvolution &SE, const SCEV *AccessFn,
                                     SmallVectorImpl<const SCEV *> &Subscripts,
                                     SmallVectorImpl<const SCEV *> &Sizes,
                                     const SCEV *ElementSize) {
  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFn));
  if (!BasePointer) {
    LLVM_DEBUG(dbgs() << "no BasePointer for AccessFn: " << *AccessFn << "\n");
    return false;
  }

  Value *BasePtr = BasePointer->getValue();

  // Extract array dimensions from alloca or global declarations.
  if (!extractArrayInfoFromAllocaOrGlobal(SE, BasePtr, Sizes, ElementSize))
    return false;

  // Get the full SCEV expression and subtract the base pointer to get
  // offset-only expression.
  const SCEV *Expr = SE.getMinusSCEV(AccessFn, BasePointer);

  computeAccessFunctions(SE, Expr, Subscripts, Sizes, ElementSize);
  if (Sizes.empty() || Subscripts.empty())
    return false;

  // Validate dimension consistency: subscripts should match array dimensions
  // (Sizes includes element size as last element, so array dimensions =
  // Sizes.size() - 1)
  unsigned ArrayDims = Sizes.size() - 1;
  if (Subscripts.size() != ArrayDims) {
    LLVM_DEBUG(
        dbgs() << "delinearizeUsingArrayInfo: Dimension mismatch - "
               << Subscripts.size() << " subscripts for " << ArrayDims
               << " array dimensions. Falling back to parametric method.\n");
    return false;
  }

  return true;
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access. Returns the remainder of the delinearization that
/// is the offset start of the array.  The SCEV->delinearize algorithm computes
/// the multiples of SCEV coefficients: that is a pattern matching of sub
/// expressions in the stride and base of a SCEV corresponding to the
/// computation of a GCD (greatest common divisor) of base and stride.  When
/// SCEV->delinearize fails, it returns the SCEV unchanged.
///
/// For example: when analyzing the memory access A[i][j][k] in this loop nest
///
///  void foo(long n, long m, long o, double A[n][m][o]) {
///
///    for (long i = 0; i < n; i++)
///      for (long j = 0; j < m; j++)
///        for (long k = 0; k < o; k++)
///          A[i][j][k] = 1.0;
///  }
///
/// the delinearization input is the following AddRec SCEV:
///
///  AddRec: {{{%A,+,(8 * %m * %o)}<%for.i>,+,(8 * %o)}<%for.j>,+,8}<%for.k>
///
/// From this SCEV, we are able to say that the base offset of the access is %A
/// because it appears as an offset that does not divide any of the strides in
/// the loops:
///
///  CHECK: Base offset: %A
///
/// and then SCEV->delinearize determines the size of some of the dimensions of
/// the array as these are the multiples by which the strides are happening:
///
///  CHECK: ArrayDecl[UnknownSize][%m][%o] with elements of sizeof(double)
///  bytes.
///
/// Note that the outermost dimension remains of UnknownSize because there are
/// no strides that would help identifying the size of the last dimension: when
/// the array has been statically allocated, one could compute the size of that
/// dimension by dividing the overall size of the array by the size of the known
/// dimensions: %m * %o * 8.
///
/// Finally delinearize provides the access functions for the array reference
/// that does correspond to A[i][j][k] of the above C testcase:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// The testcases are checking the output of a function pass:
/// DelinearizationPass that walks through all loads and stores of a function
/// asking for the SCEV of the memory access with respect to all enclosing
/// loops, calling SCEV->delinearize on that and printing the results.
void llvm::delinearize(ScalarEvolution &SE, const SCEV *Expr,
                       SmallVectorImpl<const SCEV *> &Subscripts,
                       SmallVectorImpl<const SCEV *> &Sizes,
                       const SCEV *ElementSize) {
  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  // Try array_info extraction.
  if (delinearizeUsingArrayInfo(SE, Expr, Subscripts, Sizes, ElementSize))
    return;
  LLVM_DEBUG(dbgs() << "delinearize falling back to parametric method\n");

  // Fall back to parametric delinearization.
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(Expr));
  if (BasePointer)
    Expr = SE.getMinusSCEV(Expr, BasePointer);

  SmallVector<const SCEV *, 4> Terms;
  collectParametricTerms(SE, Expr, Terms);

  if (Terms.empty())
    return;

  // Find subscript sizes.
  findArrayDimensions(SE, Terms, Sizes, ElementSize);

  if (Sizes.empty())
    return;

  // Compute the access functions for each subscript.
  computeAccessFunctions(SE, Expr, Subscripts, Sizes, ElementSize);
}

static std::optional<APInt> tryIntoAPInt(const SCEV *S) {
  if (const auto *Const = dyn_cast<SCEVConstant>(S))
    return Const->getAPInt();
  return std::nullopt;
}

/// Convert cached SCEV sizes to int sizes for compatibility.
/// TODO: Remove this after we remove GEP delinearization.
static void convertSCEVSizesToIntSizes(ArrayRef<const SCEV *> SCEVSizes,
                                       SmallVectorImpl<int> &Sizes) {
  for (const SCEV *S : SCEVSizes) {
    if (auto *Const = dyn_cast<SCEVConstant>(S)) {
      const APInt &APVal = Const->getAPInt();
      if (APVal.isSignedIntN(32)) {
        int intValue = APVal.getSExtValue();
        Sizes.push_back(intValue);
      }
    }
  }
}

/// Collects the absolute values of constant steps for all induction variables.
/// Returns true if we can prove that all step recurrences are constants and \p
/// Expr is divisible by \p ElementSize. Each step recurrence is stored in \p
/// Steps after divided by \p ElementSize.
static bool collectConstantAbsSteps(ScalarEvolution &SE, const SCEV *Expr,
                                    SmallVectorImpl<uint64_t> &Steps,
                                    uint64_t ElementSize) {
  // End of recursion. The constant value also must be a multiple of
  // ElementSize.
  if (const auto *Const = dyn_cast<SCEVConstant>(Expr)) {
    const uint64_t Mod = Const->getAPInt().urem(ElementSize);
    return Mod == 0;
  }

  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!AR || !AR->isAffine())
    return false;

  const SCEV *Step = AR->getStepRecurrence(SE);
  std::optional<APInt> StepAPInt = tryIntoAPInt(Step);
  if (!StepAPInt)
    return false;

  APInt Q;
  uint64_t R;
  APInt::udivrem(StepAPInt->abs(), ElementSize, Q, R);
  if (R != 0)
    return false;

  // Bail out when the step is too large.
  std::optional<uint64_t> StepVal = Q.tryZExtValue();
  if (!StepVal)
    return false;

  Steps.push_back(*StepVal);
  return collectConstantAbsSteps(SE, AR->getStart(), Steps, ElementSize);
}

bool llvm::findFixedSizeArrayDimensions(ScalarEvolution &SE, const SCEV *Expr,
                                        SmallVectorImpl<uint64_t> &Sizes,
                                        const SCEV *ElementSize) {
  if (!ElementSize)
    return false;

  std::optional<APInt> ElementSizeAPInt = tryIntoAPInt(ElementSize);
  if (!ElementSizeAPInt || *ElementSizeAPInt == 0)
    return false;

  std::optional<uint64_t> ElementSizeConst = ElementSizeAPInt->tryZExtValue();

  // Early exit when ElementSize is not a positive constant.
  if (!ElementSizeConst)
    return false;

  if (!collectConstantAbsSteps(SE, Expr, Sizes, *ElementSizeConst) ||
      Sizes.empty()) {
    Sizes.clear();
    return false;
  }

  // At this point, Sizes contains the absolute step recurrences for all
  // induction variables. Each step recurrence must be a multiple of the size of
  // the array element. Assuming that the each value represents the size of an
  // array for each dimension, attempts to restore the length of each dimension
  // by dividing the step recurrence by the next smaller value. For example, if
  // we have the following AddRec SCEV:
  //
  //   AddRec: {{{0,+,2048}<%for.i>,+,256}<%for.j>,+,8}<%for.k> (ElementSize=8)
  //
  // Then Sizes will become [256, 32, 1] after sorted. We don't know the size of
  // the outermost dimension, the next dimension will be computed as 256 / 32 =
  // 8, and the last dimension will be computed as 32 / 1 = 32. Thus it results
  // in like Arr[UnknownSize][8][32] with elements of size 8 bytes, where Arr is
  // a base pointer.
  //
  // TODO: Catch more cases, e.g., when a step recurrence is not divisible by
  // the next smaller one, like A[i][3*j].
  llvm::sort(Sizes.rbegin(), Sizes.rend());
  Sizes.erase(llvm::unique(Sizes), Sizes.end());

  // The last element in Sizes should be ElementSize. At this point, all values
  // in Sizes are assumed to be divided by ElementSize, so replace it with 1.
  assert(Sizes.back() != 0 && "Unexpected zero size in Sizes.");
  Sizes.back() = 1;

  for (unsigned I = 0; I + 1 < Sizes.size(); I++) {
    uint64_t PrevSize = Sizes[I + 1];
    if (Sizes[I] % PrevSize) {
      Sizes.clear();
      return false;
    }
    Sizes[I] /= PrevSize;
  }

  // Finally, the last element in Sizes should be ElementSize.
  Sizes.back() = *ElementSizeConst;
  return true;
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access, assuming that the array is a fixed size array.
///
/// E.g., if we have the code like as follows:
///
///  double A[42][8][32];
///  for i
///    for j
///      for k
///        use A[i][j][k]
///
/// The access function will be represented as an AddRec SCEV like:
///
///  AddRec: {{{0,+,2048}<%for.i>,+,256}<%for.j>,+,8}<%for.k> (ElementSize=8)
///
/// Then findFixedSizeArrayDimensions infers the size of each dimension of the
/// array based on the fact that the value of the step recurrence is a multiple
/// of the size of the corresponding array element. In the above example, it
/// results in the following:
///
///  CHECK: ArrayDecl[UnknownSize][8][32] with elements of 8 bytes.
///
/// Finally each subscript will be computed as follows:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// Note that this function doesn't check the range of possible values for each
/// subscript, so the caller should perform additional boundary checks if
/// necessary.
///
/// Also note that this function doesn't guarantee that the original array size
/// is restored "correctly". For example, in the following case:
///
///  double A[42][4][64];
///  double B[42][8][32];
///  for i
///    for j
///      for k
///        use A[i][j][k]
///        use B[i][2*j][k]
///
/// The access function for both accesses will be the same:
///
///  AddRec: {{{0,+,2048}<%for.i>,+,512}<%for.j>,+,8}<%for.k> (ElementSize=8)
///
/// The array sizes for both A and B will be computed as
/// ArrayDecl[UnknownSize][4][64], which matches for A, but not for B.
///
/// TODO: At the moment, this function can handle only simple cases. For
/// example, we cannot handle a case where a step recurrence is not divisible
/// by the next smaller step recurrence, e.g., A[i][3*j].
bool llvm::delinearizeFixedSizeArray(ScalarEvolution &SE, const SCEV *Expr,
                                     SmallVectorImpl<const SCEV *> &Subscripts,
                                     SmallVectorImpl<const SCEV *> &Sizes,
                                     const SCEV *ElementSize) {
  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  // First step: find the fixed array size.
  SmallVector<uint64_t, 4> ConstSizes;
  if (!findFixedSizeArrayDimensions(SE, Expr, ConstSizes, ElementSize)) {
    Sizes.clear();
    return false;
  }

  // Convert the constant size to SCEV.
  for (uint64_t Size : ConstSizes)
    Sizes.push_back(SE.getConstant(Expr->getType(), Size));

  // Second step: compute the access functions for each subscript.
  computeAccessFunctions(SE, Expr, Subscripts, Sizes, ElementSize);

  return !Subscripts.empty();
}

bool llvm::tryDelinearizeFixedSizeImpl(
    ScalarEvolution *SE, Instruction *Inst, const SCEV *AccessFn,
    SmallVectorImpl<const SCEV *> &Subscripts, SmallVectorImpl<int> &Sizes) {
  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  Value *SrcPtr = getLoadStorePointerOperand(Inst);

  // Check the simple case where the array dimensions are fixed size.
  auto *SrcGEP = dyn_cast<GetElementPtrInst>(SrcPtr);
  if (!SrcGEP)
    return false;

  // When flag useGEPToDelinearize is false, delinearize only using array_info.
  if (!useGEPToDelinearize) {
    SmallVector<const SCEV *, 4> SCEVSizes;
    const SCEV *ElementSize = SE->getElementSize(Inst);
    if (!delinearizeUsingArrayInfo(*SE, AccessFn, Subscripts, SCEVSizes,
                                   ElementSize))
      return false;

    // TODO: Remove the following code. Convert SCEV sizes to int sizes. This
    // conversion is only needed as long as getIndexExpressionsFromGEP is still
    // around. Remove this code and change the interface of
    // tryDelinearizeFixedSizeImpl to take a SmallVectorImpl<const SCEV *>
    // &Sizes.
    convertSCEVSizesToIntSizes(SCEVSizes, Sizes);
    return true;
  }

  // TODO: Remove all the following code once we are satisfied with array_info.
  // Run both methods when useGEPToDelinearize is true: validation is enabled.

  // Store results from both methods.
  SmallVector<const SCEV *, 4> GEPSubscripts, ArrayInfoSubscripts;
  SmallVector<int, 4> GEPSizes, ArrayInfoSizes;

  // GEP-based delinearization.
  bool GEPSuccess =
      getIndexExpressionsFromGEP(*SE, SrcGEP, GEPSubscripts, GEPSizes);

  // Array_info delinearization.
  SmallVector<const SCEV *, 4> SCEVSizes;
  const SCEV *ElementSize = SE->getElementSize(Inst);
  if (!delinearizeUsingArrayInfo(*SE, AccessFn, Subscripts, SCEVSizes,
                                 ElementSize))
    return false;

  // TODO: Remove the following code. Convert SCEV sizes to int sizes. This
  // conversion is only needed as long as getIndexExpressionsFromGEP is still
  // around. Remove this code and change the interface of
  // tryDelinearizeFixedSizeImpl to take a SmallVectorImpl<const SCEV *> &Sizes.
  convertSCEVSizesToIntSizes(SCEVSizes, Sizes);
  return true;
}

namespace {

void printDelinearization(raw_ostream &O, Function *F, LoopInfo *LI,
                          ScalarEvolution *SE) {
  O << "Printing analysis 'Delinearization' for function '" << F->getName()
    << "':";
  for (Instruction &Inst : instructions(F)) {
    // Only analyze loads and stores.
    if (!isa<StoreInst>(&Inst) && !isa<LoadInst>(&Inst))
      continue;

    const BasicBlock *BB = Inst.getParent();
    Loop *L = LI->getLoopFor(BB);
    // Only delinearize the memory access in the innermost loop.
    // Do not analyze memory accesses outside loops.
    if (!L)
      continue;
    const SCEV *AccessFn = SE->getSCEVAtScope(getPointerOperand(&Inst), L);

    O << "\n";
    O << "Inst:" << Inst << "\n";
    O << "LinearAccessFunction: " << *AccessFn << "\n";

    SmallVector<const SCEV *, 3> Subscripts, Sizes;
    auto IsDelinearizationFailed = [&]() {
      return Subscripts.size() == 0 || Sizes.size() == 0;
    };

    const SCEV *ElementSize = SE->getElementSize(&Inst);
    delinearize(*SE, AccessFn, Subscripts, Sizes, ElementSize);
    if (UseFixedSizeArrayHeuristic && IsDelinearizationFailed()) {
      Subscripts.clear();
      Sizes.clear();

      const SCEVUnknown *BasePointer =
          dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFn));
      // Fail to delinearize if we cannot find the base pointer.
      if (!BasePointer)
        continue;
      AccessFn = SE->getMinusSCEV(AccessFn, BasePointer);

      delinearizeFixedSizeArray(*SE, AccessFn, Subscripts, Sizes,
                                SE->getElementSize(&Inst));
    }

    if (IsDelinearizationFailed()) {
      O << "failed to delinearize\n";
      continue;
    }

    O << "ArrayDecl";
    int NumSubscripts = Subscripts.size();
    int NumSizes = Sizes.size();

    // Handle different size relationships between Subscripts and Sizes.
    if (NumSizes > 0) {
      // Print array dimensions (all but the last size, which is element
      // size).
      for (int i = 0; i < NumSizes - 1; i++)
        O << "[" << *Sizes[i] << "]";

      // Print element size (last element in Sizes array).
      O << " with elements of " << *Sizes[NumSizes - 1] << " bytes.\n";
    } else {
      O << " unknown sizes.\n";
    }

    O << "ArrayRef";
    for (int i = 0; i < NumSubscripts; i++)
      O << "[" << *Subscripts[i] << "]";
    O << "\n";
  }
}

} // end anonymous namespace

DelinearizationPrinterPass::DelinearizationPrinterPass(raw_ostream &OS)
    : OS(OS) {}
PreservedAnalyses DelinearizationPrinterPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  printDelinearization(OS, &F, &AM.getResult<LoopAnalysis>(F),
                       &AM.getResult<ScalarEvolutionAnalysis>(F));
  return PreservedAnalyses::all();
}
