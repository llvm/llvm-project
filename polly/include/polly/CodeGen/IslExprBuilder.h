//===-IslExprBuilder.h - Helper to generate code for isl AST expressions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISL_EXPR_BUILDER_H
#define POLLY_ISL_EXPR_BUILDER_H

#include "polly/CodeGen/IRBuilder.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/ADT/MapVector.h"
#include "isl/ast.h"

namespace llvm {
class DataLayout;
class ScalarEvolution;
}

struct isl_id;

namespace llvm {
// Provide PointerLikeTypeTraits for isl_id.
template <> class PointerLikeTypeTraits<isl_id *> {

public:
  static inline const void *getAsVoidPointer(isl_id *P) { return (void *)P; }
  static inline const Region *getFromVoidPointer(void *P) {
    return (Region *)P;
  }
  enum { NumLowBitsAvailable = 0 };
};
}

namespace polly {

/// @brief LLVM-IR generator for isl_ast_expr[essions]
///
/// This generator generates LLVM-IR that performs the computation described by
/// an isl_ast_expr[ession].
///
/// Example:
///
///   An isl_ast_expr[ession] can look like this:
///
///     (N + M) + 10
///
///   The IslExprBuilder could create the following LLVM-IR:
///
///     %tmp1 = add nsw i64 %N
///     %tmp2 = add nsw i64 %tmp1, %M
///     %tmp3 = add nsw i64 %tmp2, 10
///
/// The implementation of this class is mostly a mapping from isl_ast_expr
/// constructs to the corresponding LLVM-IR constructs.
///
/// The following decisions may need some explanation:
///
/// 1) Which data-type to choose
///
/// isl_ast_expr[essions] are untyped expressions that assume arbitrary
/// precision integer computations. LLVM-IR instead has fixed size integers.
/// When lowering to LLVM-IR we need to chose both the size of the data type and
/// the sign of the operations we use.
///
/// We use the minimal necessary bit width for the generated code and always
/// interpret values as signed. If operations might overflow (add/sub/mul) we
/// will try to adjust the types in order to ensure a non-wrapping computation.
/// If the type adjustment is not possible (e.g., the necessary type is bigger
/// than the type value of --polly-max-expr-bit-width) we will use assumptions
/// to verify the computation will not wrap. However, for run-time checks we
/// cannot build assumptions but instead utilize overflow tracking intrinsics.
///
/// It is possible to track overflows that occured in the generated IR. See the
/// description of @see OverflowState for more information.
///
class IslExprBuilder {
public:
  /// @brief A map from isl_ids to llvm::Values.
  typedef llvm::MapVector<isl_id *, llvm::AssertingVH<llvm::Value>> IDToValueTy;

  /// @brief Construct an IslExprBuilder.
  ///
  /// @param Builder The IRBuilder used to construct the isl_ast_expr[ession].
  ///                The insert location of this IRBuilder defines WHERE the
  ///                corresponding LLVM-IR is generated.
  ///
  /// @param IDToValue The isl_ast_expr[ession] may reference parameters or
  ///                  variables (identified by an isl_id). The IDTOValue map
  ///                  specifies the LLVM-IR Values that correspond to these
  ///                  parameters and variables.
  IslExprBuilder(Scop &S, PollyIRBuilder &Builder, IDToValueTy &IDToValue,
                 ValueMapT &GlobalMap, const llvm::DataLayout &DL,
                 llvm::ScalarEvolution &SE, llvm::DominatorTree &DT,
                 llvm::LoopInfo &LI);

  /// @brief Create LLVM-IR for an isl_ast_expr[ession].
  ///
  /// @param Expr The ast expression for which we generate LLVM-IR.
  ///
  /// @return The llvm::Value* containing the result of the computation.
  llvm::Value *create(__isl_take isl_ast_expr *Expr);

  /// @brief Unify the types of @p V0 and @p V1 in-place.
  ///
  /// The values @p V0 and @p V1 will be updated in place such that
  ///   type(V0) == type(V1) == MaxType
  /// where MaxType is the larger type of the initial @p V0 and @p V1.
  void unifyTypes(llvm::Value *&V0, llvm::Value *&V1) {
    unifyTypes(V0, V1, V1);
  }

  /// @brief Unify the types of @p V0, @p V1 and @p V2 in-place.
  ///
  /// The same as unifyTypes above but for three values instead of two.
  void unifyTypes(llvm::Value *&V0, llvm::Value *&V1, llvm::Value *&V2);

  /// @brief Adjust the types of @p V0 and @p V1 in-place.
  ///
  /// @param V0               One operand of an operation.
  /// @param V1               Another operand of an operation.
  /// @param RequiredBitWidth The bit with required for a safe operation.
  ///
  /// @return True if the new type has at least @p RequiredBitWidth bits.
  bool adjustTypesForSafeComputation(llvm::Value *&V0, llvm::Value *&V1,
                                     unsigned RequiredBitWidth);

  /// @brief Unify the types of @p LHS and @p RHS in-place for an add/sub op.
  ///
  /// @return False if an additive operation of @p LHS and @p RHS can overflow.
  bool adjustTypesForSafeAddition(llvm::Value *&LHS, llvm::Value *&RHS);

  /// @brief Unify the types of @p LHS and @p RHS in-place for a mul op.
  ///
  /// @return False if a multiplication of @p LHS and @p RHS can overflow.
  bool adjustTypesForSafeMultiplication(llvm::Value *&LHS, llvm::Value *&RHS);

  /// @brief Return the type with which this expression should be computed.
  ///
  /// The type needs to be large enough to hold all possible input and all
  /// possible output values.
  ///
  /// @param Expr The expression for which to find the type.
  /// @return The type with which the expression should be computed.
  llvm::IntegerType *getType(__isl_keep isl_ast_expr *Expr);

  /// @brief Change if runtime overflows are tracked or not.
  ///
  /// @param Enable Flag to enable/disable the tracking.
  ///
  /// Note that this will reset the tracking state and that tracking is only
  /// allowed if the last tracked expression dominates the current insert point.
  void setTrackOverflow(bool Enable);

  /// @brief Return the current overflow status or nullptr if it is not tracked.
  ///
  /// @return A nullptr if tracking is disabled or otherwise an i1 that has the
  ///         value of "0" if and only if no overflow happened since tracking
  ///         was enabled.
  llvm::Value *getOverflowState() const;

private:
  Scop &S;

  /// @brief Flag that will be set if an overflow occurred at runtime.
  ///
  /// Note that this flag is by default a nullptr and if it is a nullptr
  /// we will not record overflows but simply perform the computations.
  /// The intended usage is as follows:
  ///   - If overflows in [an] expression[s] should be tracked, call
  ///     the setTrackOverflow(true) function.
  ///   - Use create(...) for all expressions that should be checked.
  ///   - Call getOverflowState() to get the value representing the current
  ///     state of the overflow flag.
  ///   - To stop tracking call setTrackOverflow(false).
  llvm::Value *OverflowState;

  PollyIRBuilder &Builder;
  IDToValueTy &IDToValue;
  ValueMapT &GlobalMap;

  const llvm::DataLayout &DL;
  llvm::ScalarEvolution &SE;
  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;

  llvm::Value *createOp(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpUnary(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpAccess(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpBin(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpNAry(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpSelect(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpICmp(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpBoolean(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpBooleanConditional(__isl_take isl_ast_expr *Expr);
  llvm::Value *createId(__isl_take isl_ast_expr *Expr);
  llvm::Value *createInt(__isl_take isl_ast_expr *Expr);
  llvm::Value *createOpAddressOf(__isl_take isl_ast_expr *Expr);
  llvm::Value *createAccessAddress(__isl_take isl_ast_expr *Expr);

  /// @brief Supported kinds of division.
  enum DivisionMode { DM_SIGNED, DM_UNSIGNED, DM_FLOORED };

  /// @brief Create a binary operation @p Opc and track overflows if requested.
  ///
  /// @param OpC  The binary operation that should be performed [Add/Sub/Mul].
  /// @param LHS  The left operand.
  /// @param RHS  The right operand.
  /// @param Name The (base) name of the new IR operations.
  ///
  /// @return A value that represents the result of the binary operation.
  llvm::Value *createBinOp(llvm::BinaryOperator::BinaryOps Opc,
                           llvm::Value *LHS, llvm::Value *RHS,
                           const llvm::Twine &Name);

  /// @brief Create a division and adjust the result type if possible.
  ///
  /// @param LHS The left operand.
  /// @param RHS The right operand.
  /// @param DM  The requested division mode.
  ///
  /// @return A value that represents the result of the division.
  llvm::Value *createDiv(llvm::Value *LHS, llvm::Value *RHS, DivisionMode DM);

  /// @brief Create an addition and track overflows if requested.
  ///
  /// @param LHS  The left operand.
  /// @param RHS  The right operand.
  /// @param Name The (base) name of the new IR operations.
  ///
  /// @return A value that represents the result of the addition.
  llvm::Value *createAdd(llvm::Value *LHS, llvm::Value *RHS,
                         const llvm::Twine &Name = "");

  /// @brief Create a subtraction and track overflows if requested.
  ///
  /// @param LHS  The left operand.
  /// @param RHS  The right operand.
  /// @param Name The (base) name of the new IR operations.
  ///
  /// @return A value that represents the result of the subtraction.
  llvm::Value *createSub(llvm::Value *LHS, llvm::Value *RHS,
                         const llvm::Twine &Name = "");

  /// @brief Create a multiplication and track overflows if requested.
  ///
  /// @param LHS  The left operand.
  /// @param RHS  The right operand.
  /// @param Name The (base) name of the new IR operations.
  ///
  /// @return A value that represents the result of the multiplication.
  llvm::Value *createMul(llvm::Value *LHS, llvm::Value *RHS,
                         const llvm::Twine &Name = "");
};
}

#endif
