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
/// At the moment, we hardcode i64 bit signed computations. Our experience has
/// shown that 64 bit are generally large enough for the loop bounds that appear
/// in the wild. Signed computations are needed, as loop bounds may become
/// negative.
///
/// FIXME: Hardcoding sizes can cause issues:
///
///   a) Certain run-time checks that we may want to generate can involve the
///      size of the data types the computation is performed on. When code
///      generating these run-time checks to isl_ast_expr[essions], the
///      resulting computation may require more than 64 bit.
///
///   b) On embedded systems and especially for high-level-synthesis 64 bit
///      computations are very costly.
///
///   The right approach is to compute the minimal necessary bitwidth and
///   signedness for each subexpression during in the isl AST generation and
///   to use this information in our IslAstGenerator. Preliminary patches are
///   available, but have not been committed yet.
///
/// 2) We always flag computations with 'nsw'
///
/// As isl_ast_expr[essions] assume arbitrary precision, no wrapping should
/// ever occur in the generated LLVM-IR (assuming the data type chosen is large
/// enough).
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
                 llvm::LoopInfo &LI)
      : S(S), Builder(Builder), IDToValue(IDToValue), GlobalMap(GlobalMap),
        DL(DL), SE(SE), DT(DT), LI(LI) {}

  /// @brief Create LLVM-IR for an isl_ast_expr[ession].
  ///
  /// @param Expr The ast expression for which we generate LLVM-IR.
  ///
  /// @return The llvm::Value* containing the result of the computation.
  llvm::Value *create(__isl_take isl_ast_expr *Expr);

  /// @brief Return the largest of two types.
  ///
  /// @param T1 The first type.
  /// @param T2 The second type.
  ///
  /// @return The largest of the two types.
  llvm::Type *getWidestType(llvm::Type *T1, llvm::Type *T2);

  /// @brief Return the type with which this expression should be computed.
  ///
  /// The type needs to be large enough to hold all possible input and all
  /// possible output values.
  ///
  /// @param Expr The expression for which to find the type.
  /// @return The type with which the expression should be computed.
  llvm::IntegerType *getType(__isl_keep isl_ast_expr *Expr);

private:
  Scop &S;

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
};
}

#endif
