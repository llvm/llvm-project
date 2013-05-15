//===-BlockGenerators.h - Helper to generate code for statements-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the BlockGenerator and VectorBlockGenerator classes, which
// generate sequential code and vectorized code for a polyhedral statement,
// respectively.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_BLOCK_GENERATORS_H
#define POLLY_BLOCK_GENERATORS_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "isl/map.h"

#include <vector>

namespace llvm {
class Pass;
class Region;
class ScalarEvolution;
}

namespace polly {
extern bool SCEVCodegen;

using namespace llvm;
class ScopStmt;

typedef DenseMap<const Value *, Value *> ValueMapT;
typedef std::vector<ValueMapT> VectorValueMapT;

/// @brief Check whether an instruction can be synthesized by the code
///        generator.
///
/// Some instructions will be recalculated only from information that is code
/// generated from the polyhedral representation. For such instructions we do
/// not need to ensure that their operands are available during code generation.
///
/// @param I The instruction to check.
/// @param LI The LoopInfo analysis.
/// @param SE The scalar evolution database.
/// @param R The region out of which SSA names are parameters.
/// @return If the instruction I can be regenerated from its
///         scalar evolution representation, return true,
///         otherwise return false.
bool canSynthesize(const llvm::Instruction *I, const llvm::LoopInfo *LI,
                   llvm::ScalarEvolution *SE, const llvm::Region *R);

/// @brief Generate a new basic block for a polyhedral statement.
///
/// The only public function exposed is generate().
class BlockGenerator {
public:
  /// @brief Generate a new BasicBlock for a ScopStmt.
  ///
  /// @param Builder   The LLVM-IR Builder used to generate the statement. The
  ///                  code is generated at the location, the Builder points to.
  /// @param Stmt      The statement to code generate.
  /// @param GlobalMap A map that defines for certain Values referenced from the
  ///                  original code new Values they should be replaced with.
  /// @param P         A reference to the pass this function is called from.
  ///                  The pass is needed to update other analysis.
  static void generate(IRBuilder<> &Builder, ScopStmt &Stmt,
                       ValueMapT &GlobalMap, LoopToScevMapT &LTS, Pass *P) {
    BlockGenerator Generator(Builder, Stmt, P);
    Generator.copyBB(GlobalMap, LTS);
  }

protected:
  IRBuilder<> &Builder;
  ScopStmt &Statement;
  Pass *P;
  ScalarEvolution &SE;

  BlockGenerator(IRBuilder<> &B, ScopStmt &Stmt, Pass *P);

  /// @brief Get the new version of a Value.
  ///
  /// @param Old       The old Value.
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block).
  /// @param LTS       A mapping from loops virtual canonical induction
  ///                  variable to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block).
  /// @param L         The loop that surrounded the instruction that referenced
  ///                  this value in the original code. This loop is used to
  ///                  evaluate the scalar evolution at the right scope.
  ///
  /// @returns  o The old value, if it is still valid.
  ///           o The new value, if available.
  ///           o NULL, if no value is found.
  Value *getNewValue(const Value *Old, ValueMapT &BBMap, ValueMapT &GlobalMap,
                     LoopToScevMapT &LTS, Loop *L);

  void copyInstScalar(const Instruction *Inst, ValueMapT &BBMap,
                      ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  /// @brief Get the innermost loop that surrounds an instruction.
  ///
  /// @param Inst The instruction for which we get the loop.
  /// @return The innermost loop that surrounds the instruction.
  Loop *getLoopForInst(const Instruction *Inst);

  /// @brief Get the memory access offset to be added to the base address
  ///
  /// @param L The loop that surrounded the instruction that referenced this
  ///          memory subscript in the original code.
  std::vector<Value *> getMemoryAccessIndex(__isl_keep isl_map *AccessRelation,
                                            Value *BaseAddress,
                                            ValueMapT &BBMap,
                                            ValueMapT &GlobalMap,
                                            LoopToScevMapT &LTS, Loop *L);

  /// @brief Get the new operand address according to the changed access in
  ///        JSCOP file.
  ///
  /// @param L The loop that surrounded the instruction that used this operand
  ///          in the original code.
  Value *getNewAccessOperand(__isl_keep isl_map *NewAccessRelation,
                             Value *BaseAddress, ValueMapT &BBMap,
                             ValueMapT &GlobalMap, LoopToScevMapT &LTS,
                             Loop *L);

  /// @brief Generate the operand address
  Value *generateLocationAccessed(const Instruction *Inst, const Value *Pointer,
                                  ValueMapT &BBMap, ValueMapT &GlobalMap,
                                  LoopToScevMapT &LTS);

  Value *generateScalarLoad(const LoadInst *load, ValueMapT &BBMap,
                            ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  Value *generateScalarStore(const StoreInst *store, ValueMapT &BBMap,
                             ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  /// @brief Copy a single Instruction.
  ///
  /// This copies a single Instruction and updates references to old values
  /// with references to new values, as defined by GlobalMap and BBMap.
  ///
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  void copyInstruction(const Instruction *Inst, ValueMapT &BBMap,
                       ValueMapT &GlobalMap, LoopToScevMapT &LTS);

  /// @brief Copy the basic block.
  ///
  /// This copies the entire basic block and updates references to old values
  /// with references to new values, as defined by GlobalMap.
  ///
  /// @param GlobalMap A mapping from old values to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                  within this basic block).
  void copyBB(ValueMapT &GlobalMap, LoopToScevMapT &LTS);
};

/// @brief Generate a new vector basic block for a polyhedral statement.
///
/// The only public function exposed is generate().
class VectorBlockGenerator : BlockGenerator {
public:
  /// @brief Generate a new vector basic block for a ScoPStmt.
  ///
  /// This code generation is similar to the normal, scalar code generation,
  /// except that each instruction is code generated for several vector lanes
  /// at a time. If possible instructions are issued as actual vector
  /// instructions, but e.g. for address calculation instructions we currently
  /// generate scalar instructions for each vector lane.
  ///
  /// @param Stmt       The statement to code generate.
  /// @param GlobalMaps A vector of maps that define for certain Values
  ///                   referenced from the original code new Values they should
  ///                   be replaced with. Each map in the vector of maps is
  ///                   used for one vector lane. The number of elements in the
  ///                   vector defines the width of the generated vector
  ///                   instructions.
  /// @param Schedule   A map from the statement to a schedule where the
  ///                   innermost dimension is the dimension of the innermost
  ///                   loop containing the statemenet.
  /// @param P          A reference to the pass this function is called from.
  ///                   The pass is needed to update other analysis.
  static void generate(IRBuilder<> &B, ScopStmt &Stmt,
                       VectorValueMapT &GlobalMaps,
                       std::vector<LoopToScevMapT> &VLTS,
                       __isl_keep isl_map *Schedule, Pass *P) {
    VectorBlockGenerator Generator(B, GlobalMaps, VLTS, Stmt, Schedule, P);
    Generator.copyBB();
  }

private:
  // This is a vector of global value maps.  The first map is used for the first
  // vector lane, ...
  // Each map, contains information about Instructions in the old ScoP, which
  // are recalculated in the new SCoP. When copying the basic block, we replace
  // all referenes to the old instructions with their recalculated values.
  VectorValueMapT &GlobalMaps;

  // This is a vector of loop->scev maps.  The first map is used for the first
  // vector lane, ...
  // Each map, contains information about Instructions in the old ScoP, which
  // are recalculated in the new SCoP. When copying the basic block, we replace
  // all referenes to the old instructions with their recalculated values.
  //
  // For example, when the code generator produces this AST:
  //
  //   for (int c1 = 0; c1 <= 1023; c1 += 1)
  //     for (int c2 = 0; c2 <= 1023; c2 += VF)
  //       for (int lane = 0; lane <= VF; lane += 1)
  //         Stmt(c2 + lane + 3, c1);
  //
  // VLTS[lane] contains a map:
  //   "outer loop in the old loop nest" -> SCEV("c2 + lane + 3"),
  //   "inner loop in the old loop nest" -> SCEV("c1").
  std::vector<LoopToScevMapT> &VLTS;

  // A map from the statement to a schedule where the innermost dimension is the
  // dimension of the innermost loop containing the statemenet.
  isl_map *Schedule;

  VectorBlockGenerator(IRBuilder<> &B, VectorValueMapT &GlobalMaps,
                       std::vector<LoopToScevMapT> &VLTS, ScopStmt &Stmt,
                       __isl_keep isl_map *Schedule, Pass *P);

  int getVectorWidth();

  Value *getVectorValue(const Value *Old, ValueMapT &VectorMap,
                        VectorValueMapT &ScalarMaps, Loop *L);

  Type *getVectorPtrTy(const Value *V, int Width);

  /// @brief Load a vector from a set of adjacent scalars
  ///
  /// In case a set of scalars is known to be next to each other in memory,
  /// create a vector load that loads those scalars
  ///
  /// %vector_ptr= bitcast double* %p to <4 x double>*
  /// %vec_full = load <4 x double>* %vector_ptr
  ///
  Value *generateStrideOneLoad(const LoadInst *Load, ValueMapT &BBMap);

  /// @brief Load a vector initialized from a single scalar in memory
  ///
  /// In case all elements of a vector are initialized to the same
  /// scalar value, this value is loaded and shuffeled into all elements
  /// of the vector.
  ///
  /// %splat_one = load <1 x double>* %p
  /// %splat = shufflevector <1 x double> %splat_one, <1 x
  ///       double> %splat_one, <4 x i32> zeroinitializer
  ///
  Value *generateStrideZeroLoad(const LoadInst *Load, ValueMapT &BBMap);

  /// @brief Load a vector from scalars distributed in memory
  ///
  /// In case some scalars a distributed randomly in memory. Create a vector
  /// by loading each scalar and by inserting one after the other into the
  /// vector.
  ///
  /// %scalar_1= load double* %p_1
  /// %vec_1 = insertelement <2 x double> undef, double %scalar_1, i32 0
  /// %scalar 2 = load double* %p_2
  /// %vec_2 = insertelement <2 x double> %vec_1, double %scalar_1, i32 1
  ///
  Value *generateUnknownStrideLoad(const LoadInst *Load,
                                   VectorValueMapT &ScalarMaps);

  void generateLoad(const LoadInst *Load, ValueMapT &VectorMap,
                    VectorValueMapT &ScalarMaps);

  void copyUnaryInst(const UnaryInstruction *Inst, ValueMapT &VectorMap,
                     VectorValueMapT &ScalarMaps);

  void copyBinaryInst(const BinaryOperator *Inst, ValueMapT &VectorMap,
                      VectorValueMapT &ScalarMaps);

  void copyStore(const StoreInst *Store, ValueMapT &VectorMap,
                 VectorValueMapT &ScalarMaps);

  void copyInstScalarized(const Instruction *Inst, ValueMapT &VectorMap,
                          VectorValueMapT &ScalarMaps);

  bool extractScalarValues(const Instruction *Inst, ValueMapT &VectorMap,
                           VectorValueMapT &ScalarMaps);

  bool hasVectorOperands(const Instruction *Inst, ValueMapT &VectorMap);

  void copyInstruction(const Instruction *Inst, ValueMapT &VectorMap,
                       VectorValueMapT &ScalarMaps);

  void copyBB();
};

}
#endif
