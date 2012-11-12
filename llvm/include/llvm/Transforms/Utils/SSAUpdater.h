//===-- SSAUpdater.h - Unstructured SSA Update Tool -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SSAUpdater class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SSAUPDATER_H
#define LLVM_TRANSFORMS_UTILS_SSAUPDATER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
  class BasicBlock;
  class Instruction;
  class LoadInst;
  template<typename T> class SmallVectorImpl;
  template<typename T> class SSAUpdaterTraits;
  class PHINode;
  class Type;
  class Use;
  class Value;

/// SSAUpdater - This class updates SSA form for a set of values defined in
/// multiple blocks.  This is used when code duplication or another unstructured
/// transformation wants to rewrite a set of uses of one value with uses of a
/// set of values.
class SSAUpdater {
  friend class SSAUpdaterTraits<SSAUpdater>;

private:
  /// AvailableVals - This keeps track of which value to use on a per-block
  /// basis.  When we insert PHI nodes, we keep track of them here.
  //typedef DenseMap<BasicBlock*, Value*> AvailableValsTy;
  void *AV;

  /// ProtoType holds the type of the values being rewritten.
  Type *ProtoType;

  // PHI nodes are given a name based on ProtoName.
  std::string ProtoName;

  /// InsertedPHIs - If this is non-null, the SSAUpdater adds all PHI nodes that
  /// it creates to the vector.
  SmallVectorImpl<PHINode*> *InsertedPHIs;

public:
  /// SSAUpdater constructor.  If InsertedPHIs is specified, it will be filled
  /// in with all PHI Nodes created by rewriting.
  explicit SSAUpdater(SmallVectorImpl<PHINode*> *InsertedPHIs = 0);
  ~SSAUpdater();

  /// Initialize - Reset this object to get ready for a new set of SSA
  /// updates with type 'Ty'.  PHI nodes get a name based on 'Name'.
  void Initialize(Type *Ty, StringRef Name);

  /// AddAvailableValue - Indicate that a rewritten value is available at the
  /// end of the specified block with the specified value.
  void AddAvailableValue(BasicBlock *BB, Value *V);

  /// HasValueForBlock - Return true if the SSAUpdater already has a value for
  /// the specified block.
  bool HasValueForBlock(BasicBlock *BB) const;

  /// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
  /// live at the end of the specified block.
  Value *GetValueAtEndOfBlock(BasicBlock *BB);

  /// GetValueInMiddleOfBlock - Construct SSA form, materializing a value that
  /// is live in the middle of the specified block.
  ///
  /// GetValueInMiddleOfBlock is the same as GetValueAtEndOfBlock except in one
  /// important case: if there is a definition of the rewritten value after the
  /// 'use' in BB.  Consider code like this:
  ///
  ///      X1 = ...
  ///   SomeBB:
  ///      use(X)
  ///      X2 = ...
  ///      br Cond, SomeBB, OutBB
  ///
  /// In this case, there are two values (X1 and X2) added to the AvailableVals
  /// set by the client of the rewriter, and those values are both live out of
  /// their respective blocks.  However, the use of X happens in the *middle* of
  /// a block.  Because of this, we need to insert a new PHI node in SomeBB to
  /// merge the appropriate values, and this value isn't live out of the block.
  ///
  Value *GetValueInMiddleOfBlock(BasicBlock *BB);

  /// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
  /// which use their value in the corresponding predecessor.  Note that this
  /// will not work if the use is supposed to be rewritten to a value defined in
  /// the same block as the use, but above it.  Any 'AddAvailableValue's added
  /// for the use's block will be considered to be below it.
  void RewriteUse(Use &U);

  /// RewriteUseAfterInsertions - Rewrite a use, just like RewriteUse.  However,
  /// this version of the method can rewrite uses in the same block as a
  /// definition, because it assumes that all uses of a value are below any
  /// inserted values.
  void RewriteUseAfterInsertions(Use &U);

private:
  Value *GetValueAtEndOfBlockInternal(BasicBlock *BB);

  void operator=(const SSAUpdater&) LLVM_DELETED_FUNCTION;
  SSAUpdater(const SSAUpdater&) LLVM_DELETED_FUNCTION;
};
  
/// LoadAndStorePromoter - This little helper class provides a convenient way to
/// promote a collection of loads and stores into SSA Form using the SSAUpdater.
/// This handles complexities that SSAUpdater doesn't, such as multiple loads
/// and stores in one block.
///
/// Clients of this class are expected to subclass this and implement the
/// virtual methods.
///
class LoadAndStorePromoter {
protected:
  SSAUpdater &SSA;
public:
  LoadAndStorePromoter(const SmallVectorImpl<Instruction*> &Insts,
                       SSAUpdater &S, StringRef Name = StringRef());
  virtual ~LoadAndStorePromoter() {}
  
  /// run - This does the promotion.  Insts is a list of loads and stores to
  /// promote, and Name is the basename for the PHIs to insert.  After this is
  /// complete, the loads and stores are removed from the code.
  void run(const SmallVectorImpl<Instruction*> &Insts) const;
  
  
  /// Return true if the specified instruction is in the Inst list (which was
  /// passed into the run method).  Clients should implement this with a more
  /// efficient version if possible.
  virtual bool isInstInList(Instruction *I,
                            const SmallVectorImpl<Instruction*> &Insts) const;
  
  /// doExtraRewritesBeforeFinalDeletion - This hook is invoked after all the
  /// stores are found and inserted as available values, but 
  virtual void doExtraRewritesBeforeFinalDeletion() const {
  }
  
  /// replaceLoadWithValue - Clients can choose to implement this to get
  /// notified right before a load is RAUW'd another value.
  virtual void replaceLoadWithValue(LoadInst *LI, Value *V) const {
  }

  /// This is called before each instruction is deleted.
  virtual void instructionDeleted(Instruction *I) const {
  }

  /// updateDebugInfo - This is called to update debug info associated with the
  /// instruction.
  virtual void updateDebugInfo(Instruction *I) const {
  }
};

} // End llvm namespace

#endif
