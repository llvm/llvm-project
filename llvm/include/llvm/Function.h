//===-- llvm/Function.h - Class to represent a single function --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Function class, which represents a
// single function/procedure in LLVM.
//
// A function basically consists of a list of basic blocks, a list of arguments,
// and a symbol table.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUNCTION_H
#define LLVM_FUNCTION_H

#include "llvm/GlobalValue.h"
#include "llvm/CallingConv.h"
#include "llvm/BasicBlock.h"
#include "llvm/Argument.h"
#include "llvm/Attributes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class FunctionType;
class LLVMContext;

// Traits for intrusive list of basic blocks...
template<> struct ilist_traits<BasicBlock>
  : public SymbolTableListTraits<BasicBlock, Function> {

  // createSentinel is used to get hold of the node that marks the end of the
  // list... (same trick used here as in ilist_traits<Instruction>)
  BasicBlock *createSentinel() const {
    return static_cast<BasicBlock*>(&Sentinel);
  }
  static void destroySentinel(BasicBlock*) {}

  BasicBlock *provideInitialHead() const { return createSentinel(); }
  BasicBlock *ensureHead(BasicBlock*) const { return createSentinel(); }
  static void noteHead(BasicBlock*, BasicBlock*) {}

  static ValueSymbolTable *getSymTab(Function *ItemParent);
private:
  mutable ilist_half_node<BasicBlock> Sentinel;
};

template<> struct ilist_traits<Argument>
  : public SymbolTableListTraits<Argument, Function> {

  Argument *createSentinel() const {
    return static_cast<Argument*>(&Sentinel);
  }
  static void destroySentinel(Argument*) {}

  Argument *provideInitialHead() const { return createSentinel(); }
  Argument *ensureHead(Argument*) const { return createSentinel(); }
  static void noteHead(Argument*, Argument*) {}

  static ValueSymbolTable *getSymTab(Function *ItemParent);
private:
  mutable ilist_half_node<Argument> Sentinel;
};

class Function : public GlobalValue,
                 public ilist_node<Function> {
public:
  typedef iplist<Argument> ArgumentListType;
  typedef iplist<BasicBlock> BasicBlockListType;

  // BasicBlock iterators...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;

  typedef ArgumentListType::iterator arg_iterator;
  typedef ArgumentListType::const_iterator const_arg_iterator;

private:
  // Important things that make up a function!
  BasicBlockListType  BasicBlocks;        ///< The basic blocks
  mutable ArgumentListType ArgumentList;  ///< The formal arguments
  ValueSymbolTable *SymTab;               ///< Symbol table of args/instructions
  AttrListPtr AttributeList;              ///< Parameter attributes

  // HasLazyArguments is stored in Value::SubclassData.
  /*bool HasLazyArguments;*/
                   
  // The Calling Convention is stored in Value::SubclassData.
  /*CallingConv::ID CallingConvention;*/

  friend class SymbolTableListTraits<Function, Module>;

  void setParent(Module *parent);

  /// hasLazyArguments/CheckLazyArguments - The argument list of a function is
  /// built on demand, so that the list isn't allocated until the first client
  /// needs it.  The hasLazyArguments predicate returns true if the arg list
  /// hasn't been set up yet.
  bool hasLazyArguments() const {
    return getSubclassDataFromValue() & 1;
  }
  void CheckLazyArguments() const {
    if (hasLazyArguments())
      BuildLazyArguments();
  }
  void BuildLazyArguments() const;

  Function(const Function&) LLVM_DELETED_FUNCTION;
  void operator=(const Function&) LLVM_DELETED_FUNCTION;

  /// Function ctor - If the (optional) Module argument is specified, the
  /// function is automatically inserted into the end of the function list for
  /// the module.
  ///
  Function(FunctionType *Ty, LinkageTypes Linkage,
           const Twine &N = "", Module *M = 0);

public:
  static Function *Create(FunctionType *Ty, LinkageTypes Linkage,
                          const Twine &N = "", Module *M = 0) {
    return new(0) Function(Ty, Linkage, N, M);
  }

  ~Function();

  Type *getReturnType() const;           // Return the type of the ret val
  FunctionType *getFunctionType() const; // Return the FunctionType for me

  /// getContext - Return a pointer to the LLVMContext associated with this 
  /// function, or NULL if this function is not bound to a context yet.
  LLVMContext &getContext() const;

  /// isVarArg - Return true if this function takes a variable number of
  /// arguments.
  bool isVarArg() const;

  /// getIntrinsicID - This method returns the ID number of the specified
  /// function, or Intrinsic::not_intrinsic if the function is not an
  /// instrinsic, or if the pointer is null.  This value is always defined to be
  /// zero to allow easy checking for whether a function is intrinsic or not.
  /// The particular intrinsic functions which correspond to this value are
  /// defined in llvm/Intrinsics.h.
  ///
  unsigned getIntrinsicID() const LLVM_READONLY;
  bool isIntrinsic() const { return getIntrinsicID() != 0; }

  /// getCallingConv()/setCallingConv(CC) - These method get and set the
  /// calling convention of this function.  The enum values for the known
  /// calling conventions are defined in CallingConv.h.
  CallingConv::ID getCallingConv() const {
    return static_cast<CallingConv::ID>(getSubclassDataFromValue() >> 1);
  }
  void setCallingConv(CallingConv::ID CC) {
    setValueSubclassData((getSubclassDataFromValue() & 1) |
                         (static_cast<unsigned>(CC) << 1));
  }
  
  /// getAttributes - Return the attribute list for this Function.
  ///
  const AttrListPtr &getAttributes() const { return AttributeList; }

  /// setAttributes - Set the attribute list for this Function.
  ///
  void setAttributes(const AttrListPtr &attrs) { AttributeList = attrs; }

  /// getFnAttributes - Return the function attributes for querying.
  ///
  Attributes getFnAttributes() const {
    return AttributeList.getFnAttributes();
  }

  /// addFnAttr - Add function attributes to this function.
  ///
  void addFnAttr(Attributes::AttrVal N) { 
    // Function Attributes are stored at ~0 index 
    addAttribute(AttrListPtr::FunctionIndex, Attributes::get(getContext(), N));
  }

  /// removeFnAttr - Remove function attributes from this function.
  ///
  void removeFnAttr(Attributes N) {
    // Function Attributes are stored at ~0 index 
    removeAttribute(~0U, N);
  }

  /// hasGC/getGC/setGC/clearGC - The name of the garbage collection algorithm
  ///                             to use during code generation.
  bool hasGC() const;
  const char *getGC() const;
  void setGC(const char *Str);
  void clearGC();


  /// getRetAttributes - Return the return attributes for querying.
  Attributes getRetAttributes() const {
    return AttributeList.getRetAttributes();
  }

  /// getParamAttributes - Return the parameter attributes for querying.
  Attributes getParamAttributes(unsigned Idx) const {
    return AttributeList.getParamAttributes(Idx);
  }

  /// addAttribute - adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attributes attr);
  
  /// removeAttribute - removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, Attributes attr);

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  unsigned getParamAlignment(unsigned i) const {
    return AttributeList.getParamAlignment(i);
  }

  /// @brief Determine if the function does not access memory.
  bool doesNotAccessMemory() const {
    return getFnAttributes().hasAttribute(Attributes::ReadNone);
  }
  void setDoesNotAccessMemory() {
    addFnAttr(Attributes::ReadNone);
  }

  /// @brief Determine if the function does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() ||
      getFnAttributes().hasAttribute(Attributes::ReadOnly);
  }
  void setOnlyReadsMemory() {
    addFnAttr(Attributes::ReadOnly);
  }

  /// @brief Determine if the function cannot return.
  bool doesNotReturn() const {
    return getFnAttributes().hasAttribute(Attributes::NoReturn);
  }
  void setDoesNotReturn() {
    addFnAttr(Attributes::NoReturn);
  }

  /// @brief Determine if the function cannot unwind.
  bool doesNotThrow() const {
    return getFnAttributes().hasAttribute(Attributes::NoUnwind);
  }
  void setDoesNotThrow() {
    addFnAttr(Attributes::NoUnwind);
  }

  /// @brief True if the ABI mandates (or the user requested) that this
  /// function be in a unwind table.
  bool hasUWTable() const {
    return getFnAttributes().hasAttribute(Attributes::UWTable);
  }
  void setHasUWTable() {
    addFnAttr(Attributes::UWTable);
  }

  /// @brief True if this function needs an unwind table.
  bool needsUnwindTableEntry() const {
    return hasUWTable() || !doesNotThrow();
  }

  /// @brief Determine if the function returns a structure through first 
  /// pointer argument.
  bool hasStructRetAttr() const {
    return getParamAttributes(1).hasAttribute(Attributes::StructRet);
  }

  /// @brief Determine if the parameter does not alias other parameters.
  /// @param n The parameter to check. 1 is the first parameter, 0 is the return
  bool doesNotAlias(unsigned n) const {
    return getParamAttributes(n).hasAttribute(Attributes::NoAlias);
  }
  void setDoesNotAlias(unsigned n) {
    addAttribute(n, Attributes::get(getContext(), Attributes::NoAlias));
  }

  /// @brief Determine if the parameter can be captured.
  /// @param n The parameter to check. 1 is the first parameter, 0 is the return
  bool doesNotCapture(unsigned n) const {
    return getParamAttributes(n).hasAttribute(Attributes::NoCapture);
  }
  void setDoesNotCapture(unsigned n) {
    addAttribute(n, Attributes::get(getContext(), Attributes::NoCapture));
  }

  /// copyAttributesFrom - copy all additional attributes (those not needed to
  /// create a Function) from the Function Src to this one.
  void copyAttributesFrom(const GlobalValue *Src);

  /// deleteBody - This method deletes the body of the function, and converts
  /// the linkage to external.
  ///
  void deleteBody() {
    dropAllReferences();
    setLinkage(ExternalLinkage);
  }

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  virtual void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  virtual void eraseFromParent();


  /// Get the underlying elements of the Function... the basic block list is
  /// empty for external functions.
  ///
  const ArgumentListType &getArgumentList() const {
    CheckLazyArguments();
    return ArgumentList;
  }
  ArgumentListType &getArgumentList() {
    CheckLazyArguments();
    return ArgumentList;
  }
  static iplist<Argument> Function::*getSublistAccess(Argument*) {
    return &Function::ArgumentList;
  }

  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }
  static iplist<BasicBlock> Function::*getSublistAccess(BasicBlock*) {
    return &Function::BasicBlocks;
  }

  const BasicBlock       &getEntryBlock() const   { return front(); }
        BasicBlock       &getEntryBlock()         { return front(); }

  //===--------------------------------------------------------------------===//
  // Symbol Table Accessing functions...

  /// getSymbolTable() - Return the symbol table...
  ///
  inline       ValueSymbolTable &getValueSymbolTable()       { return *SymTab; }
  inline const ValueSymbolTable &getValueSymbolTable() const { return *SymTab; }


  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                begin()       { return BasicBlocks.begin(); }
  const_iterator          begin() const { return BasicBlocks.begin(); }
  iterator                end  ()       { return BasicBlocks.end();   }
  const_iterator          end  () const { return BasicBlocks.end();   }

  size_t                   size() const { return BasicBlocks.size();  }
  bool                    empty() const { return BasicBlocks.empty(); }
  const BasicBlock       &front() const { return BasicBlocks.front(); }
        BasicBlock       &front()       { return BasicBlocks.front(); }
  const BasicBlock        &back() const { return BasicBlocks.back();  }
        BasicBlock        &back()       { return BasicBlocks.back();  }

  //===--------------------------------------------------------------------===//
  // Argument iterator forwarding functions
  //
  arg_iterator arg_begin() {
    CheckLazyArguments();
    return ArgumentList.begin();
  }
  const_arg_iterator arg_begin() const {
    CheckLazyArguments();
    return ArgumentList.begin();
  }
  arg_iterator arg_end() {
    CheckLazyArguments();
    return ArgumentList.end();
  }
  const_arg_iterator arg_end() const {
    CheckLazyArguments();
    return ArgumentList.end();
  }

  size_t arg_size() const;
  bool arg_empty() const;

  /// viewCFG - This function is meant for use from the debugger.  You can just
  /// say 'call F->viewCFG()' and a ghostview window should pop up from the
  /// program, displaying the CFG of the current function with the code for each
  /// basic block inside.  This depends on there being a 'dot' and 'gv' program
  /// in your path.
  ///
  void viewCFG() const;

  /// viewCFGOnly - This function is meant for use from the debugger.  It works
  /// just like viewCFG, but it does not include the contents of basic blocks
  /// into the nodes, just the label.  If you are only interested in the CFG
  /// this can make the graph smaller.
  ///
  void viewCFGOnly() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::FunctionVal;
  }

  /// dropAllReferences() - This method causes all the subinstructions to "let
  /// go" of all references that they are maintaining.  This allows one to
  /// 'delete' a whole module at a time, even though there may be circular
  /// references... first all references are dropped, and all use counts go to
  /// zero.  Then everything is deleted for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator
  /// delete.
  ///
  /// Since no other object in the module can have references into the body of a
  /// function, dropping all references deletes the entire body of the function,
  /// including any contained basic blocks.
  ///
  void dropAllReferences();

  /// hasAddressTaken - returns true if there are any uses of this function
  /// other than direct calls or invokes to it, or blockaddress expressions.
  /// Optionally passes back an offending user for diagnostic purposes.
  ///
  bool hasAddressTaken(const User** = 0) const;

  /// isDefTriviallyDead - Return true if it is trivially safe to remove
  /// this function definition from the module (because it isn't externally
  /// visible, does not have its address taken, and has no callers).  To make
  /// this more accurate, call removeDeadConstantUsers first.
  bool isDefTriviallyDead() const;

  /// callsFunctionThatReturnsTwice - Return true if the function has a call to
  /// setjmp or other function that gcc recognizes as "returning twice".
  bool callsFunctionThatReturnsTwice() const;

private:
  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
};

inline ValueSymbolTable *
ilist_traits<BasicBlock>::getSymTab(Function *F) {
  return F ? &F->getValueSymbolTable() : 0;
}

inline ValueSymbolTable *
ilist_traits<Argument>::getSymTab(Function *F) {
  return F ? &F->getValueSymbolTable() : 0;
}

} // End llvm namespace

#endif
