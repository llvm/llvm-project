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
#include "llvm/BasicBlock.h"
#include "llvm/Argument.h"
#include "llvm/Support/Annotation.h"
#include "llvm/Attributes.h"

namespace llvm {

class FunctionType;

// Traits for intrusive list of instructions...
template<> struct ilist_traits<BasicBlock>
  : public SymbolTableListTraits<BasicBlock, Function> {

  // createSentinel is used to create a node that marks the end of the list...
  static BasicBlock *createSentinel();
  static void destroySentinel(BasicBlock *BB) { delete BB; }
  static iplist<BasicBlock> &getList(Function *F);
  static ValueSymbolTable *getSymTab(Function *ItemParent);
  static int getListOffset();
};

template<> struct ilist_traits<Argument>
  : public SymbolTableListTraits<Argument, Function> {

  // createSentinel is used to create a node that marks the end of the list...
  static Argument *createSentinel();
  static void destroySentinel(Argument *A) { delete A; }
  static iplist<Argument> &getList(Function *F);
  static ValueSymbolTable *getSymTab(Function *ItemParent);
  static int getListOffset();
};

class Function : public GlobalValue, public Annotable,
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

  // The Calling Convention is stored in Value::SubclassData.
  /*unsigned CallingConvention;*/

  friend class SymbolTableListTraits<Function, Module>;

  void setParent(Module *parent);

  /// hasLazyArguments/CheckLazyArguments - The argument list of a function is
  /// built on demand, so that the list isn't allocated until the first client
  /// needs it.  The hasLazyArguments predicate returns true if the arg list
  /// hasn't been set up yet.
  bool hasLazyArguments() const {
    return SubclassData & 1;
  }
  void CheckLazyArguments() const {
    if (hasLazyArguments())
      BuildLazyArguments();
  }
  void BuildLazyArguments() const;
  
  Function(const Function&); // DO NOT IMPLEMENT
  void operator=(const Function&); // DO NOT IMPLEMENT

  /// Function ctor - If the (optional) Module argument is specified, the
  /// function is automatically inserted into the end of the function list for
  /// the module.
  ///
  Function(const FunctionType *Ty, LinkageTypes Linkage,
           const std::string &N = "", Module *M = 0);

public:
  static Function *Create(const FunctionType *Ty, LinkageTypes Linkage,
                          const std::string &N = "", Module *M = 0) {
    return new(0) Function(Ty, Linkage, N, M);
  }

  ~Function();

  const Type *getReturnType() const;           // Return the type of the ret val
  const FunctionType *getFunctionType() const; // Return the FunctionType for me

  /// isVarArg - Return true if this function takes a variable number of
  /// arguments.
  bool isVarArg() const;

  /// isDeclaration - Is the body of this function unknown? (The basic block 
  /// list is empty if so.) This is true for function declarations, but not 
  /// true for function definitions.
  ///
  virtual bool isDeclaration() const { return BasicBlocks.empty(); }

  /// getIntrinsicID - This method returns the ID number of the specified
  /// function, or Intrinsic::not_intrinsic if the function is not an
  /// instrinsic, or if the pointer is null.  This value is always defined to be
  /// zero to allow easy checking for whether a function is intrinsic or not.
  /// The particular intrinsic functions which correspond to this value are
  /// defined in llvm/Intrinsics.h.
  ///
  unsigned getIntrinsicID(bool noAssert = false) const;
  bool isIntrinsic() const { return getIntrinsicID() != 0; }

  /// getCallingConv()/setCallingConv(uint) - These method get and set the
  /// calling convention of this function.  The enum values for the known
  /// calling conventions are defined in CallingConv.h.
  unsigned getCallingConv() const { return SubclassData >> 1; }
  void setCallingConv(unsigned CC) {
    SubclassData = (SubclassData & 1) | (CC << 1);
  }
  
  /// getAttributes - Return the attribute list for this Function.
  ///
  const AttrListPtr &getAttributes() const { return AttributeList; }

  /// setAttributes - Set the attribute list for this Function.
  ///
  void setAttributes(const AttrListPtr &attrs) { AttributeList = attrs; }

  /// hasFnAttr - Return true if this function has the given attribute.
  bool hasFnAttr(Attributes N) const {
    // Function Attributes are stored at ~0 index 
    return AttributeList.paramHasAttr(~0U, N);
  }

  /// addFnAttr - Add function attributes to this function.
  ///
  void addFnAttr(Attributes N) { 
    // Function Attributes are stored at ~0 index 
    addAttribute(~0U, N);
  }

  /// hasGC/getGC/setGC/clearGC - The name of the garbage collection algorithm
  ///                             to use during code generation.
  bool hasGC() const;
  const char *getGC() const;
  void setGC(const char *Str);
  void clearGC();

  /// @brief Determine whether the function has the given attribute.
  bool paramHasAttr(unsigned i, Attributes attr) const {
    return AttributeList.paramHasAttr(i, attr);
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
    return paramHasAttr(~0, Attribute::ReadNone);
  }
  void setDoesNotAccessMemory(bool DoesNotAccessMemory = true) {
    if (DoesNotAccessMemory) addAttribute(~0, Attribute::ReadNone);
    else removeAttribute(~0, Attribute::ReadNone);
  }

  /// @brief Determine if the function does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() || paramHasAttr(~0, Attribute::ReadOnly);
  }
  void setOnlyReadsMemory(bool OnlyReadsMemory = true) {
    if (OnlyReadsMemory) addAttribute(~0, Attribute::ReadOnly);
    else removeAttribute(~0, Attribute::ReadOnly | Attribute::ReadNone);
  }

  /// @brief Determine if the function cannot return.
  bool doesNotReturn() const {
    return paramHasAttr(~0, Attribute::NoReturn);
  }
  void setDoesNotReturn(bool DoesNotReturn = true) {
    if (DoesNotReturn) addAttribute(~0, Attribute::NoReturn);
    else removeAttribute(~0, Attribute::NoReturn);
  }

  /// @brief Determine if the function cannot unwind.
  bool doesNotThrow() const {
    return paramHasAttr(~0, Attribute::NoUnwind);
  }
  void setDoesNotThrow(bool DoesNotThrow = true) {
    if (DoesNotThrow) addAttribute(~0, Attribute::NoUnwind);
    else removeAttribute(~0, Attribute::NoUnwind);
  }

  /// @brief Determine if the function returns a structure through first 
  /// pointer argument.
  bool hasStructRetAttr() const {
    return paramHasAttr(1, Attribute::StructRet);
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

  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }

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
  static inline bool classof(const Function *) { return true; }
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
  
  static unsigned getBasicBlockListOffset() {
    Function *Obj = 0;
    return unsigned(reinterpret_cast<uintptr_t>(&Obj->BasicBlocks));
  }
  static unsigned getArgumentListOffset() {
    Function *Obj = 0;
    return unsigned(reinterpret_cast<uintptr_t>(&Obj->ArgumentList));
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

inline int 
ilist_traits<BasicBlock>::getListOffset() {
  return Function::getBasicBlockListOffset();
}

inline int 
ilist_traits<Argument>::getListOffset() {
  return Function::getArgumentListOffset();
}


} // End llvm namespace

#endif
