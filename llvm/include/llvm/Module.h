//===-- llvm/Module.h - C++ class to represent a VM module ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// Module.h This file contains the declarations for the Module class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULE_H
#define LLVM_MODULE_H

#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class FunctionType;
class GVMaterializer;
class LLVMContext;
class StructType;
template<typename T> struct DenseMapInfo;
template<typename KeyT, typename ValueT, typename KeyInfoT> class DenseMap;

template<> struct ilist_traits<Function>
  : public SymbolTableListTraits<Function, Module> {

  // createSentinel is used to get hold of the node that marks the end of the
  // list... (same trick used here as in ilist_traits<Instruction>)
  Function *createSentinel() const {
    return static_cast<Function*>(&Sentinel);
  }
  static void destroySentinel(Function*) {}

  Function *provideInitialHead() const { return createSentinel(); }
  Function *ensureHead(Function*) const { return createSentinel(); }
  static void noteHead(Function*, Function*) {}

private:
  mutable ilist_node<Function> Sentinel;
};

template<> struct ilist_traits<GlobalVariable>
  : public SymbolTableListTraits<GlobalVariable, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  GlobalVariable *createSentinel() const {
    return static_cast<GlobalVariable*>(&Sentinel);
  }
  static void destroySentinel(GlobalVariable*) {}

  GlobalVariable *provideInitialHead() const { return createSentinel(); }
  GlobalVariable *ensureHead(GlobalVariable*) const { return createSentinel(); }
  static void noteHead(GlobalVariable*, GlobalVariable*) {}
private:
  mutable ilist_node<GlobalVariable> Sentinel;
};

template<> struct ilist_traits<GlobalAlias>
  : public SymbolTableListTraits<GlobalAlias, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  GlobalAlias *createSentinel() const {
    return static_cast<GlobalAlias*>(&Sentinel);
  }
  static void destroySentinel(GlobalAlias*) {}

  GlobalAlias *provideInitialHead() const { return createSentinel(); }
  GlobalAlias *ensureHead(GlobalAlias*) const { return createSentinel(); }
  static void noteHead(GlobalAlias*, GlobalAlias*) {}
private:
  mutable ilist_node<GlobalAlias> Sentinel;
};

template<> struct ilist_traits<NamedMDNode>
  : public ilist_default_traits<NamedMDNode> {
  // createSentinel is used to get hold of a node that marks the end of
  // the list...
  NamedMDNode *createSentinel() const {
    return static_cast<NamedMDNode*>(&Sentinel);
  }
  static void destroySentinel(NamedMDNode*) {}

  NamedMDNode *provideInitialHead() const { return createSentinel(); }
  NamedMDNode *ensureHead(NamedMDNode*) const { return createSentinel(); }
  static void noteHead(NamedMDNode*, NamedMDNode*) {}
  void addNodeToList(NamedMDNode *) {}
  void removeNodeFromList(NamedMDNode *) {}
private:
  mutable ilist_node<NamedMDNode> Sentinel;
};

/// A Module instance is used to store all the information related to an
/// LLVM module. Modules are the top level container of all other LLVM
/// Intermediate Representation (IR) objects. Each module directly contains a
/// list of globals variables, a list of functions, a list of libraries (or
/// other modules) this module depends on, a symbol table, and various data
/// about the target's characteristics.
///
/// A module maintains a GlobalValRefMap object that is used to hold all
/// constant references to global variables in the module.  When a global
/// variable is destroyed, it should have no entries in the GlobalValueRefMap.
/// @brief The main container class for the LLVM Intermediate Representation.
class Module {
/// @name Types And Enumerations
/// @{
public:
  /// The type for the list of global variables.
  typedef iplist<GlobalVariable> GlobalListType;
  /// The type for the list of functions.
  typedef iplist<Function> FunctionListType;
  /// The type for the list of aliases.
  typedef iplist<GlobalAlias> AliasListType;
  /// The type for the list of named metadata.
  typedef ilist<NamedMDNode> NamedMDListType;

  /// The type for the list of dependent libraries.
  typedef std::vector<std::string> LibraryListType;

  /// The Global Variable iterator.
  typedef GlobalListType::iterator                      global_iterator;
  /// The Global Variable constant iterator.
  typedef GlobalListType::const_iterator          const_global_iterator;

  /// The Function iterators.
  typedef FunctionListType::iterator                           iterator;
  /// The Function constant iterator
  typedef FunctionListType::const_iterator               const_iterator;

  /// The Global Alias iterators.
  typedef AliasListType::iterator                        alias_iterator;
  /// The Global Alias constant iterator
  typedef AliasListType::const_iterator            const_alias_iterator;

  /// The named metadata iterators.
  typedef NamedMDListType::iterator             named_metadata_iterator;
  /// The named metadata constant interators.
  typedef NamedMDListType::const_iterator const_named_metadata_iterator;
  /// The Library list iterator.
  typedef LibraryListType::const_iterator lib_iterator;

  /// An enumeration for describing the endianess of the target machine.
  enum Endianness  { AnyEndianness, LittleEndian, BigEndian };

  /// An enumeration for describing the size of a pointer on the target machine.
  enum PointerSize { AnyPointerSize, Pointer32, Pointer64 };

  /// An enumeration for the supported behaviors of module flags. The following
  /// module flags behavior values are supported:
  ///
  ///    Value        Behavior
  ///    -----        --------
  ///      1          Error
  ///                   Emits an error if two values disagree.
  ///
  ///      2          Warning
  ///                   Emits a warning if two values disagree.
  ///
  ///      3          Require
  ///                   Emits an error when the specified value is not present
  ///                   or doesn't have the specified value. It is an error for
  ///                   two (or more) llvm.module.flags with the same ID to have
  ///                   the Require behavior but different values. There may be
  ///                   multiple Require flags per ID.
  ///
  ///      4          Override
  ///                   Uses the specified value if the two values disagree. It
  ///                   is an error for two (or more) llvm.module.flags with the
  ///                   same ID to have the Override behavior but different
  ///                   values.
  enum ModFlagBehavior { Error = 1, Warning  = 2, Require = 3, Override = 4 };

  struct ModuleFlagEntry {
    ModFlagBehavior Behavior;
    MDString *Key;
    Value *Val;
    ModuleFlagEntry(ModFlagBehavior B, MDString *K, Value *V)
      : Behavior(B), Key(K), Val(V) {}
  };

/// @}
/// @name Member Variables
/// @{
private:
  LLVMContext &Context;           ///< The LLVMContext from which types and
                                  ///< constants are allocated.
  GlobalListType GlobalList;      ///< The Global Variables in the module
  FunctionListType FunctionList;  ///< The Functions in the module
  AliasListType AliasList;        ///< The Aliases in the module
  LibraryListType LibraryList;    ///< The Libraries needed by the module
  NamedMDListType NamedMDList;    ///< The named metadata in the module
  std::string GlobalScopeAsm;     ///< Inline Asm at global scope.
  ValueSymbolTable *ValSymTab;    ///< Symbol table for values
  OwningPtr<GVMaterializer> Materializer;  ///< Used to materialize GlobalValues
  std::string ModuleID;           ///< Human readable identifier for the module
  std::string TargetTriple;       ///< Platform target triple Module compiled on
  std::string DataLayout;         ///< Target data description
  void *NamedMDSymTab;            ///< NamedMDNode names.

  friend class Constant;

/// @}
/// @name Constructors
/// @{
public:
  /// The Module constructor. Note that there is no default constructor. You
  /// must provide a name for the module upon construction.
  explicit Module(StringRef ModuleID, LLVMContext& C);
  /// The module destructor. This will dropAllReferences.
  ~Module();

/// @}
/// @name Module Level Accessors
/// @{

  /// Get the module identifier which is, essentially, the name of the module.
  /// @returns the module identifier as a string
  const std::string &getModuleIdentifier() const { return ModuleID; }

  /// Get the data layout string for the module's target platform.  This encodes
  /// the type sizes and alignments expected by this module.
  /// @returns the data layout as a string
  const std::string &getDataLayout() const { return DataLayout; }

  /// Get the target triple which is a string describing the target host.
  /// @returns a string containing the target triple.
  const std::string &getTargetTriple() const { return TargetTriple; }

  /// Get the target endian information.
  /// @returns Endianess - an enumeration for the endianess of the target
  Endianness getEndianness() const;

  /// Get the target pointer size.
  /// @returns PointerSize - an enumeration for the size of the target's pointer
  PointerSize getPointerSize() const;

  /// Get the global data context.
  /// @returns LLVMContext - a container for LLVM's global information
  LLVMContext &getContext() const { return Context; }

  /// Get any module-scope inline assembly blocks.
  /// @returns a string containing the module-scope inline assembly blocks.
  const std::string &getModuleInlineAsm() const { return GlobalScopeAsm; }

/// @}
/// @name Module Level Mutators
/// @{

  /// Set the module identifier.
  void setModuleIdentifier(StringRef ID) { ModuleID = ID; }

  /// Set the data layout
  void setDataLayout(StringRef DL) { DataLayout = DL; }

  /// Set the target triple.
  void setTargetTriple(StringRef T) { TargetTriple = T; }

  /// Set the module-scope inline assembly blocks.
  void setModuleInlineAsm(StringRef Asm) {
    GlobalScopeAsm = Asm;
    if (!GlobalScopeAsm.empty() &&
        GlobalScopeAsm[GlobalScopeAsm.size()-1] != '\n')
      GlobalScopeAsm += '\n';
  }

  /// Append to the module-scope inline assembly blocks, automatically inserting
  /// a separating newline if necessary.
  void appendModuleInlineAsm(StringRef Asm) {
    GlobalScopeAsm += Asm;
    if (!GlobalScopeAsm.empty() &&
        GlobalScopeAsm[GlobalScopeAsm.size()-1] != '\n')
      GlobalScopeAsm += '\n';
  }

/// @}
/// @name Generic Value Accessors
/// @{

  /// getNamedValue - Return the global value in the module with
  /// the specified name, of arbitrary type.  This method returns null
  /// if a global with the specified name is not found.
  GlobalValue *getNamedValue(StringRef Name) const;

  /// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
  /// This ID is uniqued across modules in the current LLVMContext.
  unsigned getMDKindID(StringRef Name) const;

  /// getMDKindNames - Populate client supplied SmallVector with the name for
  /// custom metadata IDs registered in this LLVMContext.
  void getMDKindNames(SmallVectorImpl<StringRef> &Result) const;

  
  typedef DenseMap<StructType*, unsigned, DenseMapInfo<StructType*> >
                   NumeredTypesMapTy;

  /// getTypeByName - Return the type with the specified name, or null if there
  /// is none by that name.
  StructType *getTypeByName(StringRef Name) const;

/// @}
/// @name Function Accessors
/// @{

  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  Four possibilities:
  ///   1. If it does not exist, add a prototype for the function and return it.
  ///   2. If it exists, and has a local linkage, the existing function is
  ///      renamed and a new one is inserted.
  ///   3. Otherwise, if the existing function has the correct prototype, return
  ///      the existing function.
  ///   4. Finally, the function exists but has the wrong prototype: return the
  ///      function with a constantexpr cast to the right prototype.
  Constant *getOrInsertFunction(StringRef Name, FunctionType *T,
                                AttrListPtr AttributeList);

  Constant *getOrInsertFunction(StringRef Name, FunctionType *T);

  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  If it does not exist, add a prototype for the function and return
  /// it.  This function guarantees to return a constant of pointer to the
  /// specified function type or a ConstantExpr BitCast of that type if the
  /// named function has a different type.  This version of the method takes a
  /// null terminated list of function arguments, which makes it easier for
  /// clients to use.
  Constant *getOrInsertFunction(StringRef Name,
                                AttrListPtr AttributeList,
                                Type *RetTy, ...)  END_WITH_NULL;

  /// getOrInsertFunction - Same as above, but without the attributes.
  Constant *getOrInsertFunction(StringRef Name, Type *RetTy, ...)
    END_WITH_NULL;

  Constant *getOrInsertTargetIntrinsic(StringRef Name,
                                       FunctionType *Ty,
                                       AttrListPtr AttributeList);

  /// getFunction - Look up the specified function in the module symbol table.
  /// If it does not exist, return null.
  Function *getFunction(StringRef Name) const;

/// @}
/// @name Global Variable Accessors
/// @{

  /// getGlobalVariable - Look up the specified global variable in the module
  /// symbol table.  If it does not exist, return null. If AllowInternal is set
  /// to true, this function will return types that have InternalLinkage. By
  /// default, these types are not returned.
  GlobalVariable *getGlobalVariable(StringRef Name,
                                    bool AllowInternal = false) const;

  /// getNamedGlobal - Return the global variable in the module with the
  /// specified name, of arbitrary type.  This method returns null if a global
  /// with the specified name is not found.
  GlobalVariable *getNamedGlobal(StringRef Name) const {
    return getGlobalVariable(Name, true);
  }

  /// getOrInsertGlobal - Look up the specified global in the module symbol
  /// table.
  ///   1. If it does not exist, add a declaration of the global and return it.
  ///   2. Else, the global exists but has the wrong type: return the function
  ///      with a constantexpr cast to the right type.
  ///   3. Finally, if the existing global is the correct declaration, return
  ///      the existing global.
  Constant *getOrInsertGlobal(StringRef Name, Type *Ty);

/// @}
/// @name Global Alias Accessors
/// @{

  /// getNamedAlias - Return the global alias in the module with the
  /// specified name, of arbitrary type.  This method returns null if a global
  /// with the specified name is not found.
  GlobalAlias *getNamedAlias(StringRef Name) const;

/// @}
/// @name Named Metadata Accessors
/// @{

  /// getNamedMetadata - Return the NamedMDNode in the module with the
  /// specified name. This method returns null if a NamedMDNode with the
  /// specified name is not found.
  NamedMDNode *getNamedMetadata(const Twine &Name) const;

  /// getOrInsertNamedMetadata - Return the named MDNode in the module
  /// with the specified name. This method returns a new NamedMDNode if a
  /// NamedMDNode with the specified name is not found.
  NamedMDNode *getOrInsertNamedMetadata(StringRef Name);

  /// eraseNamedMetadata - Remove the given NamedMDNode from this module
  /// and delete it.
  void eraseNamedMetadata(NamedMDNode *NMD);

/// @}
/// @name Module Flags Accessors
/// @{

  /// getModuleFlagsMetadata - Returns the module flags in the provided vector.
  void getModuleFlagsMetadata(SmallVectorImpl<ModuleFlagEntry> &Flags) const;

  /// getModuleFlagsMetadata - Returns the NamedMDNode in the module that
  /// represents module-level flags. This method returns null if there are no
  /// module-level flags.
  NamedMDNode *getModuleFlagsMetadata() const;

  /// getOrInsertModuleFlagsMetadata - Returns the NamedMDNode in the module
  /// that represents module-level flags. If module-level flags aren't found,
  /// it creates the named metadata that contains them.
  NamedMDNode *getOrInsertModuleFlagsMetadata();

  /// addModuleFlag - Add a module-level flag to the module-level flags
  /// metadata. It will create the module-level flags named metadata if it
  /// doesn't already exist.
  void addModuleFlag(ModFlagBehavior Behavior, StringRef Key, Value *Val);
  void addModuleFlag(ModFlagBehavior Behavior, StringRef Key, uint32_t Val);
  void addModuleFlag(MDNode *Node);

/// @}
/// @name Materialization
/// @{

  /// setMaterializer - Sets the GVMaterializer to GVM.  This module must not
  /// yet have a Materializer.  To reset the materializer for a module that
  /// already has one, call MaterializeAllPermanently first.  Destroying this
  /// module will destroy its materializer without materializing any more
  /// GlobalValues.  Without destroying the Module, there is no way to detach or
  /// destroy a materializer without materializing all the GVs it controls, to
  /// avoid leaving orphan unmaterialized GVs.
  void setMaterializer(GVMaterializer *GVM);
  /// getMaterializer - Retrieves the GVMaterializer, if any, for this Module.
  GVMaterializer *getMaterializer() const { return Materializer.get(); }

  /// isMaterializable - True if the definition of GV has yet to be materialized
  /// from the GVMaterializer.
  bool isMaterializable(const GlobalValue *GV) const;
  /// isDematerializable - Returns true if this GV was loaded from this Module's
  /// GVMaterializer and the GVMaterializer knows how to dematerialize the GV.
  bool isDematerializable(const GlobalValue *GV) const;

  /// Materialize - Make sure the GlobalValue is fully read.  If the module is
  /// corrupt, this returns true and fills in the optional string with
  /// information about the problem.  If successful, this returns false.
  bool Materialize(GlobalValue *GV, std::string *ErrInfo = 0);
  /// Dematerialize - If the GlobalValue is read in, and if the GVMaterializer
  /// supports it, release the memory for the function, and set it up to be
  /// materialized lazily.  If !isDematerializable(), this method is a noop.
  void Dematerialize(GlobalValue *GV);

  /// MaterializeAll - Make sure all GlobalValues in this Module are fully read.
  /// If the module is corrupt, this returns true and fills in the optional
  /// string with information about the problem.  If successful, this returns
  /// false.
  bool MaterializeAll(std::string *ErrInfo = 0);

  /// MaterializeAllPermanently - Make sure all GlobalValues in this Module are
  /// fully read and clear the Materializer.  If the module is corrupt, this
  /// returns true, fills in the optional string with information about the
  /// problem, and DOES NOT clear the old Materializer.  If successful, this
  /// returns false.
  bool MaterializeAllPermanently(std::string *ErrInfo = 0);

/// @}
/// @name Direct access to the globals list, functions list, and symbol table
/// @{

  /// Get the Module's list of global variables (constant).
  const GlobalListType   &getGlobalList() const       { return GlobalList; }
  /// Get the Module's list of global variables.
  GlobalListType         &getGlobalList()             { return GlobalList; }
  static iplist<GlobalVariable> Module::*getSublistAccess(GlobalVariable*) {
    return &Module::GlobalList;
  }
  /// Get the Module's list of functions (constant).
  const FunctionListType &getFunctionList() const     { return FunctionList; }
  /// Get the Module's list of functions.
  FunctionListType       &getFunctionList()           { return FunctionList; }
  static iplist<Function> Module::*getSublistAccess(Function*) {
    return &Module::FunctionList;
  }
  /// Get the Module's list of aliases (constant).
  const AliasListType    &getAliasList() const        { return AliasList; }
  /// Get the Module's list of aliases.
  AliasListType          &getAliasList()              { return AliasList; }
  static iplist<GlobalAlias> Module::*getSublistAccess(GlobalAlias*) {
    return &Module::AliasList;
  }
  /// Get the Module's list of named metadata (constant).
  const NamedMDListType  &getNamedMDList() const      { return NamedMDList; }
  /// Get the Module's list of named metadata.
  NamedMDListType        &getNamedMDList()            { return NamedMDList; }
  static ilist<NamedMDNode> Module::*getSublistAccess(NamedMDNode*) {
    return &Module::NamedMDList;
  }
  /// Get the symbol table of global variable and function identifiers
  const ValueSymbolTable &getValueSymbolTable() const { return *ValSymTab; }
  /// Get the Module's symbol table of global variable and function identifiers.
  ValueSymbolTable       &getValueSymbolTable()       { return *ValSymTab; }

/// @}
/// @name Global Variable Iteration
/// @{

  global_iterator       global_begin()       { return GlobalList.begin(); }
  const_global_iterator global_begin() const { return GlobalList.begin(); }
  global_iterator       global_end  ()       { return GlobalList.end(); }
  const_global_iterator global_end  () const { return GlobalList.end(); }
  bool                  global_empty() const { return GlobalList.empty(); }

/// @}
/// @name Function Iteration
/// @{

  iterator                begin()       { return FunctionList.begin(); }
  const_iterator          begin() const { return FunctionList.begin(); }
  iterator                end  ()       { return FunctionList.end();   }
  const_iterator          end  () const { return FunctionList.end();   }
  size_t                  size() const  { return FunctionList.size(); }
  bool                    empty() const { return FunctionList.empty(); }

/// @}
/// @name Dependent Library Iteration
/// @{

  /// @brief Get a constant iterator to beginning of dependent library list.
  inline lib_iterator lib_begin() const { return LibraryList.begin(); }
  /// @brief Get a constant iterator to end of dependent library list.
  inline lib_iterator lib_end()   const { return LibraryList.end();   }
  /// @brief Returns the number of items in the list of libraries.
  inline size_t       lib_size()  const { return LibraryList.size();  }
  /// @brief Add a library to the list of dependent libraries
  void addLibrary(StringRef Lib);
  /// @brief Remove a library from the list of dependent libraries
  void removeLibrary(StringRef Lib);
  /// @brief Get all the libraries
  inline const LibraryListType& getLibraries() const { return LibraryList; }

/// @}
/// @name Alias Iteration
/// @{

  alias_iterator       alias_begin()            { return AliasList.begin(); }
  const_alias_iterator alias_begin() const      { return AliasList.begin(); }
  alias_iterator       alias_end  ()            { return AliasList.end();   }
  const_alias_iterator alias_end  () const      { return AliasList.end();   }
  size_t               alias_size () const      { return AliasList.size();  }
  bool                 alias_empty() const      { return AliasList.empty(); }


/// @}
/// @name Named Metadata Iteration
/// @{

  named_metadata_iterator named_metadata_begin() { return NamedMDList.begin(); }
  const_named_metadata_iterator named_metadata_begin() const {
    return NamedMDList.begin();
  }

  named_metadata_iterator named_metadata_end() { return NamedMDList.end(); }
  const_named_metadata_iterator named_metadata_end() const {
    return NamedMDList.end();
  }

  size_t named_metadata_size() const { return NamedMDList.size();  }
  bool named_metadata_empty() const { return NamedMDList.empty(); }


/// @}
/// @name Utility functions for printing and dumping Module objects
/// @{

  /// Print the module to an output stream with an optional
  /// AssemblyAnnotationWriter.
  void print(raw_ostream &OS, AssemblyAnnotationWriter *AAW) const;

  /// Dump the module to stderr (for debugging).
  void dump() const;
  
  /// This function causes all the subinstructions to "let go" of all references
  /// that they are maintaining.  This allows one to 'delete' a whole class at
  /// a time, even though there may be circular references... first all
  /// references are dropped, and all use counts go to zero.  Then everything
  /// is delete'd for real.  Note that no operations are valid on an object
  /// that has "dropped all references", except operator delete.
  void dropAllReferences();
/// @}
};

/// An raw_ostream inserter for modules.
inline raw_ostream &operator<<(raw_ostream &O, const Module &M) {
  M.print(O, 0);
  return O;
}

} // End llvm namespace

#endif
