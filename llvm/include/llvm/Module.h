//===-- llvm/Module.h - C++ class to represent a VM module ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file This file contains the declarations for the Module class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULE_H
#define LLVM_MODULE_H

#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class GlobalValueRefMap;   // Used by ConstantVals.cpp
class FunctionType;

template<> struct ilist_traits<Function>
  : public SymbolTableListTraits<Function, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  static Function *createSentinel();
  static void destroySentinel(Function *F) { delete F; }
  static iplist<Function> &getList(Module *M);
  static inline ValueSymbolTable *getSymTab(Module *M);
  static int getListOffset();
};
template<> struct ilist_traits<GlobalVariable>
  : public SymbolTableListTraits<GlobalVariable, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  static GlobalVariable *createSentinel();
  static void destroySentinel(GlobalVariable *GV) { delete GV; }
  static iplist<GlobalVariable> &getList(Module *M);
  static inline ValueSymbolTable *getSymTab(Module *M);
  static int getListOffset();
};
template<> struct ilist_traits<GlobalAlias>
  : public SymbolTableListTraits<GlobalAlias, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  static GlobalAlias *createSentinel();
  static void destroySentinel(GlobalAlias *GA) { delete GA; }
  static iplist<GlobalAlias> &getList(Module *M);
  static inline ValueSymbolTable *getSymTab(Module *M);
  static int getListOffset();
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

  /// The type for the list of dependent libraries.
  typedef std::vector<std::string> LibraryListType;

  /// The Global Variable iterator.
  typedef GlobalListType::iterator                     global_iterator;
  /// The Global Variable constant iterator.
  typedef GlobalListType::const_iterator         const_global_iterator;

  /// The Function iterators.
  typedef FunctionListType::iterator                          iterator;
  /// The Function constant iterator
  typedef FunctionListType::const_iterator              const_iterator;

  /// The Global Alias iterators.
  typedef AliasListType::iterator                       alias_iterator;
  /// The Global Alias constant iterator
  typedef AliasListType::const_iterator           const_alias_iterator;

  /// The Library list iterator.
  typedef LibraryListType::const_iterator lib_iterator;

  /// An enumeration for describing the endianess of the target machine.
  enum Endianness  { AnyEndianness, LittleEndian, BigEndian };

  /// An enumeration for describing the size of a pointer on the target machine.
  enum PointerSize { AnyPointerSize, Pointer32, Pointer64 };

/// @}
/// @name Member Variables
/// @{
private:
  GlobalListType GlobalList;     ///< The Global Variables in the module
  FunctionListType FunctionList; ///< The Functions in the module
  AliasListType AliasList;       ///< The Aliases in the module
  LibraryListType LibraryList;   ///< The Libraries needed by the module
  std::string GlobalScopeAsm;    ///< Inline Asm at global scope.
  ValueSymbolTable *ValSymTab;   ///< Symbol table for values
  TypeSymbolTable *TypeSymTab;   ///< Symbol table for types
  std::string ModuleID;          ///< Human readable identifier for the module
  std::string TargetTriple;      ///< Platform target triple Module compiled on
  std::string DataLayout;        ///< Target data description

  friend class Constant;

/// @}
/// @name Constructors
/// @{
public:
  /// The Module constructor. Note that there is no default constructor. You
  /// must provide a name for the module upon construction.
  explicit Module(const std::string &ModuleID);
  /// The module destructor. This will dropAllReferences.
  ~Module();

/// @}
/// @name Module Level Accessors
/// @{
public:
  /// Get the module identifier which is, essentially, the name of the module.
  /// @returns the module identifier as a string
  const std::string &getModuleIdentifier() const { return ModuleID; }

  /// Get the data layout string for the module's target platform.  This encodes
  /// the type sizes and alignments expected by this module.
  /// @returns the data layout as a string
  const std::string& getDataLayout() const { return DataLayout; }

  /// Get the target triple which is a string describing the target host.
  /// @returns a string containing the target triple.
  const std::string &getTargetTriple() const { return TargetTriple; }

  /// Get the target endian information.
  /// @returns Endianess - an enumeration for the endianess of the target
  Endianness getEndianness() const;

  /// Get the target pointer size.
  /// @returns PointerSize - an enumeration for the size of the target's pointer
  PointerSize getPointerSize() const;

  /// Get any module-scope inline assembly blocks.
  /// @returns a string containing the module-scope inline assembly blocks.
  const std::string &getModuleInlineAsm() const { return GlobalScopeAsm; }
/// @}
/// @name Module Level Mutators
/// @{
public:

  /// Set the module identifier.
  void setModuleIdentifier(const std::string &ID) { ModuleID = ID; }

  /// Set the data layout
  void setDataLayout(const std::string& DL) { DataLayout = DL; }

  /// Set the target triple.
  void setTargetTriple(const std::string &T) { TargetTriple = T; }

  /// Set the module-scope inline assembly blocks.
  void setModuleInlineAsm(const std::string &Asm) { GlobalScopeAsm = Asm; }

  /// Append to the module-scope inline assembly blocks, automatically
  /// appending a newline to the end.
  void appendModuleInlineAsm(const std::string &Asm) {
    GlobalScopeAsm += Asm;
    GlobalScopeAsm += '\n';
  }

/// @}
/// @name Function Accessors
/// @{
public:
  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  Four possibilities:
  ///   1. If it does not exist, add a prototype for the function and return it.
  ///   2. If it exists, and has a local linkage, the existing function is
  ///      renamed and a new one is inserted.
  ///   3. Otherwise, if the existing function has the correct prototype, return
  ///      the existing function.
  ///   4. Finally, the function exists but has the wrong prototype: return the
  ///      function with a constantexpr cast to the right prototype.
  Constant *getOrInsertFunction(const std::string &Name, const FunctionType *T,
                                AttrListPtr AttributeList);

  Constant *getOrInsertFunction(const std::string &Name, const FunctionType *T);

  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  If it does not exist, add a prototype for the function and return
  /// it.  This function guarantees to return a constant of pointer to the
  /// specified function type or a ConstantExpr BitCast of that type if the
  /// named function has a different type.  This version of the method takes a
  /// null terminated list of function arguments, which makes it easier for
  /// clients to use.
  Constant *getOrInsertFunction(const std::string &Name,
                                AttrListPtr AttributeList,
                                const Type *RetTy, ...)  END_WITH_NULL;

  Constant *getOrInsertFunction(const std::string &Name, const Type *RetTy, ...)
    END_WITH_NULL;

  /// getFunction - Look up the specified function in the module symbol table.
  /// If it does not exist, return null.
  Function *getFunction(const std::string &Name) const;
  Function *getFunction(const char *Name) const;

/// @}
/// @name Global Variable Accessors
/// @{
public:
  /// getGlobalVariable - Look up the specified global variable in the module
  /// symbol table.  If it does not exist, return null. If AllowInternal is set
  /// to true, this function will return types that have InternalLinkage. By
  /// default, these types are not returned.
  GlobalVariable *getGlobalVariable(const std::string &Name,
                                    bool AllowInternal = false) const;

  /// getNamedGlobal - Return the first global variable in the module with the
  /// specified name, of arbitrary type.  This method returns null if a global
  /// with the specified name is not found.
  GlobalVariable *getNamedGlobal(const std::string &Name) const {
    return getGlobalVariable(Name, true);
  }

  /// getOrInsertGlobal - Look up the specified global in the module symbol
  /// table.
  ///   1. If it does not exist, add a declaration of the global and return it.
  ///   2. Else, the global exists but has the wrong type: return the function
  ///      with a constantexpr cast to the right type.
  ///   3. Finally, if the existing global is the correct delclaration, return
  ///      the existing global.
  Constant *getOrInsertGlobal(const std::string &Name, const Type *Ty);

/// @}
/// @name Global Alias Accessors
/// @{
public:
  /// getNamedAlias - Return the first global alias in the module with the
  /// specified name, of arbitrary type.  This method returns null if a global
  /// with the specified name is not found.
  GlobalAlias *getNamedAlias(const std::string &Name) const;

/// @}
/// @name Type Accessors
/// @{
public:
  /// addTypeName - Insert an entry in the symbol table mapping Str to Type.  If
  /// there is already an entry for this name, true is returned and the symbol
  /// table is not modified.
  bool addTypeName(const std::string &Name, const Type *Ty);

  /// getTypeName - If there is at least one entry in the symbol table for the
  /// specified type, return it.
  std::string getTypeName(const Type *Ty) const;

  /// getTypeByName - Return the type with the specified name in this module, or
  /// null if there is none by that name.
  const Type *getTypeByName(const std::string &Name) const;

/// @}
/// @name Direct access to the globals list, functions list, and symbol table
/// @{
public:
  /// Get the Module's list of global variables (constant).
  const GlobalListType   &getGlobalList() const       { return GlobalList; }
  /// Get the Module's list of global variables.
  GlobalListType         &getGlobalList()             { return GlobalList; }
  /// Get the Module's list of functions (constant).
  const FunctionListType &getFunctionList() const     { return FunctionList; }
  /// Get the Module's list of functions.
  FunctionListType       &getFunctionList()           { return FunctionList; }
  /// Get the Module's list of aliases (constant).
  const AliasListType    &getAliasList() const        { return AliasList; }
  /// Get the Module's list of aliases.
  AliasListType          &getAliasList()              { return AliasList; }
  /// Get the symbol table of global variable and function identifiers
  const ValueSymbolTable &getValueSymbolTable() const { return *ValSymTab; }
  /// Get the Module's symbol table of global variable and function identifiers.
  ValueSymbolTable       &getValueSymbolTable()       { return *ValSymTab; }
  /// Get the symbol table of types
  const TypeSymbolTable  &getTypeSymbolTable() const  { return *TypeSymTab; }
  /// Get the Module's symbol table of types
  TypeSymbolTable        &getTypeSymbolTable()        { return *TypeSymTab; }

/// @}
/// @name Global Variable Iteration
/// @{
public:
  /// Get an iterator to the first global variable
  global_iterator       global_begin()       { return GlobalList.begin(); }
  /// Get a constant iterator to the first global variable
  const_global_iterator global_begin() const { return GlobalList.begin(); }
  /// Get an iterator to the last global variable
  global_iterator       global_end  ()       { return GlobalList.end(); }
  /// Get a constant iterator to the last global variable
  const_global_iterator global_end  () const { return GlobalList.end(); }
  /// Determine if the list of globals is empty.
  bool                  global_empty() const { return GlobalList.empty(); }

/// @}
/// @name Function Iteration
/// @{
public:
  /// Get an iterator to the first function.
  iterator                begin()       { return FunctionList.begin(); }
  /// Get a constant iterator to the first function.
  const_iterator          begin() const { return FunctionList.begin(); }
  /// Get an iterator to the last function.
  iterator                end  ()       { return FunctionList.end();   }
  /// Get a constant iterator to the last function.
  const_iterator          end  () const { return FunctionList.end();   }
  /// Determine how many functions are in the Module's list of functions.
  size_t                  size() const  { return FunctionList.size(); }
  /// Determine if the list of functions is empty.
  bool                    empty() const { return FunctionList.empty(); }

/// @}
/// @name Dependent Library Iteration
/// @{
public:
  /// @brief Get a constant iterator to beginning of dependent library list.
  inline lib_iterator lib_begin() const { return LibraryList.begin(); }
  /// @brief Get a constant iterator to end of dependent library list.
  inline lib_iterator lib_end()   const { return LibraryList.end();   }
  /// @brief Returns the number of items in the list of libraries.
  inline size_t       lib_size()  const { return LibraryList.size();  }
  /// @brief Add a library to the list of dependent libraries
  void addLibrary(const std::string& Lib);
  /// @brief Remove a library from the list of dependent libraries
  void removeLibrary(const std::string& Lib);
  /// @brief Get all the libraries
  inline const LibraryListType& getLibraries() const { return LibraryList; }

/// @}
/// @name Alias Iteration
/// @{
public:
  /// Get an iterator to the first alias.
  alias_iterator       alias_begin()            { return AliasList.begin(); }
  /// Get a constant iterator to the first alias.
  const_alias_iterator alias_begin() const      { return AliasList.begin(); }
  /// Get an iterator to the last alias.
  alias_iterator       alias_end  ()            { return AliasList.end();   }
  /// Get a constant iterator to the last alias.
  const_alias_iterator alias_end  () const      { return AliasList.end();   }
  /// Determine how many functions are in the Module's list of aliases.
  size_t               alias_size () const      { return AliasList.size();  }
  /// Determine if the list of aliases is empty.
  bool                 alias_empty() const      { return AliasList.empty(); }

/// @}
/// @name Utility functions for printing and dumping Module objects
/// @{
public:
  /// Print the module to an output stream with AssemblyAnnotationWriter.
  void print(raw_ostream &OS, AssemblyAnnotationWriter *AAW) const;
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;
  
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

  static unsigned getFunctionListOffset() {
    Module *Obj = 0;
    return unsigned(reinterpret_cast<uintptr_t>(&Obj->FunctionList));
  }
  static unsigned getGlobalVariableListOffset() {
    Module *Obj = 0;
    return unsigned(reinterpret_cast<uintptr_t>(&Obj->GlobalList));
  }
  static unsigned getAliasListOffset() {
    Module *Obj = 0;
    return unsigned(reinterpret_cast<uintptr_t>(&Obj->AliasList));
  }
};

/// An iostream inserter for modules.
inline std::ostream &operator<<(std::ostream &O, const Module &M) {
  M.print(O, 0);
  return O;
}
inline raw_ostream &operator<<(raw_ostream &O, const Module &M) {
  M.print(O, 0);
  return O;
}
  

inline ValueSymbolTable *
ilist_traits<Function>::getSymTab(Module *M) {
  return M ? &M->getValueSymbolTable() : 0;
}

inline ValueSymbolTable *
ilist_traits<GlobalVariable>::getSymTab(Module *M) {
  return M ? &M->getValueSymbolTable() : 0;
}

inline ValueSymbolTable *
ilist_traits<GlobalAlias>::getSymTab(Module *M) {
  return M ? &M->getValueSymbolTable() : 0;
}

inline int
ilist_traits<Function>::getListOffset() {
  return Module::getFunctionListOffset();
}

inline int
ilist_traits<GlobalVariable>::getListOffset() {
  return Module::getGlobalVariableListOffset();
}

inline int
ilist_traits<GlobalAlias>::getListOffset() {
  return Module::getAliasListOffset();
}

} // End llvm namespace

#endif
