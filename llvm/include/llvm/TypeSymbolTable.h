//===-- llvm/TypeSymbolTable.h - Implement a Type Symtab --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the name/type symbol table for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPE_SYMBOL_TABLE_H
#define LLVM_TYPE_SYMBOL_TABLE_H

#include "llvm/Type.h"
#include <map>

namespace llvm {

/// This class provides a symbol table of name/type pairs with operations to
/// support constructing, searching and iterating over the symbol table. The
/// class derives from AbstractTypeUser so that the contents of the symbol
/// table can be updated when abstract types become concrete.
class TypeSymbolTable : public AbstractTypeUser {

/// @name Types
/// @{
public:

  /// @brief A mapping of names to types.
  typedef std::map<const std::string, const Type*> TypeMap;

  /// @brief An iterator over the TypeMap.
  typedef TypeMap::iterator iterator;

  /// @brief A const_iterator over the TypeMap.
  typedef TypeMap::const_iterator const_iterator;

/// @}
/// @name Constructors
/// @{
public:

  TypeSymbolTable():LastUnique(0) {}
  ~TypeSymbolTable();

/// @}
/// @name Accessors
/// @{
public:

  /// Generates a unique name for a type based on the \p BaseName by
  /// incrementing an integer and appending it to the name, if necessary
  /// @returns the unique name
  /// @brief Get a unique name for a type
  std::string getUniqueName(const std::string &BaseName) const;

  /// This method finds the type with the given \p name in the type map
  /// and returns it.
  /// @returns null if the name is not found, otherwise the Type
  /// associated with the \p name.
  /// @brief Lookup a type by name.
  Type* lookup(const std::string& name) const;

  /// @returns true iff the symbol table is empty.
  /// @brief Determine if the symbol table is empty
  inline bool empty() const { return tmap.empty(); }

  /// @returns the size of the symbol table
  /// @brief The number of name/type pairs is returned.
  inline unsigned size() const { return unsigned(tmap.size()); }

  /// This function can be used from the debugger to display the
  /// content of the symbol table while debugging.
  /// @brief Print out symbol table on stderr
  void dump() const;

/// @}
/// @name Iteration
/// @{
public:
  /// Get an iterator to the start of the symbol table
  inline iterator begin() { return tmap.begin(); }

  /// @brief Get a const_iterator to the start of the symbol table
  inline const_iterator begin() const { return tmap.begin(); }

  /// Get an iterator to the end of the symbol talbe. 
  inline iterator end() { return tmap.end(); }

  /// Get a const_iterator to the end of the symbol table.
  inline const_iterator end() const { return tmap.end(); }

/// @}
/// @name Mutators
/// @{
public:

  /// Inserts a type into the symbol table with the specified name. There can be
  /// a many-to-one mapping between names and types. This method allows a type
  /// with an existing entry in the symbol table to get a new name.
  /// @brief Insert a type under a new name.
  void insert(const std::string &Name, const Type *Typ);

  /// Remove a type at the specified position in the symbol table.
  /// @returns the removed Type.
  /// @returns the Type that was erased from the symbol table.
  Type* remove(iterator TI);

/// @}
/// @name AbstractTypeUser Methods
/// @{
private:
  /// This function is called when one of the types in the type plane
  /// is refined.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  /// This function markes a type as being concrete (defined).
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

/// @}
/// @name Internal Data
/// @{
private:
  TypeMap tmap; ///< This is the mapping of names to types.
  mutable uint32_t LastUnique; ///< Counter for tracking unique names

/// @}

};

} // End llvm namespace

#endif
