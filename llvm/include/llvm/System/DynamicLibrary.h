//===-- llvm/System/DynamicLibrary.h - Portable Dynamic Library -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the sys::DynamicLibrary class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_DYNAMIC_LIBRARY_H
#define LLVM_SYSTEM_DYNAMIC_LIBRARY_H

#include "llvm/System/Path.h"
#include <string>

namespace llvm {
namespace sys {

  /// This class provides a portable interface to dynamic libraries which also
  /// might be known as shared libraries, shared objects, dynamic shared
  /// objects, or dynamic link libraries. Regardless of the terminology or the
  /// operating system interface, this class provides a portable interface that
  /// allows dynamic libraries to be loaded and and searched for externally
  /// defined symbols. This is typically used to provide "plug-in" support.
  /// It also allows for symbols to be defined which don't live in any library,
  /// but rather the main program itself, useful on Windows where the main
  /// executable cannot be searched.
  class DynamicLibrary {
    /// @name Constructors
    /// @{
    public:
      /// Construct a DynamicLibrary that represents the currently executing
      /// program. The program must have been linked with -export-dynamic or
      /// -dlopen self for this to work. 
      /// @throws std::string indicating why the program couldn't be opened.
      /// @brief Open program as dynamic library.
      DynamicLibrary();

      /// After destruction, the symbols of the library will no longer be
      /// available to the program. 
      /// @brief Closes the DynamicLibrary
      ~DynamicLibrary();

    /// @}
    /// @name Functions
    /// @{
    public:
      /// This function allows a library to be loaded without instantiating a
      /// DynamicLibrary object. Consequently, it is marked as being permanent
      /// and will only be unloaded when the program terminates.  This returns
      /// false on success or returns true and fills in *ErrMsg on failure.
      /// @brief Open a dynamic library permanently.
      static bool LoadLibraryPermanently(const char* filename,
                                         std::string *ErrMsg = 0);

      /// This function will search through all previously loaded dynamic
      /// libraries for the symbol \p symbolName. If it is found, the addressof
      /// that symbol is returned. If not, null is returned. Note that this will
      /// search permanently loaded libraries (LoadLibraryPermanently) as well
      /// as ephemerally loaded libraries (constructors).
      /// @throws std::string on error.
      /// @brief Search through libraries for address of a symbol
      static void* SearchForAddressOfSymbol(const char* symbolName);

      /// @brief Convenience function for C++ophiles.
      static void* SearchForAddressOfSymbol(const std::string& symbolName) {
        return SearchForAddressOfSymbol(symbolName.c_str());
      }

      /// This functions permanently adds the symbol \p symbolName with the
      /// value \p symbolValue.  These symbols are searched before any
      /// libraries.
      /// @brief Add searchable symbol/value pair.
      static void AddSymbol(const char* symbolName, void *symbolValue);

      /// @brief Convenience function for C++ophiles.
      static void AddSymbol(const std::string& symbolName, void *symbolValue) {
        AddSymbol(symbolName.c_str(), symbolValue);
      }

    /// @}
    /// @name Implementation
    /// @{
    protected:
      void* handle;  // Opaque handle for information about the library
      DynamicLibrary(const DynamicLibrary&); ///< Do not implement
      DynamicLibrary& operator=(const DynamicLibrary&); ///< Do not implement
    /// @}
  };

} // End sys namespace
} // End llvm namespace

#endif // LLVM_SYSTEM_DYNAMIC_LIBRARY_H
