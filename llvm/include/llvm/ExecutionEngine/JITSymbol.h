//===----------- JITSymbol.h - JIT symbol abstraction -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Abstraction for target process addresses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITSYMBOL_H
#define LLVM_EXECUTIONENGINE_JITSYMBOL_H

#include "llvm/ExecutionEngine/JITSymbolFlags.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <functional>

namespace llvm {

/// @brief Represents an address in the target process's address space.
typedef uint64_t JITTargetAddress;

/// @brief Represents a symbol that has been evaluated to an address already.
class JITEvaluatedSymbol : public JITSymbolBase {
public:

  /// @brief Create a 'null' symbol.
  JITEvaluatedSymbol(std::nullptr_t)
      : JITSymbolBase(JITSymbolFlags::None), Address(0) {}

  /// @brief Create a symbol for the given address and flags.
  JITEvaluatedSymbol(JITTargetAddress Address, JITSymbolFlags Flags)
      : JITSymbolBase(Flags), Address(Address) {}

  /// @brief An evaluated symbol converts to 'true' if its address is non-zero.
  explicit operator bool() const { return Address != 0; }

  /// @brief Return the address of this symbol.
  JITTargetAddress getAddress() const { return Address; }

private:
  JITTargetAddress Address;
};

/// @brief Represents a symbol in the JIT.
class JITSymbol : public JITSymbolBase {
public:

  typedef std::function<JITTargetAddress()> GetAddressFtor;

  /// @brief Create a 'null' symbol that represents failure to find a symbol
  ///        definition.
  JITSymbol(std::nullptr_t)
      : JITSymbolBase(JITSymbolFlags::None), CachedAddr(0) {}

  /// @brief Create a symbol for a definition with a known address.
  JITSymbol(JITTargetAddress Addr, JITSymbolFlags Flags)
    : JITSymbolBase(Flags), CachedAddr(Addr) {}

  /// @brief Construct a JITSymbol from a JITEvaluatedSymbol.
  JITSymbol(JITEvaluatedSymbol Sym)
    : JITSymbolBase(Sym.getFlags()), CachedAddr(Sym.getAddress()) {}

  /// @brief Create a symbol for a definition that doesn't have a known address
  ///        yet.
  /// @param GetAddress A functor to materialize a definition (fixing the
  ///        address) on demand.
  ///
  ///   This constructor allows a JIT layer to provide a reference to a symbol
  /// definition without actually materializing the definition up front. The
  /// user can materialize the definition at any time by calling the getAddress
  /// method.
  JITSymbol(GetAddressFtor GetAddress, JITSymbolFlags Flags)
      : JITSymbolBase(Flags), GetAddress(std::move(GetAddress)), CachedAddr(0) {}

  /// @brief Returns true if the symbol exists, false otherwise.
  explicit operator bool() const { return CachedAddr || GetAddress; }

  /// @brief Get the address of the symbol in the target address space. Returns
  ///        '0' if the symbol does not exist.
  JITTargetAddress getAddress() {
    if (GetAddress) {
      CachedAddr = GetAddress();
      assert(CachedAddr && "Symbol could not be materialized.");
      GetAddress = nullptr;
    }
    return CachedAddr;
  }

private:
  GetAddressFtor GetAddress;
  JITTargetAddress CachedAddr;
};

/// \brief Symbol resolution.
class JITSymbolResolver {
public:
  virtual ~JITSymbolResolver() {}

  /// This method returns the address of the specified symbol if it exists
  /// within the logical dynamic library represented by this JITSymbolResolver.
  /// Unlike findSymbol, queries through this interface should return addresses
  /// for hidden symbols.
  ///
  /// This is of particular importance for the Orc JIT APIs, which support lazy
  /// compilation by breaking up modules: Each of those broken out modules
  /// must be able to resolve hidden symbols provided by the others. Clients
  /// writing memory managers for MCJIT can usually ignore this method.
  ///
  /// This method will be queried by RuntimeDyld when checking for previous
  /// definitions of common symbols.
  virtual JITSymbol findSymbolInLogicalDylib(const std::string &Name) = 0;

  /// This method returns the address of the specified function or variable.
  /// It is used to resolve symbols during module linking.
  ///
  /// If the returned symbol's address is equal to ~0ULL then RuntimeDyld will
  /// skip all relocations for that symbol, and the client will be responsible
  /// for handling them manually.
  virtual JITSymbol findSymbol(const std::string &Name) = 0;

private:
  virtual void anchor();
};

} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_JITSYMBOL_H
