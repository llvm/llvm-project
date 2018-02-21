//===- JITSymbol.h - JIT symbol abstraction ---------------------*- C++ -*-===//
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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {

class GlobalValue;

namespace object {

class BasicSymbolRef;

} // end namespace object

/// @brief Represents an address in the target process's address space.
using JITTargetAddress = uint64_t;

/// @brief Flags for symbols in the JIT.
class JITSymbolFlags {
public:
  using UnderlyingType = uint8_t;
  using TargetFlagsType = uint64_t;

  enum FlagNames : UnderlyingType {
    None = 0,
    HasError = 1U << 0,
    Weak = 1U << 1,
    Common = 1U << 2,
    Absolute = 1U << 3,
    Exported = 1U << 4,
    NotMaterialized = 1U << 5
  };

  static JITSymbolFlags stripTransientFlags(JITSymbolFlags Orig) {
    return static_cast<FlagNames>(Orig.Flags & ~NotMaterialized);
  }

  /// @brief Default-construct a JITSymbolFlags instance.
  JITSymbolFlags() = default;

  /// @brief Construct a JITSymbolFlags instance from the given flags.
  JITSymbolFlags(FlagNames Flags) : Flags(Flags) {}

  /// @brief Construct a JITSymbolFlags instance from the given flags and target
  ///        flags.
  JITSymbolFlags(FlagNames Flags, TargetFlagsType TargetFlags)
    : Flags(Flags), TargetFlags(TargetFlags) {}

  /// @brief Return true if there was an error retrieving this symbol.
  bool hasError() const {
    return (Flags & HasError) == HasError;
  }

  /// @brief Returns true if this symbol has been fully materialized (i.e. is
  ///        callable).
  bool isMaterialized() const { return !(Flags & NotMaterialized); }

  /// @brief Returns true if the Weak flag is set.
  bool isWeak() const {
    return (Flags & Weak) == Weak;
  }

  /// @brief Returns true if the Common flag is set.
  bool isCommon() const {
    return (Flags & Common) == Common;
  }

  /// @brief Returns true if the symbol isn't weak or common.
  bool isStrong() const {
    return !isWeak() && !isCommon();
  }

  /// @brief Returns true if the Exported flag is set.
  bool isExported() const {
    return (Flags & Exported) == Exported;
  }

  /// @brief Implicitly convert to the underlying flags type.
  operator UnderlyingType&() { return Flags; }

  /// @brief Implicitly convert to the underlying flags type.
  operator const UnderlyingType&() const { return Flags; }

  /// @brief Return a reference to the target-specific flags.
  TargetFlagsType& getTargetFlags() { return TargetFlags; }

  /// @brief Return a reference to the target-specific flags.
  const TargetFlagsType& getTargetFlags() const { return TargetFlags; }

  /// Construct a JITSymbolFlags value based on the flags of the given global
  /// value.
  static JITSymbolFlags fromGlobalValue(const GlobalValue &GV);

  /// Construct a JITSymbolFlags value based on the flags of the given libobject
  /// symbol.
  static JITSymbolFlags fromObjectSymbol(const object::BasicSymbolRef &Symbol);

private:
  UnderlyingType Flags = None;
  TargetFlagsType TargetFlags = 0;
};

/// @brief ARM-specific JIT symbol flags.
/// FIXME: This should be moved into a target-specific header.
class ARMJITSymbolFlags {
public:
  ARMJITSymbolFlags() = default;

  enum FlagNames {
    None = 0,
    Thumb = 1 << 0
  };

  operator JITSymbolFlags::TargetFlagsType&() { return Flags; }

  static ARMJITSymbolFlags fromObjectSymbol(
                                           const object::BasicSymbolRef &Symbol);
private:
  JITSymbolFlags::TargetFlagsType Flags = 0;
};

/// @brief Represents a symbol that has been evaluated to an address already.
class JITEvaluatedSymbol {
public:
  JITEvaluatedSymbol() = default;

  /// @brief Create a 'null' symbol.
  JITEvaluatedSymbol(std::nullptr_t) {}

  /// @brief Create a symbol for the given address and flags.
  JITEvaluatedSymbol(JITTargetAddress Address, JITSymbolFlags Flags)
      : Address(Address), Flags(Flags) {}

  /// @brief An evaluated symbol converts to 'true' if its address is non-zero.
  explicit operator bool() const { return Address != 0; }

  /// @brief Return the address of this symbol.
  JITTargetAddress getAddress() const { return Address; }

  /// @brief Return the flags for this symbol.
  JITSymbolFlags getFlags() const { return Flags; }

private:
  JITTargetAddress Address = 0;
  JITSymbolFlags Flags;
};

/// @brief Represents a symbol in the JIT.
class JITSymbol {
public:
  using GetAddressFtor = std::function<Expected<JITTargetAddress>()>;

  /// @brief Create a 'null' symbol, used to represent a "symbol not found"
  ///        result from a successful (non-erroneous) lookup.
  JITSymbol(std::nullptr_t)
      : CachedAddr(0) {}

  /// @brief Create a JITSymbol representing an error in the symbol lookup
  ///        process (e.g. a network failure during a remote lookup).
  JITSymbol(Error Err)
    : Err(std::move(Err)), Flags(JITSymbolFlags::HasError) {}

  /// @brief Create a symbol for a definition with a known address.
  JITSymbol(JITTargetAddress Addr, JITSymbolFlags Flags)
      : CachedAddr(Addr), Flags(Flags) {}

  /// @brief Construct a JITSymbol from a JITEvaluatedSymbol.
  JITSymbol(JITEvaluatedSymbol Sym)
      : CachedAddr(Sym.getAddress()), Flags(Sym.getFlags()) {}

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
      : GetAddress(std::move(GetAddress)), CachedAddr(0), Flags(Flags) {}

  JITSymbol(const JITSymbol&) = delete;
  JITSymbol& operator=(const JITSymbol&) = delete;

  JITSymbol(JITSymbol &&Other)
    : GetAddress(std::move(Other.GetAddress)), Flags(std::move(Other.Flags)) {
    if (Flags.hasError())
      Err = std::move(Other.Err);
    else
      CachedAddr = std::move(Other.CachedAddr);
  }

  JITSymbol& operator=(JITSymbol &&Other) {
    GetAddress = std::move(Other.GetAddress);
    Flags = std::move(Other.Flags);
    if (Flags.hasError())
      Err = std::move(Other.Err);
    else
      CachedAddr = std::move(Other.CachedAddr);
    return *this;
  }

  ~JITSymbol() {
    if (Flags.hasError())
      Err.~Error();
    else
      CachedAddr.~JITTargetAddress();
  }

  /// @brief Returns true if the symbol exists, false otherwise.
  explicit operator bool() const {
    return !Flags.hasError() && (CachedAddr || GetAddress);
  }

  /// @brief Move the error field value out of this JITSymbol.
  Error takeError() {
    if (Flags.hasError())
      return std::move(Err);
    return Error::success();
  }

  /// @brief Get the address of the symbol in the target address space. Returns
  ///        '0' if the symbol does not exist.
  Expected<JITTargetAddress> getAddress() {
    assert(!Flags.hasError() && "getAddress called on error value");
    if (GetAddress) {
      if (auto CachedAddrOrErr = GetAddress()) {
        GetAddress = nullptr;
        CachedAddr = *CachedAddrOrErr;
        assert(CachedAddr && "Symbol could not be materialized.");
      } else
        return CachedAddrOrErr.takeError();
    }
    return CachedAddr;
  }

  JITSymbolFlags getFlags() const { return Flags; }

private:
  GetAddressFtor GetAddress;
  union {
    JITTargetAddress CachedAddr;
    Error Err;
  };
  JITSymbolFlags Flags;
};

/// @brief Symbol resolution interface.
///
/// Allows symbol flags and addresses to be looked up by name.
/// Symbol queries are done in bulk (i.e. you request resolution of a set of
/// symbols, rather than a single one) to reduce IPC overhead in the case of
/// remote JITing, and expose opportunities for parallel compilation.
class JITSymbolResolver {
public:
  using LookupSet = std::set<StringRef>;
  using LookupResult = std::map<StringRef, JITEvaluatedSymbol>;
  using LookupFlagsResult = std::map<StringRef, JITSymbolFlags>;

  virtual ~JITSymbolResolver() = default;

  /// @brief Returns the fully resolved address and flags for each of the given
  ///        symbols.
  ///
  /// This method will return an error if any of the given symbols can not be
  /// resolved, or if the resolution process itself triggers an error.
  virtual Expected<LookupResult> lookup(const LookupSet &Symbols) = 0;

  /// @brief Returns the symbol flags for each of the given symbols.
  ///
  /// This method does NOT return an error if any of the given symbols is
  /// missing. Instead, that symbol will be left out of the result map.
  virtual Expected<LookupFlagsResult> lookupFlags(const LookupSet &Symbols) = 0;

private:
  virtual void anchor();
};

/// \brief Legacy symbol resolution interface.
class LegacyJITSymbolResolver : public JITSymbolResolver {
public:
  /// @brief Performs lookup by, for each symbol, first calling
  ///        findSymbolInLogicalDylib and if that fails calling
  ///        findSymbol.
  Expected<LookupResult> lookup(const LookupSet &Symbols) final;

  /// @brief Performs flags lookup by calling findSymbolInLogicalDylib and
  ///        returning the flags value for that symbol.
  Expected<LookupFlagsResult> lookupFlags(const LookupSet &Symbols) final;

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

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITSYMBOL_H
