//===- SymCache.h - Declare Symbol Cache ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a Symbol Cache.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_SYMCACHE_H
#define MLIR_SUPPORT_SYMCACHE_H

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"

namespace mlir {

/// Base symbol cache class to allow for cache lookup through a pointer to some
/// abstract cache. A symbol cache stores lookup tables to make manipulating and
/// working with the IR more efficient.
class SymbolCacheBase {
public:
  virtual ~SymbolCacheBase();

  /// Defines 'op' as associated with the 'symbol' in the cache.
  virtual void addDefinition(mlir::Attribute symbol, mlir::Operation *op) = 0;

  /// Adds the symbol-defining 'op' to the cache.
  void addSymbol(mlir::SymbolOpInterface op) {
    addDefinition(op.getNameAttr(), op);
  }

  /// Populate the symbol cache with all symbol-defining operations within the
  /// 'top' operation.
  void addDefinitions(mlir::Operation *top);

  /// Lookup a definition for 'symbol' in the cache.
  virtual mlir::Operation *getDefinition(mlir::Attribute symbol) const = 0;

  /// Lookup a definition for 'symbol' in the cache.
  mlir::Operation *getDefinition(mlir::FlatSymbolRefAttr symbol) const {
    return getDefinition(symbol.getAttr());
  }

  /// Iterator support through a pointer to some abstract cache.
  /// The implementing cache must provide an iterator that carries values on the
  /// form of <mlir::Attribute, mlir::Operation*>.
  using CacheItem = std::pair<mlir::Attribute, mlir::Operation *>;
  struct CacheIteratorImpl {
    virtual ~CacheIteratorImpl() {}
    virtual void operator++() = 0;
    virtual CacheItem operator*() = 0;
    virtual bool operator==(CacheIteratorImpl *other) = 0;
  };

  struct Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          CacheItem> {
    Iterator(std::unique_ptr<CacheIteratorImpl> &&impl)
        : impl(std::move(impl)) {}
    CacheItem operator*() const { return **impl; }
    using llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                     CacheItem>::operator++;
    bool operator==(const Iterator &other) const {
      return *impl == other.impl.get();
    }
    void operator++() { impl->operator++(); }

  private:
    std::unique_ptr<CacheIteratorImpl> impl;
  };
  virtual Iterator begin() = 0;
  virtual Iterator end() = 0;
};

/// Default symbol cache implementation; stores associations between names
/// (StringAttr's) to mlir::Operation's.
/// Adding/getting definitions from the symbol cache is not
/// thread safe. If this is required, synchronizing cache acccess should be
/// ensured by the caller.
class SymbolCache : public SymbolCacheBase {
public:
  /// In the building phase, add symbols.
  void addDefinition(mlir::Attribute key, mlir::Operation *op) override {
    symbolCache.try_emplace(key, op);
  }

  // Pull in getDefinition(mlir::FlatSymbolRefAttr symbol)
  using SymbolCacheBase::getDefinition;
  mlir::Operation *getDefinition(mlir::Attribute attr) const override {
    auto it = symbolCache.find(attr);
    if (it == symbolCache.end())
      return nullptr;
    return it->second;
  }

protected:
  /// This stores a lookup table from symbol attribute to the operation
  /// that defines it.
  llvm::DenseMap<mlir::Attribute, mlir::Operation *> symbolCache;

private:
  /// Iterator support: A simple mapping between decltype(symbolCache)::iterator
  /// to SymbolCacheBase::Iterator.
  using Iterator = decltype(symbolCache)::iterator;
  struct SymbolCacheIteratorImpl : public CacheIteratorImpl {
    SymbolCacheIteratorImpl(Iterator it) : it(it) {}
    CacheItem operator*() override { return {it->getFirst(), it->getSecond()}; }
    void operator++() override { it++; }
    bool operator==(CacheIteratorImpl *other) override {
      return it == static_cast<SymbolCacheIteratorImpl *>(other)->it;
    }
    Iterator it;
  };

public:
  SymbolCacheBase::Iterator begin() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<SymbolCacheIteratorImpl>(symbolCache.begin()));
  }
  SymbolCacheBase::Iterator end() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<SymbolCacheIteratorImpl>(symbolCache.end()));
  }
};

} // namespace mlir

#endif // MLIR_SUPPORT_SYMCACHE_H
