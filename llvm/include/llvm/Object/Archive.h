//===- Archive.h - ar archive file format -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ar archive file format class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ARCHIVE_H
#define LLVM_OBJECT_ARCHIVE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace object {
struct ArchiveMemberHeader {
  char Name[16];
  char LastModified[12];
  char UID[6];
  char GID[6];
  char AccessMode[8];
  char Size[10]; ///< Size of data, not including header or padding.
  char Terminator[2];

  /// Get the name without looking up long names.
  llvm::StringRef getName() const;

  /// Members are not larger than 4GB.
  ErrorOr<uint32_t> getSize() const;

  sys::fs::perms getAccessMode() const;
  sys::TimeValue getLastModified() const;
  llvm::StringRef getRawLastModified() const {
    return StringRef(LastModified, sizeof(LastModified)).rtrim(' ');
  }
  unsigned getUID() const;
  unsigned getGID() const;
};

class Archive : public Binary {
  virtual void anchor();
public:
  class Child {
    friend Archive;
    const Archive *Parent;
    /// \brief Includes header but not padding byte.
    StringRef Data;
    /// \brief Offset from Data to the start of the file.
    uint16_t StartOfFile;

    const ArchiveMemberHeader *getHeader() const {
      return reinterpret_cast<const ArchiveMemberHeader *>(Data.data());
    }

    bool isThinMember() const;

  public:
    Child(const Archive *Parent, const char *Start, std::error_code *EC);
    Child(const Archive *Parent, StringRef Data, uint16_t StartOfFile);

    bool operator ==(const Child &other) const {
      assert(Parent == other.Parent);
      return Data.begin() == other.Data.begin();
    }

    const Archive *getParent() const { return Parent; }
    ErrorOr<Child> getNext() const;

    ErrorOr<StringRef> getName() const;
    ErrorOr<std::string> getFullName() const;
    StringRef getRawName() const { return getHeader()->getName(); }
    sys::TimeValue getLastModified() const {
      return getHeader()->getLastModified();
    }
    StringRef getRawLastModified() const {
      return getHeader()->getRawLastModified();
    }
    unsigned getUID() const { return getHeader()->getUID(); }
    unsigned getGID() const { return getHeader()->getGID(); }
    sys::fs::perms getAccessMode() const {
      return getHeader()->getAccessMode();
    }
    /// \return the size of the archive member without the header or padding.
    ErrorOr<uint64_t> getSize() const;
    /// \return the size in the archive header for this member.
    ErrorOr<uint64_t> getRawSize() const;

    ErrorOr<StringRef> getBuffer() const;
    uint64_t getChildOffset() const;

    ErrorOr<MemoryBufferRef> getMemoryBufferRef() const;

    Expected<std::unique_ptr<Binary>>
    getAsBinary(LLVMContext *Context = nullptr) const;
  };

  class child_iterator {
    ErrorOr<Child> child;

  public:
    child_iterator() : child(Child(nullptr, nullptr, nullptr)) {}
    child_iterator(const Child &c) : child(c) {}
    child_iterator(std::error_code EC) : child(EC) {}
    const ErrorOr<Child> *operator->() const { return &child; }
    const ErrorOr<Child> &operator*() const { return child; }

    bool operator==(const child_iterator &other) const {
      // We ignore error states so that comparisions with end() work, which
      // allows range loops.
      if (child.getError() || other.child.getError())
        return false;
      return *child == *other.child;
    }

    bool operator!=(const child_iterator &other) const {
      return !(*this == other);
    }

    // Code in loops with child_iterators must check for errors on each loop
    // iteration.  And if there is an error break out of the loop.
    child_iterator &operator++() { // Preincrement
      assert(child && "Can't increment iterator with error");
      child = child->getNext();
      return *this;
    }
  };

  class Symbol {
    const Archive *Parent;
    uint32_t SymbolIndex;
    uint32_t StringIndex; // Extra index to the string.

  public:
    bool operator ==(const Symbol &other) const {
      return (Parent == other.Parent) && (SymbolIndex == other.SymbolIndex);
    }

    Symbol(const Archive *p, uint32_t symi, uint32_t stri)
      : Parent(p)
      , SymbolIndex(symi)
      , StringIndex(stri) {}
    StringRef getName() const;
    ErrorOr<Child> getMember() const;
    Symbol getNext() const;
  };

  class symbol_iterator {
    Symbol symbol;
  public:
    symbol_iterator(const Symbol &s) : symbol(s) {}
    const Symbol *operator->() const { return &symbol; }
    const Symbol &operator*() const { return symbol; }

    bool operator==(const symbol_iterator &other) const {
      return symbol == other.symbol;
    }

    bool operator!=(const symbol_iterator &other) const {
      return !(*this == other);
    }

    symbol_iterator& operator++() {  // Preincrement
      symbol = symbol.getNext();
      return *this;
    }
  };

  Archive(MemoryBufferRef Source, std::error_code &EC);
  static ErrorOr<std::unique_ptr<Archive>> create(MemoryBufferRef Source);

  enum Kind {
    K_GNU,
    K_MIPS64,
    K_BSD,
    K_DARWIN64,
    K_COFF
  };

  Kind kind() const { return (Kind)Format; }
  bool isThin() const { return IsThin; }

  child_iterator child_begin(bool SkipInternal = true) const;
  child_iterator child_end() const;
  iterator_range<child_iterator> children(bool SkipInternal = true) const {
    return make_range(child_begin(SkipInternal), child_end());
  }

  symbol_iterator symbol_begin() const;
  symbol_iterator symbol_end() const;
  iterator_range<symbol_iterator> symbols() const {
    return make_range(symbol_begin(), symbol_end());
  }

  // Cast methods.
  static inline bool classof(Binary const *v) {
    return v->isArchive();
  }

  // check if a symbol is in the archive
  child_iterator findSym(StringRef name) const;

  bool hasSymbolTable() const;
  StringRef getSymbolTable() const { return SymbolTable; }
  uint32_t getNumberOfSymbols() const;

  std::vector<std::unique_ptr<MemoryBuffer>> takeThinBuffers() {
    return std::move(ThinBuffers);
  }

private:
  StringRef SymbolTable;
  StringRef StringTable;

  StringRef FirstRegularData;
  uint16_t FirstRegularStartOfFile = -1;
  void setFirstRegular(const Child &C);

  unsigned Format : 3;
  unsigned IsThin : 1;
  mutable std::vector<std::unique_ptr<MemoryBuffer>> ThinBuffers;
};

}
}

#endif
