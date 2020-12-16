//===------------ JITLink.h - JIT linker functionality ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains generic JIT-linker types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_JITLINK_H
#define LLVM_EXECUTIONENGINE_JITLINK_JITLINK_H

#include "JITLinkMemoryManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"

#include <map>
#include <string>
#include <system_error>

namespace llvm {
namespace jitlink {

class Symbol;
class Section;

/// Base class for errors originating in JIT linker, e.g. missing relocation
/// support.
class JITLinkError : public ErrorInfo<JITLinkError> {
public:
  static char ID;

  JITLinkError(Twine ErrMsg) : ErrMsg(ErrMsg.str()) {}

  void log(raw_ostream &OS) const override;
  const std::string &getErrorMessage() const { return ErrMsg; }
  std::error_code convertToErrorCode() const override;

private:
  std::string ErrMsg;
};

/// Represents fixups and constraints in the LinkGraph.
class Edge {
public:
  using Kind = uint8_t;

  enum GenericEdgeKind : Kind {
    Invalid,                    // Invalid edge value.
    FirstKeepAlive,             // Keeps target alive. Offset/addend zero.
    KeepAlive = FirstKeepAlive, // Tag first edge kind that preserves liveness.
    FirstRelocation             // First architecture specific relocation.
  };

  using OffsetT = uint32_t;
  using AddendT = int64_t;

  Edge(Kind K, OffsetT Offset, Symbol &Target, AddendT Addend)
      : Target(&Target), Offset(Offset), Addend(Addend), K(K) {}

  OffsetT getOffset() const { return Offset; }
  void setOffset(OffsetT Offset) { this->Offset = Offset; }
  Kind getKind() const { return K; }
  void setKind(Kind K) { this->K = K; }
  bool isRelocation() const { return K >= FirstRelocation; }
  Kind getRelocation() const {
    assert(isRelocation() && "Not a relocation edge");
    return K - FirstRelocation;
  }
  bool isKeepAlive() const { return K >= FirstKeepAlive; }
  Symbol &getTarget() const { return *Target; }
  void setTarget(Symbol &Target) { this->Target = &Target; }
  AddendT getAddend() const { return Addend; }
  void setAddend(AddendT Addend) { this->Addend = Addend; }

private:
  Symbol *Target = nullptr;
  OffsetT Offset = 0;
  AddendT Addend = 0;
  Kind K = 0;
};

/// Returns the string name of the given generic edge kind, or "unknown"
/// otherwise. Useful for debugging.
const char *getGenericEdgeKindName(Edge::Kind K);

/// Base class for Addressable entities (externals, absolutes, blocks).
class Addressable {
  friend class LinkGraph;

protected:
  Addressable(JITTargetAddress Address, bool IsDefined)
      : Address(Address), IsDefined(IsDefined), IsAbsolute(false) {}

  Addressable(JITTargetAddress Address)
      : Address(Address), IsDefined(false), IsAbsolute(true) {
    assert(!(IsDefined && IsAbsolute) &&
           "Block cannot be both defined and absolute");
  }

public:
  Addressable(const Addressable &) = delete;
  Addressable &operator=(const Addressable &) = default;
  Addressable(Addressable &&) = delete;
  Addressable &operator=(Addressable &&) = default;

  JITTargetAddress getAddress() const { return Address; }
  void setAddress(JITTargetAddress Address) { this->Address = Address; }

  /// Returns true if this is a defined addressable, in which case you
  /// can downcast this to a .
  bool isDefined() const { return static_cast<bool>(IsDefined); }
  bool isAbsolute() const { return static_cast<bool>(IsAbsolute); }

private:
  JITTargetAddress Address = 0;
  uint64_t IsDefined : 1;
  uint64_t IsAbsolute : 1;
};

using SectionOrdinal = unsigned;

/// An Addressable with content and edges.
class Block : public Addressable {
  friend class LinkGraph;

private:
  /// Create a zero-fill defined addressable.
  Block(Section &Parent, JITTargetAddress Size, JITTargetAddress Address,
        uint64_t Alignment, uint64_t AlignmentOffset)
      : Addressable(Address, true), Parent(Parent), Size(Size) {
    assert(isPowerOf2_64(Alignment) && "Alignment must be power of 2");
    assert(AlignmentOffset < Alignment &&
           "Alignment offset cannot exceed alignment");
    assert(AlignmentOffset <= MaxAlignmentOffset &&
           "Alignment offset exceeds maximum");
    P2Align = Alignment ? countTrailingZeros(Alignment) : 0;
    this->AlignmentOffset = AlignmentOffset;
  }

  /// Create a defined addressable for the given content.
  Block(Section &Parent, StringRef Content, JITTargetAddress Address,
        uint64_t Alignment, uint64_t AlignmentOffset)
      : Addressable(Address, true), Parent(Parent), Data(Content.data()),
        Size(Content.size()) {
    assert(isPowerOf2_64(Alignment) && "Alignment must be power of 2");
    assert(AlignmentOffset < Alignment &&
           "Alignment offset cannot exceed alignment");
    assert(AlignmentOffset <= MaxAlignmentOffset &&
           "Alignment offset exceeds maximum");
    P2Align = Alignment ? countTrailingZeros(Alignment) : 0;
    this->AlignmentOffset = AlignmentOffset;
  }

public:
  using EdgeVector = std::vector<Edge>;
  using edge_iterator = EdgeVector::iterator;
  using const_edge_iterator = EdgeVector::const_iterator;

  Block(const Block &) = delete;
  Block &operator=(const Block &) = delete;
  Block(Block &&) = delete;
  Block &operator=(Block &&) = delete;

  /// Return the parent section for this block.
  Section &getSection() const { return Parent; }

  /// Returns true if this is a zero-fill block.
  ///
  /// If true, getSize is callable but getContent is not (the content is
  /// defined to be a sequence of zero bytes of length Size).
  bool isZeroFill() const { return !Data; }

  /// Returns the size of this defined addressable.
  size_t getSize() const { return Size; }

  /// Get the content for this block. Block must not be a zero-fill block.
  StringRef getContent() const {
    assert(Data && "Section does not contain content");
    return StringRef(Data, Size);
  }

  /// Set the content for this block.
  /// Caller is responsible for ensuring the underlying bytes are not
  /// deallocated while pointed to by this block.
  void setContent(StringRef Content) {
    Data = Content.data();
    Size = Content.size();
  }

  /// Get the alignment for this content.
  uint64_t getAlignment() const { return 1ull << P2Align; }

  /// Set the alignment for this content.
  void setAlignment(uint64_t Alignment) {
    assert(isPowerOf2_64(Alignment) && "Alignment must be a power of two");
    P2Align = Alignment ? countTrailingZeros(Alignment) : 0;
  }

  /// Get the alignment offset for this content.
  uint64_t getAlignmentOffset() const { return AlignmentOffset; }

  /// Set the alignment offset for this content.
  void setAlignmentOffset(uint64_t AlignmentOffset) {
    assert(AlignmentOffset < (1ull << P2Align) &&
           "Alignment offset can't exceed alignment");
    this->AlignmentOffset = AlignmentOffset;
  }

  /// Add an edge to this block.
  void addEdge(Edge::Kind K, Edge::OffsetT Offset, Symbol &Target,
               Edge::AddendT Addend) {
    Edges.push_back(Edge(K, Offset, Target, Addend));
  }

  /// Add an edge by copying an existing one. This is typically used when
  /// moving edges between blocks.
  void addEdge(const Edge &E) { Edges.push_back(E); }

  /// Return the list of edges attached to this content.
  iterator_range<edge_iterator> edges() {
    return make_range(Edges.begin(), Edges.end());
  }

  /// Returns the list of edges attached to this content.
  iterator_range<const_edge_iterator> edges() const {
    return make_range(Edges.begin(), Edges.end());
  }

  /// Return the size of the edges list.
  size_t edges_size() const { return Edges.size(); }

  /// Returns true if the list of edges is empty.
  bool edges_empty() const { return Edges.empty(); }

  /// Remove the edge pointed to by the given iterator.
  /// Returns an iterator to the new next element.
  edge_iterator removeEdge(edge_iterator I) { return Edges.erase(I); }

private:
  static constexpr uint64_t MaxAlignmentOffset = (1ULL << 57) - 1;

  uint64_t P2Align : 5;
  uint64_t AlignmentOffset : 57;
  Section &Parent;
  const char *Data = nullptr;
  size_t Size = 0;
  std::vector<Edge> Edges;
};

/// Describes symbol linkage. This can be used to make resolve definition
/// clashes.
enum class Linkage : uint8_t {
  Strong,
  Weak,
};

/// For errors and debugging output.
const char *getLinkageName(Linkage L);

/// Defines the scope in which this symbol should be visible:
///   Default -- Visible in the public interface of the linkage unit.
///   Hidden -- Visible within the linkage unit, but not exported from it.
///   Local -- Visible only within the LinkGraph.
enum class Scope : uint8_t { Default, Hidden, Local };

/// For debugging output.
const char *getScopeName(Scope S);

raw_ostream &operator<<(raw_ostream &OS, const Block &B);

/// Symbol representation.
///
/// Symbols represent locations within Addressable objects.
/// They can be either Named or Anonymous.
/// Anonymous symbols have neither linkage nor visibility, and must point at
/// ContentBlocks.
/// Named symbols may be in one of four states:
///   - Null: Default initialized. Assignable, but otherwise unusable.
///   - Defined: Has both linkage and visibility and points to a ContentBlock
///   - Common: Has both linkage and visibility, points to a null Addressable.
///   - External: Has neither linkage nor visibility, points to an external
///     Addressable.
///
class Symbol {
  friend class LinkGraph;

private:
  Symbol(Addressable &Base, JITTargetAddress Offset, StringRef Name,
         JITTargetAddress Size, Linkage L, Scope S, bool IsLive,
         bool IsCallable)
      : Name(Name), Base(&Base), Offset(Offset), Size(Size) {
    assert(Offset <= MaxOffset && "Offset out of range");
    setLinkage(L);
    setScope(S);
    setLive(IsLive);
    setCallable(IsCallable);
  }

  static Symbol &constructCommon(void *SymStorage, Block &Base, StringRef Name,
                                 JITTargetAddress Size, Scope S, bool IsLive) {
    assert(SymStorage && "Storage cannot be null");
    assert(!Name.empty() && "Common symbol name cannot be empty");
    assert(Base.isDefined() &&
           "Cannot create common symbol from undefined block");
    assert(static_cast<Block &>(Base).getSize() == Size &&
           "Common symbol size should match underlying block size");
    auto *Sym = reinterpret_cast<Symbol *>(SymStorage);
    new (Sym) Symbol(Base, 0, Name, Size, Linkage::Weak, S, IsLive, false);
    return *Sym;
  }

  static Symbol &constructExternal(void *SymStorage, Addressable &Base,
                                   StringRef Name, JITTargetAddress Size,
                                   Linkage L) {
    assert(SymStorage && "Storage cannot be null");
    assert(!Base.isDefined() &&
           "Cannot create external symbol from defined block");
    assert(!Name.empty() && "External symbol name cannot be empty");
    auto *Sym = reinterpret_cast<Symbol *>(SymStorage);
    new (Sym) Symbol(Base, 0, Name, Size, L, Scope::Default, false, false);
    return *Sym;
  }

  static Symbol &constructAbsolute(void *SymStorage, Addressable &Base,
                                   StringRef Name, JITTargetAddress Size,
                                   Linkage L, Scope S, bool IsLive) {
    assert(SymStorage && "Storage cannot be null");
    assert(!Base.isDefined() &&
           "Cannot create absolute symbol from a defined block");
    auto *Sym = reinterpret_cast<Symbol *>(SymStorage);
    new (Sym) Symbol(Base, 0, Name, Size, L, S, IsLive, false);
    return *Sym;
  }

  static Symbol &constructAnonDef(void *SymStorage, Block &Base,
                                  JITTargetAddress Offset,
                                  JITTargetAddress Size, bool IsCallable,
                                  bool IsLive) {
    assert(SymStorage && "Storage cannot be null");
    assert((Offset + Size) <= Base.getSize() &&
           "Symbol extends past end of block");
    auto *Sym = reinterpret_cast<Symbol *>(SymStorage);
    new (Sym) Symbol(Base, Offset, StringRef(), Size, Linkage::Strong,
                     Scope::Local, IsLive, IsCallable);
    return *Sym;
  }

  static Symbol &constructNamedDef(void *SymStorage, Block &Base,
                                   JITTargetAddress Offset, StringRef Name,
                                   JITTargetAddress Size, Linkage L, Scope S,
                                   bool IsLive, bool IsCallable) {
    assert(SymStorage && "Storage cannot be null");
    assert((Offset + Size) <= Base.getSize() &&
           "Symbol extends past end of block");
    assert(!Name.empty() && "Name cannot be empty");
    auto *Sym = reinterpret_cast<Symbol *>(SymStorage);
    new (Sym) Symbol(Base, Offset, Name, Size, L, S, IsLive, IsCallable);
    return *Sym;
  }

public:
  /// Create a null Symbol. This allows Symbols to be default initialized for
  /// use in containers (e.g. as map values). Null symbols are only useful for
  /// assigning to.
  Symbol() = default;

  // Symbols are not movable or copyable.
  Symbol(const Symbol &) = delete;
  Symbol &operator=(const Symbol &) = delete;
  Symbol(Symbol &&) = delete;
  Symbol &operator=(Symbol &&) = delete;

  /// Returns true if this symbol has a name.
  bool hasName() const { return !Name.empty(); }

  /// Returns the name of this symbol (empty if the symbol is anonymous).
  StringRef getName() const {
    assert((!Name.empty() || getScope() == Scope::Local) &&
           "Anonymous symbol has non-local scope");
    return Name;
  }

  /// Rename this symbol. The client is responsible for updating scope and
  /// linkage if this name-change requires it.
  void setName(StringRef Name) { this->Name = Name; }

  /// Returns true if this Symbol has content (potentially) defined within this
  /// object file (i.e. is anything but an external or absolute symbol).
  bool isDefined() const {
    assert(Base && "Attempt to access null symbol");
    return Base->isDefined();
  }

  /// Returns true if this symbol is live (i.e. should be treated as a root for
  /// dead stripping).
  bool isLive() const {
    assert(Base && "Attempting to access null symbol");
    return IsLive;
  }

  /// Set this symbol's live bit.
  void setLive(bool IsLive) { this->IsLive = IsLive; }

  /// Returns true is this symbol is callable.
  bool isCallable() const { return IsCallable; }

  /// Set this symbol's callable bit.
  void setCallable(bool IsCallable) { this->IsCallable = IsCallable; }

  /// Returns true if the underlying addressable is an unresolved external.
  bool isExternal() const {
    assert(Base && "Attempt to access null symbol");
    return !Base->isDefined() && !Base->isAbsolute();
  }

  /// Returns true if the underlying addressable is an absolute symbol.
  bool isAbsolute() const {
    assert(Base && "Attempt to access null symbol");
    return !Base->isDefined() && Base->isAbsolute();
  }

  /// Return the addressable that this symbol points to.
  Addressable &getAddressable() {
    assert(Base && "Cannot get underlying addressable for null symbol");
    return *Base;
  }

  /// Return the addressable that thsi symbol points to.
  const Addressable &getAddressable() const {
    assert(Base && "Cannot get underlying addressable for null symbol");
    return *Base;
  }

  /// Return the Block for this Symbol (Symbol must be defined).
  Block &getBlock() {
    assert(Base && "Cannot get block for null symbol");
    assert(Base->isDefined() && "Not a defined symbol");
    return static_cast<Block &>(*Base);
  }

  /// Return the Block for this Symbol (Symbol must be defined).
  const Block &getBlock() const {
    assert(Base && "Cannot get block for null symbol");
    assert(Base->isDefined() && "Not a defined symbol");
    return static_cast<const Block &>(*Base);
  }

  /// Returns the offset for this symbol within the underlying addressable.
  JITTargetAddress getOffset() const { return Offset; }

  /// Returns the address of this symbol.
  JITTargetAddress getAddress() const { return Base->getAddress() + Offset; }

  /// Returns the size of this symbol.
  JITTargetAddress getSize() const { return Size; }

  /// Returns true if this symbol is backed by a zero-fill block.
  /// This method may only be called on defined symbols.
  bool isSymbolZeroFill() const { return getBlock().isZeroFill(); }

  /// Returns the content in the underlying block covered by this symbol.
  /// This method may only be called on defined non-zero-fill symbols.
  StringRef getSymbolContent() const {
    return getBlock().getContent().substr(Offset, Size);
  }

  /// Get the linkage for this Symbol.
  Linkage getLinkage() const { return static_cast<Linkage>(L); }

  /// Set the linkage for this Symbol.
  void setLinkage(Linkage L) {
    assert((L == Linkage::Strong || (!Base->isAbsolute() && !Name.empty())) &&
           "Linkage can only be applied to defined named symbols");
    this->L = static_cast<uint8_t>(L);
  }

  /// Get the visibility for this Symbol.
  Scope getScope() const { return static_cast<Scope>(S); }

  /// Set the visibility for this Symbol.
  void setScope(Scope S) {
    assert((!Name.empty() || S == Scope::Local) &&
           "Can not set anonymous symbol to non-local scope");
    assert((S == Scope::Default || Base->isDefined() || Base->isAbsolute()) &&
           "Invalid visibility for symbol type");
    this->S = static_cast<uint8_t>(S);
  }

private:
  void makeExternal(Addressable &A) {
    assert(!A.isDefined() && "Attempting to make external with defined block");
    Base = &A;
    Offset = 0;
    setLinkage(Linkage::Strong);
    setScope(Scope::Default);
    IsLive = 0;
    // note: Size and IsCallable fields left unchanged.
  }

  void setBlock(Block &B) { Base = &B; }

  void setOffset(uint64_t NewOffset) {
    assert(NewOffset <= MaxOffset && "Offset out of range");
    Offset = NewOffset;
  }

  static constexpr uint64_t MaxOffset = (1ULL << 59) - 1;

  // FIXME: A char* or SymbolStringPtr may pack better.
  StringRef Name;
  Addressable *Base = nullptr;
  uint64_t Offset : 59;
  uint64_t L : 1;
  uint64_t S : 2;
  uint64_t IsLive : 1;
  uint64_t IsCallable : 1;
  JITTargetAddress Size = 0;
};

raw_ostream &operator<<(raw_ostream &OS, const Symbol &A);

void printEdge(raw_ostream &OS, const Block &B, const Edge &E,
               StringRef EdgeKindName);

/// Represents an object file section.
class Section {
  friend class LinkGraph;

private:
  Section(StringRef Name, sys::Memory::ProtectionFlags Prot,
          SectionOrdinal SecOrdinal)
      : Name(Name), Prot(Prot), SecOrdinal(SecOrdinal) {}

  using SymbolSet = DenseSet<Symbol *>;
  using BlockSet = DenseSet<Block *>;

public:
  using symbol_iterator = SymbolSet::iterator;
  using const_symbol_iterator = SymbolSet::const_iterator;

  using block_iterator = BlockSet::iterator;
  using const_block_iterator = BlockSet::const_iterator;

  ~Section();

  /// Returns the name of this section.
  StringRef getName() const { return Name; }

  /// Returns the protection flags for this section.
  sys::Memory::ProtectionFlags getProtectionFlags() const { return Prot; }

  /// Returns the ordinal for this section.
  SectionOrdinal getOrdinal() const { return SecOrdinal; }

  /// Returns an iterator over the blocks defined in this section.
  iterator_range<block_iterator> blocks() {
    return make_range(Blocks.begin(), Blocks.end());
  }

  /// Returns an iterator over the blocks defined in this section.
  iterator_range<const_block_iterator> blocks() const {
    return make_range(Blocks.begin(), Blocks.end());
  }

  /// Returns an iterator over the symbols defined in this section.
  iterator_range<symbol_iterator> symbols() {
    return make_range(Symbols.begin(), Symbols.end());
  }

  /// Returns an iterator over the symbols defined in this section.
  iterator_range<const_symbol_iterator> symbols() const {
    return make_range(Symbols.begin(), Symbols.end());
  }

  /// Return the number of symbols in this section.
  SymbolSet::size_type symbols_size() { return Symbols.size(); }

private:
  void addSymbol(Symbol &Sym) {
    assert(!Symbols.count(&Sym) && "Symbol is already in this section");
    Symbols.insert(&Sym);
  }

  void removeSymbol(Symbol &Sym) {
    assert(Symbols.count(&Sym) && "symbol is not in this section");
    Symbols.erase(&Sym);
  }

  void addBlock(Block &B) {
    assert(!Blocks.count(&B) && "Block is already in this section");
    Blocks.insert(&B);
  }

  void removeBlock(Block &B) {
    assert(Blocks.count(&B) && "Block is not in this section");
    Blocks.erase(&B);
  }

  StringRef Name;
  sys::Memory::ProtectionFlags Prot;
  SectionOrdinal SecOrdinal = 0;
  BlockSet Blocks;
  SymbolSet Symbols;
};

/// Represents a section address range via a pair of Block pointers
/// to the first and last Blocks in the section.
class SectionRange {
public:
  SectionRange() = default;
  SectionRange(const Section &Sec) {
    if (llvm::empty(Sec.blocks()))
      return;
    First = Last = *Sec.blocks().begin();
    for (auto *B : Sec.blocks()) {
      if (B->getAddress() < First->getAddress())
        First = B;
      if (B->getAddress() > Last->getAddress())
        Last = B;
    }
  }
  Block *getFirstBlock() const {
    assert((!Last || First) && "First can not be null if end is non-null");
    return First;
  }
  Block *getLastBlock() const {
    assert((First || !Last) && "Last can not be null if start is non-null");
    return Last;
  }
  bool isEmpty() const {
    assert((First || !Last) && "Last can not be null if start is non-null");
    return !First;
  }
  JITTargetAddress getStart() const {
    return First ? First->getAddress() : 0;
  }
  JITTargetAddress getEnd() const {
    return Last ? Last->getAddress() + Last->getSize() : 0;
  }
  uint64_t getSize() const { return getEnd() - getStart(); }

private:
  Block *First = nullptr;
  Block *Last = nullptr;
};

class LinkGraph {
private:
  using SectionList = std::vector<std::unique_ptr<Section>>;
  using ExternalSymbolSet = DenseSet<Symbol *>;
  using BlockSet = DenseSet<Block *>;

  template <typename... ArgTs>
  Addressable &createAddressable(ArgTs &&... Args) {
    Addressable *A =
        reinterpret_cast<Addressable *>(Allocator.Allocate<Addressable>());
    new (A) Addressable(std::forward<ArgTs>(Args)...);
    return *A;
  }

  void destroyAddressable(Addressable &A) {
    A.~Addressable();
    Allocator.Deallocate(&A);
  }

  template <typename... ArgTs> Block &createBlock(ArgTs &&... Args) {
    Block *B = reinterpret_cast<Block *>(Allocator.Allocate<Block>());
    new (B) Block(std::forward<ArgTs>(Args)...);
    B->getSection().addBlock(*B);
    return *B;
  }

  void destroyBlock(Block &B) {
    B.~Block();
    Allocator.Deallocate(&B);
  }

  void destroySymbol(Symbol &S) {
    S.~Symbol();
    Allocator.Deallocate(&S);
  }

  static iterator_range<Section::block_iterator> getSectionBlocks(Section &S) {
    return S.blocks();
  }

  static iterator_range<Section::const_block_iterator>
  getSectionConstBlocks(Section &S) {
    return S.blocks();
  }

  static iterator_range<Section::symbol_iterator>
  getSectionSymbols(Section &S) {
    return S.symbols();
  }

  static iterator_range<Section::const_symbol_iterator>
  getSectionConstSymbols(Section &S) {
    return S.symbols();
  }

public:
  using external_symbol_iterator = ExternalSymbolSet::iterator;

  using section_iterator = pointee_iterator<SectionList::iterator>;
  using const_section_iterator = pointee_iterator<SectionList::const_iterator>;

  template <typename OuterItrT, typename InnerItrT, typename T,
            iterator_range<InnerItrT> getInnerRange(
                typename OuterItrT::reference)>
  class nested_collection_iterator
      : public iterator_facade_base<
            nested_collection_iterator<OuterItrT, InnerItrT, T, getInnerRange>,
            std::forward_iterator_tag, T> {
  public:
    nested_collection_iterator() = default;

    nested_collection_iterator(OuterItrT OuterI, OuterItrT OuterE)
        : OuterI(OuterI), OuterE(OuterE),
          InnerI(getInnerBegin(OuterI, OuterE)) {
      moveToNonEmptyInnerOrEnd();
    }

    bool operator==(const nested_collection_iterator &RHS) const {
      return (OuterI == RHS.OuterI) && (InnerI == RHS.InnerI);
    }

    T operator*() const {
      assert(InnerI != getInnerRange(*OuterI).end() && "Dereferencing end?");
      return *InnerI;
    }

    nested_collection_iterator operator++() {
      ++InnerI;
      moveToNonEmptyInnerOrEnd();
      return *this;
    }

  private:
    static InnerItrT getInnerBegin(OuterItrT OuterI, OuterItrT OuterE) {
      return OuterI != OuterE ? getInnerRange(*OuterI).begin() : InnerItrT();
    }

    void moveToNonEmptyInnerOrEnd() {
      while (OuterI != OuterE && InnerI == getInnerRange(*OuterI).end()) {
        ++OuterI;
        InnerI = getInnerBegin(OuterI, OuterE);
      }
    }

    OuterItrT OuterI, OuterE;
    InnerItrT InnerI;
  };

  using defined_symbol_iterator =
      nested_collection_iterator<const_section_iterator,
                                 Section::symbol_iterator, Symbol *,
                                 getSectionSymbols>;

  using const_defined_symbol_iterator =
      nested_collection_iterator<const_section_iterator,
                                 Section::const_symbol_iterator, const Symbol *,
                                 getSectionConstSymbols>;

  using block_iterator = nested_collection_iterator<const_section_iterator,
                                                    Section::block_iterator,
                                                    Block *, getSectionBlocks>;

  using const_block_iterator =
      nested_collection_iterator<const_section_iterator,
                                 Section::const_block_iterator, const Block *,
                                 getSectionConstBlocks>;

  LinkGraph(std::string Name, const Triple &TT, unsigned PointerSize,
            support::endianness Endianness)
      : Name(std::move(Name)), TT(TT), PointerSize(PointerSize),
        Endianness(Endianness) {}

  /// Returns the name of this graph (usually the name of the original
  /// underlying MemoryBuffer).
  const std::string &getName() { return Name; }

  /// Returns the target triple for this Graph.
  const Triple &getTargetTriple() const { return TT; }

  /// Returns the pointer size for use in this graph.
  unsigned getPointerSize() const { return PointerSize; }

  /// Returns the endianness of content in this graph.
  support::endianness getEndianness() const { return Endianness; }

  /// Allocate a copy of the given string using the LinkGraph's allocator.
  /// This can be useful when renaming symbols or adding new content to the
  /// graph.
  StringRef allocateString(StringRef Source) {
    auto *AllocatedBuffer = Allocator.Allocate<char>(Source.size());
    llvm::copy(Source, AllocatedBuffer);
    return StringRef(AllocatedBuffer, Source.size());
  }

  /// Allocate a copy of the given string using the LinkGraph's allocator.
  /// This can be useful when renaming symbols or adding new content to the
  /// graph.
  ///
  /// Note: This Twine-based overload requires an extra string copy and an
  /// extra heap allocation for large strings. The StringRef overload should
  /// be preferred where possible.
  StringRef allocateString(Twine Source) {
    SmallString<256> TmpBuffer;
    auto SourceStr = Source.toStringRef(TmpBuffer);
    auto *AllocatedBuffer = Allocator.Allocate<char>(SourceStr.size());
    llvm::copy(SourceStr, AllocatedBuffer);
    return StringRef(AllocatedBuffer, SourceStr.size());
  }

  /// Create a section with the given name, protection flags, and alignment.
  Section &createSection(StringRef Name, sys::Memory::ProtectionFlags Prot) {
    std::unique_ptr<Section> Sec(new Section(Name, Prot, Sections.size()));
    Sections.push_back(std::move(Sec));
    return *Sections.back();
  }

  /// Create a content block.
  Block &createContentBlock(Section &Parent, StringRef Content,
                            uint64_t Address, uint64_t Alignment,
                            uint64_t AlignmentOffset) {
    return createBlock(Parent, Content, Address, Alignment, AlignmentOffset);
  }

  /// Create a zero-fill block.
  Block &createZeroFillBlock(Section &Parent, uint64_t Size, uint64_t Address,
                             uint64_t Alignment, uint64_t AlignmentOffset) {
    return createBlock(Parent, Size, Address, Alignment, AlignmentOffset);
  }

  /// Cache type for the splitBlock function.
  using SplitBlockCache = Optional<SmallVector<Symbol *, 8>>;

  /// Splits block B at the given index which must be greater than zero.
  /// If SplitIndex == B.getSize() then this function is a no-op and returns B.
  /// If SplitIndex < B.getSize() then this function returns a new block
  /// covering the range [ 0, SplitIndex ), and B is modified to cover the range
  /// [ SplitIndex, B.size() ).
  ///
  /// The optional Cache parameter can be used to speed up repeated calls to
  /// splitBlock for a single block. If the value is None the cache will be
  /// treated as uninitialized and splitBlock will populate it. Otherwise it
  /// is assumed to contain the list of Symbols pointing at B, sorted in
  /// descending order of offset.
  ///
  /// Notes:
  ///
  /// 1. The newly introduced block will have a new ordinal which will be
  ///    higher than any other ordinals in the section. Clients are responsible
  ///    for re-assigning block ordinals to restore a compatible order if
  ///    needed.
  ///
  /// 2. The cache is not automatically updated if new symbols are introduced
  ///    between calls to splitBlock. Any newly introduced symbols may be
  ///    added to the cache manually (descending offset order must be
  ///    preserved), or the cache can be set to None and rebuilt by
  ///    splitBlock on the next call.
  Block &splitBlock(Block &B, size_t SplitIndex,
                    SplitBlockCache *Cache = nullptr);

  /// Add an external symbol.
  /// Some formats (e.g. ELF) allow Symbols to have sizes. For Symbols whose
  /// size is not known, you should substitute '0'.
  /// For external symbols Linkage determines whether the symbol must be
  /// present during lookup: Externals with strong linkage must be found or
  /// an error will be emitted. Externals with weak linkage are permitted to
  /// be undefined, in which case they are assigned a value of 0.
  Symbol &addExternalSymbol(StringRef Name, uint64_t Size, Linkage L) {
    auto &Sym =
        Symbol::constructExternal(Allocator.Allocate<Symbol>(),
                                  createAddressable(0, false), Name, Size, L);
    ExternalSymbols.insert(&Sym);
    return Sym;
  }

  /// Add an absolute symbol.
  Symbol &addAbsoluteSymbol(StringRef Name, JITTargetAddress Address,
                            uint64_t Size, Linkage L, Scope S, bool IsLive) {
    auto &Sym = Symbol::constructAbsolute(Allocator.Allocate<Symbol>(),
                                          createAddressable(Address), Name,
                                          Size, L, S, IsLive);
    AbsoluteSymbols.insert(&Sym);
    return Sym;
  }

  /// Convenience method for adding a weak zero-fill symbol.
  Symbol &addCommonSymbol(StringRef Name, Scope S, Section &Section,
                          JITTargetAddress Address, uint64_t Size,
                          uint64_t Alignment, bool IsLive) {
    auto &Sym = Symbol::constructCommon(
        Allocator.Allocate<Symbol>(),
        createBlock(Section, Size, Address, Alignment, 0), Name, Size, S,
        IsLive);
    Section.addSymbol(Sym);
    return Sym;
  }

  /// Add an anonymous symbol.
  Symbol &addAnonymousSymbol(Block &Content, JITTargetAddress Offset,
                             JITTargetAddress Size, bool IsCallable,
                             bool IsLive) {
    auto &Sym = Symbol::constructAnonDef(Allocator.Allocate<Symbol>(), Content,
                                         Offset, Size, IsCallable, IsLive);
    Content.getSection().addSymbol(Sym);
    return Sym;
  }

  /// Add a named symbol.
  Symbol &addDefinedSymbol(Block &Content, JITTargetAddress Offset,
                           StringRef Name, JITTargetAddress Size, Linkage L,
                           Scope S, bool IsCallable, bool IsLive) {
    auto &Sym =
        Symbol::constructNamedDef(Allocator.Allocate<Symbol>(), Content, Offset,
                                  Name, Size, L, S, IsLive, IsCallable);
    Content.getSection().addSymbol(Sym);
    return Sym;
  }

  iterator_range<section_iterator> sections() {
    return make_range(section_iterator(Sections.begin()),
                      section_iterator(Sections.end()));
  }

  /// Returns the section with the given name if it exists, otherwise returns
  /// null.
  Section *findSectionByName(StringRef Name) {
    for (auto &S : sections())
      if (S.getName() == Name)
        return &S;
    return nullptr;
  }

  iterator_range<block_iterator> blocks() {
    return make_range(block_iterator(Sections.begin(), Sections.end()),
                      block_iterator(Sections.end(), Sections.end()));
  }

  iterator_range<const_block_iterator> blocks() const {
    return make_range(const_block_iterator(Sections.begin(), Sections.end()),
                      const_block_iterator(Sections.end(), Sections.end()));
  }

  iterator_range<external_symbol_iterator> external_symbols() {
    return make_range(ExternalSymbols.begin(), ExternalSymbols.end());
  }

  iterator_range<external_symbol_iterator> absolute_symbols() {
    return make_range(AbsoluteSymbols.begin(), AbsoluteSymbols.end());
  }

  iterator_range<defined_symbol_iterator> defined_symbols() {
    return make_range(defined_symbol_iterator(Sections.begin(), Sections.end()),
                      defined_symbol_iterator(Sections.end(), Sections.end()));
  }

  iterator_range<const_defined_symbol_iterator> defined_symbols() const {
    return make_range(
        const_defined_symbol_iterator(Sections.begin(), Sections.end()),
        const_defined_symbol_iterator(Sections.end(), Sections.end()));
  }

  /// Turn a defined symbol into an external one.
  void makeExternal(Symbol &Sym) {
    if (Sym.getAddressable().isAbsolute()) {
      assert(AbsoluteSymbols.count(&Sym) &&
             "Sym is not in the absolute symbols set");
      AbsoluteSymbols.erase(&Sym);
    } else {
      assert(Sym.isDefined() && "Sym is not a defined symbol");
      Section &Sec = Sym.getBlock().getSection();
      Sec.removeSymbol(Sym);
    }
    Sym.makeExternal(createAddressable(0, false));
    ExternalSymbols.insert(&Sym);
  }

  /// Removes an external symbol. Also removes the underlying Addressable.
  void removeExternalSymbol(Symbol &Sym) {
    assert(!Sym.isDefined() && !Sym.isAbsolute() &&
           "Sym is not an external symbol");
    assert(ExternalSymbols.count(&Sym) && "Symbol is not in the externals set");
    ExternalSymbols.erase(&Sym);
    Addressable &Base = *Sym.Base;
    destroySymbol(Sym);
    destroyAddressable(Base);
  }

  /// Remove an absolute symbol. Also removes the underlying Addressable.
  void removeAbsoluteSymbol(Symbol &Sym) {
    assert(!Sym.isDefined() && Sym.isAbsolute() &&
           "Sym is not an absolute symbol");
    assert(AbsoluteSymbols.count(&Sym) &&
           "Symbol is not in the absolute symbols set");
    AbsoluteSymbols.erase(&Sym);
    Addressable &Base = *Sym.Base;
    destroySymbol(Sym);
    destroyAddressable(Base);
  }

  /// Removes defined symbols. Does not remove the underlying block.
  void removeDefinedSymbol(Symbol &Sym) {
    assert(Sym.isDefined() && "Sym is not a defined symbol");
    Sym.getBlock().getSection().removeSymbol(Sym);
    destroySymbol(Sym);
  }

  /// Remove a block.
  void removeBlock(Block &B) {
    assert(llvm::none_of(B.getSection().symbols(),
                         [&](const Symbol *Sym) {
                           return &Sym->getBlock() == &B;
                         }) &&
           "Block still has symbols attached");
    B.getSection().removeBlock(B);
    destroyBlock(B);
  }

  /// Dump the graph.
  ///
  /// If supplied, the EdgeKindToName function will be used to name edge
  /// kinds in the debug output. Otherwise raw edge kind numbers will be
  /// displayed.
  void dump(raw_ostream &OS,
            std::function<StringRef(Edge::Kind)> EdegKindToName =
                std::function<StringRef(Edge::Kind)>());

private:
  // Put the BumpPtrAllocator first so that we don't free any of the underlying
  // memory until the Symbol/Addressable destructors have been run.
  BumpPtrAllocator Allocator;

  std::string Name;
  Triple TT;
  unsigned PointerSize;
  support::endianness Endianness;
  SectionList Sections;
  ExternalSymbolSet ExternalSymbols;
  ExternalSymbolSet AbsoluteSymbols;
};

/// Enables easy lookup of blocks by addresses.
class BlockAddressMap {
public:
  using AddrToBlockMap = std::map<JITTargetAddress, Block *>;
  using const_iterator = AddrToBlockMap::const_iterator;

  /// A block predicate that always adds all blocks.
  static bool includeAllBlocks(const Block &B) { return true; }

  /// A block predicate that always includes blocks with non-null addresses.
  static bool includeNonNull(const Block &B) { return B.getAddress(); }

  BlockAddressMap() = default;

  /// Add a block to the map. Returns an error if the block overlaps with any
  /// existing block.
  template <typename PredFn = decltype(includeAllBlocks)>
  Error addBlock(Block &B, PredFn Pred = includeAllBlocks) {
    if (!Pred(B))
      return Error::success();

    auto I = AddrToBlock.upper_bound(B.getAddress());

    // If we're not at the end of the map, check for overlap with the next
    // element.
    if (I != AddrToBlock.end()) {
      if (B.getAddress() + B.getSize() > I->second->getAddress())
        return overlapError(B, *I->second);
    }

    // If we're not at the start of the map, check for overlap with the previous
    // element.
    if (I != AddrToBlock.begin()) {
      auto &PrevBlock = *std::prev(I)->second;
      if (PrevBlock.getAddress() + PrevBlock.getSize() > B.getAddress())
        return overlapError(B, PrevBlock);
    }

    AddrToBlock.insert(I, std::make_pair(B.getAddress(), &B));
    return Error::success();
  }

  /// Add a block to the map without checking for overlap with existing blocks.
  /// The client is responsible for ensuring that the block added does not
  /// overlap with any existing block.
  void addBlockWithoutChecking(Block &B) { AddrToBlock[B.getAddress()] = &B; }

  /// Add a range of blocks to the map. Returns an error if any block in the
  /// range overlaps with any other block in the range, or with any existing
  /// block in the map.
  template <typename BlockPtrRange,
            typename PredFn = decltype(includeAllBlocks)>
  Error addBlocks(BlockPtrRange &&Blocks, PredFn Pred = includeAllBlocks) {
    for (auto *B : Blocks)
      if (auto Err = addBlock(*B, Pred))
        return Err;
    return Error::success();
  }

  /// Add a range of blocks to the map without checking for overlap with
  /// existing blocks. The client is responsible for ensuring that the block
  /// added does not overlap with any existing block.
  template <typename BlockPtrRange>
  void addBlocksWithoutChecking(BlockPtrRange &&Blocks) {
    for (auto *B : Blocks)
      addBlockWithoutChecking(*B);
  }

  /// Iterates over (Address, Block*) pairs in ascending order of address.
  const_iterator begin() const { return AddrToBlock.begin(); }
  const_iterator end() const { return AddrToBlock.end(); }

  /// Returns the block starting at the given address, or nullptr if no such
  /// block exists.
  Block *getBlockAt(JITTargetAddress Addr) const {
    auto I = AddrToBlock.find(Addr);
    if (I == AddrToBlock.end())
      return nullptr;
    return I->second;
  }

  /// Returns the block covering the given address, or nullptr if no such block
  /// exists.
  Block *getBlockCovering(JITTargetAddress Addr) const {
    auto I = AddrToBlock.upper_bound(Addr);
    if (I == AddrToBlock.begin())
      return nullptr;
    auto *B = std::prev(I)->second;
    if (Addr < B->getAddress() + B->getSize())
      return B;
    return nullptr;
  }

private:
  Error overlapError(Block &NewBlock, Block &ExistingBlock) {
    auto NewBlockEnd = NewBlock.getAddress() + NewBlock.getSize();
    auto ExistingBlockEnd =
        ExistingBlock.getAddress() + ExistingBlock.getSize();
    return make_error<JITLinkError>(
        "Block at " +
        formatv("{0:x16} -- {1:x16}", NewBlock.getAddress(), NewBlockEnd) +
        " overlaps " +
        formatv("{0:x16} -- {1:x16}", ExistingBlock.getAddress(),
                ExistingBlockEnd));
  }

  AddrToBlockMap AddrToBlock;
};

/// A map of addresses to Symbols.
class SymbolAddressMap {
public:
  using SymbolVector = SmallVector<Symbol *, 1>;

  /// Add a symbol to the SymbolAddressMap.
  void addSymbol(Symbol &Sym) {
    AddrToSymbols[Sym.getAddress()].push_back(&Sym);
  }

  /// Add all symbols in a given range to the SymbolAddressMap.
  template <typename SymbolPtrCollection>
  void addSymbols(SymbolPtrCollection &&Symbols) {
    for (auto *Sym : Symbols)
      addSymbol(*Sym);
  }

  /// Returns the list of symbols that start at the given address, or nullptr if
  /// no such symbols exist.
  const SymbolVector *getSymbolsAt(JITTargetAddress Addr) const {
    auto I = AddrToSymbols.find(Addr);
    if (I == AddrToSymbols.end())
      return nullptr;
    return &I->second;
  }

private:
  std::map<JITTargetAddress, SymbolVector> AddrToSymbols;
};

/// A function for mutating LinkGraphs.
using LinkGraphPassFunction = std::function<Error(LinkGraph &)>;

/// A list of LinkGraph passes.
using LinkGraphPassList = std::vector<LinkGraphPassFunction>;

/// An LinkGraph pass configuration, consisting of a list of pre-prune,
/// post-prune, and post-fixup passes.
struct PassConfiguration {

  /// Pre-prune passes.
  ///
  /// These passes are called on the graph after it is built, and before any
  /// symbols have been pruned. Graph nodes still have their original vmaddrs.
  ///
  /// Notable use cases: Marking symbols live or should-discard.
  LinkGraphPassList PrePrunePasses;

  /// Post-prune passes.
  ///
  /// These passes are called on the graph after dead stripping, but before
  /// memory is allocated or nodes assigned their final addresses.
  ///
  /// Notable use cases: Building GOT, stub, and TLV symbols.
  LinkGraphPassList PostPrunePasses;

  /// Pre-fixup passes.
  ///
  /// These passes are called on the graph after memory has been allocated,
  /// content copied into working memory, and nodes have been assigned their
  /// final addresses.
  ///
  /// Notable use cases: Late link-time optimizations like GOT and stub
  /// elimination.
  LinkGraphPassList PostAllocationPasses;

  /// Post-fixup passes.
  ///
  /// These passes are called on the graph after block contents has been copied
  /// to working memory, and fixups applied. Graph nodes have been updated to
  /// their final target vmaddrs.
  ///
  /// Notable use cases: Testing and validation.
  LinkGraphPassList PostFixupPasses;
};

/// Flags for symbol lookup.
///
/// FIXME: These basically duplicate orc::SymbolLookupFlags -- We should merge
///        the two types once we have an OrcSupport library.
enum class SymbolLookupFlags { RequiredSymbol, WeaklyReferencedSymbol };

raw_ostream &operator<<(raw_ostream &OS, const SymbolLookupFlags &LF);

/// A map of symbol names to resolved addresses.
using AsyncLookupResult = DenseMap<StringRef, JITEvaluatedSymbol>;

/// A function object to call with a resolved symbol map (See AsyncLookupResult)
/// or an error if resolution failed.
class JITLinkAsyncLookupContinuation {
public:
  virtual ~JITLinkAsyncLookupContinuation() {}
  virtual void run(Expected<AsyncLookupResult> LR) = 0;

private:
  virtual void anchor();
};

/// Create a lookup continuation from a function object.
template <typename Continuation>
std::unique_ptr<JITLinkAsyncLookupContinuation>
createLookupContinuation(Continuation Cont) {

  class Impl final : public JITLinkAsyncLookupContinuation {
  public:
    Impl(Continuation C) : C(std::move(C)) {}
    void run(Expected<AsyncLookupResult> LR) override { C(std::move(LR)); }

  private:
    Continuation C;
  };

  return std::make_unique<Impl>(std::move(Cont));
}

/// Holds context for a single jitLink invocation.
class JITLinkContext {
public:
  using LookupMap = DenseMap<StringRef, SymbolLookupFlags>;

  /// Create a JITLinkContext.
  JITLinkContext(const JITLinkDylib *JD) : JD(JD) {}

  /// Destroy a JITLinkContext.
  virtual ~JITLinkContext();

  /// Return the JITLinkDylib that this link is targeting, if any.
  const JITLinkDylib *getJITLinkDylib() const { return JD; }

  /// Return the MemoryManager to be used for this link.
  virtual JITLinkMemoryManager &getMemoryManager() = 0;

  /// Notify this context that linking failed.
  /// Called by JITLink if linking cannot be completed.
  virtual void notifyFailed(Error Err) = 0;

  /// Called by JITLink to resolve external symbols. This method is passed a
  /// lookup continutation which it must call with a result to continue the
  /// linking process.
  virtual void lookup(const LookupMap &Symbols,
                      std::unique_ptr<JITLinkAsyncLookupContinuation> LC) = 0;

  /// Called by JITLink once all defined symbols in the graph have been assigned
  /// their final memory locations in the target process. At this point the
  /// LinkGraph can be inspected to build a symbol table, however the block
  /// content will not generally have been copied to the target location yet.
  ///
  /// If the client detects an error in the LinkGraph state (e.g. unexpected or
  /// missing symbols) they may return an error here. The error will be
  /// propagated to notifyFailed and the linker will bail out.
  virtual Error notifyResolved(LinkGraph &G) = 0;

  /// Called by JITLink to notify the context that the object has been
  /// finalized (i.e. emitted to memory and memory permissions set). If all of
  /// this objects dependencies have also been finalized then the code is ready
  /// to run.
  virtual void
  notifyFinalized(std::unique_ptr<JITLinkMemoryManager::Allocation> A) = 0;

  /// Called by JITLink prior to linking to determine whether default passes for
  /// the target should be added. The default implementation returns true.
  /// If subclasses override this method to return false for any target then
  /// they are required to fully configure the pass pipeline for that target.
  virtual bool shouldAddDefaultTargetPasses(const Triple &TT) const;

  /// Returns the mark-live pass to be used for this link. If no pass is
  /// returned (the default) then the target-specific linker implementation will
  /// choose a conservative default (usually marking all symbols live).
  /// This function is only called if shouldAddDefaultTargetPasses returns true,
  /// otherwise the JITContext is responsible for adding a mark-live pass in
  /// modifyPassConfig.
  virtual LinkGraphPassFunction getMarkLivePass(const Triple &TT) const;

  /// Called by JITLink to modify the pass pipeline prior to linking.
  /// The default version performs no modification.
  virtual Error modifyPassConfig(const Triple &TT, PassConfiguration &Config);

private:
  const JITLinkDylib *JD = nullptr;
};

/// Marks all symbols in a graph live. This can be used as a default,
/// conservative mark-live implementation.
Error markAllSymbolsLive(LinkGraph &G);

/// Create a LinkGraph from the given object buffer.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromObject(MemoryBufferRef ObjectBuffer);

/// Link the given graph.
void link(std::unique_ptr<LinkGraph> G, std::unique_ptr<JITLinkContext> Ctx);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINK_H
