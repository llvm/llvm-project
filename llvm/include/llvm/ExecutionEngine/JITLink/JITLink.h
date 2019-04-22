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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"

#include <map>
#include <string>
#include <system_error>

namespace llvm {
namespace jitlink {

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

/// Manages allocations of JIT memory.
///
/// Instances of this class may be accessed concurrently from multiple threads
/// and their implemetations should include any necessary synchronization.
class JITLinkMemoryManager {
public:
  using ProtectionFlags = sys::Memory::ProtectionFlags;

  class SegmentRequest {
  public:
    SegmentRequest() = default;
    SegmentRequest(size_t ContentSize, unsigned ContentAlign,
                   uint64_t ZeroFillSize, unsigned ZeroFillAlign)
        : ContentSize(ContentSize), ZeroFillSize(ZeroFillSize),
          ContentAlign(ContentAlign), ZeroFillAlign(ZeroFillAlign) {}
    size_t getContentSize() const { return ContentSize; }
    unsigned getContentAlignment() const { return ContentAlign; }
    uint64_t getZeroFillSize() const { return ZeroFillSize; }
    unsigned getZeroFillAlignment() const { return ZeroFillAlign; }

  private:
    size_t ContentSize = 0;
    uint64_t ZeroFillSize = 0;
    unsigned ContentAlign = 0;
    unsigned ZeroFillAlign = 0;
  };

  using SegmentsRequestMap = DenseMap<unsigned, SegmentRequest>;

  using FinalizeContinuation = std::function<void(Error)>;

  /// Represents an allocation created by the memory manager.
  ///
  /// An allocation object is responsible for allocating and owning jit-linker
  /// working and target memory, and for transfering from working to target
  /// memory.
  ///
  class Allocation {
  public:
    virtual ~Allocation();

    /// Should return the address of linker working memory for the segment with
    /// the given protection flags.
    virtual MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) = 0;

    /// Should return the final address in the target process where the segment
    /// will reside.
    virtual JITTargetAddress getTargetMemory(ProtectionFlags Seg) = 0;

    /// Should transfer from working memory to target memory, and release
    /// working memory.
    virtual void finalizeAsync(FinalizeContinuation OnFinalize) = 0;

    /// Should deallocate target memory.
    virtual Error deallocate() = 0;
  };

  virtual ~JITLinkMemoryManager();

  /// Create an Allocation object.
  virtual Expected<std::unique_ptr<Allocation>>
  allocate(const SegmentsRequestMap &Request) = 0;
};

// Forward declare the Atom class.
class Atom;

/// Edge class. Represents both object file relocations, as well as layout and
/// keep-alive constraints.
class Edge {
public:
  using Kind = uint8_t;

  using GenericEdgeKind = enum : Kind {
    Invalid,                    // Invalid edge value.
    FirstKeepAlive,             // Keeps target alive. Offset/addend zero.
    KeepAlive = FirstKeepAlive, // Tag first edge kind that preserves liveness.
    LayoutNext,                 // Layout constraint. Offset/Addend zero.
    FirstRelocation             // First architecture specific relocation.
  };

  using OffsetT = uint32_t;
  using AddendT = int64_t;

  Edge(Kind K, OffsetT Offset, Atom &Target, AddendT Addend)
      : Target(&Target), Offset(Offset), Addend(Addend), K(K) {}

  OffsetT getOffset() const { return Offset; }
  Kind getKind() const { return K; }
  void setKind(Kind K) { this->K = K; }
  bool isRelocation() const { return K >= FirstRelocation; }
  Kind getRelocation() const {
    assert(isRelocation() && "Not a relocation edge");
    return K - FirstRelocation;
  }
  bool isKeepAlive() const { return K >= FirstKeepAlive; }
  Atom &getTarget() const { return *Target; }
  void setTarget(Atom &Target) { this->Target = &Target; }
  AddendT getAddend() const { return Addend; }
  void setAddend(AddendT Addend) { this->Addend = Addend; }

private:
  Atom *Target;
  OffsetT Offset;
  AddendT Addend;
  Kind K = 0;
};

using EdgeVector = std::vector<Edge>;

const StringRef getGenericEdgeKindName(Edge::Kind K);

/// Base Atom class. Used by absolute and undefined atoms.
class Atom {
  friend class AtomGraph;

protected:
  /// Create a named (as yet unresolved) atom.
  Atom(StringRef Name)
      : Name(Name), IsDefined(false), IsLive(false), ShouldDiscard(false),
        IsGlobal(false), IsAbsolute(false), IsCallable(false),
        IsExported(false), IsWeak(false), HasLayoutNext(false),
        IsCommon(false) {}

  /// Create an absolute symbol atom.
  Atom(StringRef Name, JITTargetAddress Address)
      : Name(Name), Address(Address), IsDefined(true), IsLive(false),
        ShouldDiscard(false), IsGlobal(false), IsAbsolute(false),
        IsCallable(false), IsExported(false), IsWeak(false),
        HasLayoutNext(false), IsCommon(false) {}

public:
  /// Returns true if this atom has a name.
  bool hasName() const { return Name != StringRef(); }

  /// Returns the name of this atom.
  StringRef getName() const { return Name; }

  /// Returns the current target address of this atom.
  /// The initial target address (for atoms that have one) will be taken from
  /// the input object file's virtual address space. During the layout phase
  /// of JIT linking the atom's address will be updated to point to its final
  /// address in the JIT'd process.
  JITTargetAddress getAddress() const { return Address; }

  /// Set the current target address of this atom.
  void setAddress(JITTargetAddress Address) { this->Address = Address; }

  /// Returns true if this is a defined atom.
  bool isDefined() const { return IsDefined; }

  /// Returns true if this atom is marked as live.
  bool isLive() const { return IsLive; }

  /// Mark this atom as live.
  ///
  /// Note: Only defined and absolute atoms can be marked live.
  void setLive(bool IsLive) {
    assert((IsDefined || IsAbsolute || !IsLive) &&
           "Only defined and absolute atoms can be marked live");
    this->IsLive = IsLive;
  }

  /// Returns true if this atom should be discarded during pruning.
  bool shouldDiscard() const { return ShouldDiscard; }

  /// Mark this atom to be discarded.
  ///
  /// Note: Only defined and absolute atoms can be marked live.
  void setShouldDiscard(bool ShouldDiscard) {
    assert((IsDefined || IsAbsolute || !ShouldDiscard) &&
           "Only defined and absolute atoms can be marked live");
    this->ShouldDiscard = ShouldDiscard;
  }

  /// Returns true if this definition is global (i.e. visible outside this
  /// linkage unit).
  ///
  /// Note: This is distict from Exported, which means visibile outside the
  /// JITDylib that this graph is being linked in to.
  bool isGlobal() const { return IsGlobal; }

  /// Mark this atom as global.
  void setGlobal(bool IsGlobal) { this->IsGlobal = IsGlobal; }

  /// Returns true if this atom represents an absolute symbol.
  bool isAbsolute() const { return IsAbsolute; }

  /// Returns true if this atom is known to be callable.
  ///
  /// Primarily provided for easy interoperability with ORC, which uses the
  /// JITSymbolFlags::Common flag to identify symbols that can be interposed
  /// with stubs.
  bool isCallable() const { return IsCallable; }

  /// Mark this atom as callable.
  void setCallable(bool IsCallable) {
    assert((IsDefined || IsAbsolute || !IsCallable) &&
           "Callable atoms must be defined or absolute");
    this->IsCallable = IsCallable;
  }

  /// Returns true if this atom should appear in the symbol table of a final
  /// linked image.
  bool isExported() const { return IsExported; }

  /// Mark this atom as exported.
  void setExported(bool IsExported) {
    assert((!IsExported || ((IsDefined || IsAbsolute) && hasName())) &&
           "Exported atoms must have names");
    this->IsExported = IsExported;
  }

  /// Returns true if this is a weak symbol.
  bool isWeak() const { return IsWeak; }

  /// Mark this atom as weak.
  void setWeak(bool IsWeak) { this->IsWeak = IsWeak; }

private:
  StringRef Name;
  JITTargetAddress Address = 0;

  bool IsDefined : 1;
  bool IsLive : 1;
  bool ShouldDiscard : 1;

  bool IsGlobal : 1;
  bool IsAbsolute : 1;
  bool IsCallable : 1;
  bool IsExported : 1;
  bool IsWeak : 1;

protected:
  // These flags only make sense for DefinedAtom, but we can minimize the size
  // of DefinedAtom by defining them here.
  bool HasLayoutNext : 1;
  bool IsCommon : 1;
};

// Forward declare DefinedAtom.
class DefinedAtom;

raw_ostream &operator<<(raw_ostream &OS, const Atom &A);
void printEdge(raw_ostream &OS, const Atom &FixupAtom, const Edge &E,
               StringRef EdgeKindName);

/// Represents an object file section.
class Section {
  friend class AtomGraph;

private:
  Section(StringRef Name, sys::Memory::ProtectionFlags Prot, unsigned Ordinal,
          bool IsZeroFill)
      : Name(Name), Prot(Prot), Ordinal(Ordinal), IsZeroFill(IsZeroFill) {}

  using DefinedAtomSet = DenseSet<DefinedAtom *>;

public:
  using atom_iterator = DefinedAtomSet::iterator;
  using const_atom_iterator = DefinedAtomSet::const_iterator;

  ~Section();
  StringRef getName() const { return Name; }
  sys::Memory::ProtectionFlags getProtectionFlags() const { return Prot; }
  unsigned getSectionOrdinal() const { return Ordinal; }
  size_t getNextAtomOrdinal() { return ++NextAtomOrdinal; }

  bool isZeroFill() const { return IsZeroFill; }

  /// Returns an iterator over the atoms in the section (in no particular
  /// order).
  iterator_range<atom_iterator> atoms() {
    return make_range(DefinedAtoms.begin(), DefinedAtoms.end());
  }

  /// Returns an iterator over the atoms in the section (in no particular
  /// order).
  iterator_range<const_atom_iterator> atoms() const {
    return make_range(DefinedAtoms.begin(), DefinedAtoms.end());
  }

  /// Return the number of atoms in this section.
  DefinedAtomSet::size_type atoms_size() { return DefinedAtoms.size(); }

  /// Return true if this section contains no atoms.
  bool atoms_empty() const { return DefinedAtoms.empty(); }

private:
  void addAtom(DefinedAtom &DA) {
    assert(!DefinedAtoms.count(&DA) && "Atom is already in this section");
    DefinedAtoms.insert(&DA);
  }

  void removeAtom(DefinedAtom &DA) {
    assert(DefinedAtoms.count(&DA) && "Atom is not in this section");
    DefinedAtoms.erase(&DA);
  }

  StringRef Name;
  sys::Memory::ProtectionFlags Prot;
  unsigned Ordinal = 0;
  unsigned NextAtomOrdinal = 0;
  bool IsZeroFill = false;
  DefinedAtomSet DefinedAtoms;
};

/// Defined atom class. Suitable for use by defined named and anonymous
/// atoms.
class DefinedAtom : public Atom {
  friend class AtomGraph;

private:
  DefinedAtom(Section &Parent, JITTargetAddress Address, uint32_t Alignment)
      : Atom("", Address), Parent(Parent), Ordinal(Parent.getNextAtomOrdinal()),
        Alignment(Alignment) {}

  DefinedAtom(Section &Parent, StringRef Name, JITTargetAddress Address,
              uint32_t Alignment)
      : Atom(Name, Address), Parent(Parent),
        Ordinal(Parent.getNextAtomOrdinal()), Alignment(Alignment) {}

public:
  using edge_iterator = EdgeVector::iterator;

  Section &getSection() const { return Parent; }

  uint64_t getSize() const { return Size; }

  StringRef getContent() const {
    assert(!Parent.isZeroFill() && "Trying to get content for zero-fill atom");
    assert(Size <= std::numeric_limits<size_t>::max() &&
           "Content size too large");
    return {ContentPtr, static_cast<size_t>(Size)};
  }
  void setContent(StringRef Content) {
    assert(!Parent.isZeroFill() && "Calling setContent on zero-fill atom?");
    ContentPtr = Content.data();
    Size = Content.size();
  }

  bool isZeroFill() const { return Parent.isZeroFill(); }

  void setZeroFill(uint64_t Size) {
    assert(Parent.isZeroFill() && !ContentPtr &&
           "Can't set zero-fill length of a non zero-fill atom");
    this->Size = Size;
  }

  uint64_t getZeroFillSize() const {
    assert(Parent.isZeroFill() &&
           "Can't get zero-fill length of a non zero-fill atom");
    return Size;
  }

  uint32_t getAlignment() const { return Alignment; }

  bool hasLayoutNext() const { return HasLayoutNext; }
  void setLayoutNext(DefinedAtom &Next) {
    assert(!HasLayoutNext && "Atom already has layout-next constraint");
    HasLayoutNext = true;
    Edges.push_back(Edge(Edge::LayoutNext, 0, Next, 0));
  }
  DefinedAtom &getLayoutNext() {
    assert(HasLayoutNext && "Atom does not have a layout-next constraint");
    DefinedAtom *Next = nullptr;
    for (auto &E : edges())
      if (E.getKind() == Edge::LayoutNext) {
        assert(E.getTarget().isDefined() &&
               "layout-next target atom must be a defined atom");
        Next = static_cast<DefinedAtom *>(&E.getTarget());
        break;
      }
    assert(Next && "Missing LayoutNext edge");
    return *Next;
  }

  bool isCommon() const { return IsCommon; }

  void addEdge(Edge::Kind K, Edge::OffsetT Offset, Atom &Target,
               Edge::AddendT Addend) {
    assert(K != Edge::LayoutNext &&
           "Layout edges should be added via addLayoutNext");
    Edges.push_back(Edge(K, Offset, Target, Addend));
  }

  iterator_range<edge_iterator> edges() {
    return make_range(Edges.begin(), Edges.end());
  }
  size_t edges_size() const { return Edges.size(); }
  bool edges_empty() const { return Edges.empty(); }

  unsigned getOrdinal() const { return Ordinal; }

private:
  void setCommon(uint64_t Size) {
    assert(ContentPtr == 0 && "Atom already has content?");
    IsCommon = true;
    setZeroFill(Size);
  }

  EdgeVector Edges;
  uint64_t Size = 0;
  Section &Parent;
  const char *ContentPtr = nullptr;
  unsigned Ordinal = 0;
  uint32_t Alignment = 0;
};

class AtomGraph {
private:
  using SectionList = std::vector<std::unique_ptr<Section>>;
  using AddressToAtomMap = std::map<JITTargetAddress, DefinedAtom *>;
  using NamedAtomMap = DenseMap<StringRef, Atom *>;
  using ExternalAtomSet = DenseSet<Atom *>;

public:
  using external_atom_iterator = ExternalAtomSet::iterator;

  using section_iterator = pointee_iterator<SectionList::iterator>;
  using const_section_iterator = pointee_iterator<SectionList::const_iterator>;

  template <typename SecItrT, typename AtomItrT, typename T>
  class defined_atom_iterator_impl
      : public iterator_facade_base<
            defined_atom_iterator_impl<SecItrT, AtomItrT, T>,
            std::forward_iterator_tag, T> {
  public:
    defined_atom_iterator_impl() = default;

    defined_atom_iterator_impl(SecItrT SI, SecItrT SE)
        : SI(SI), SE(SE),
          AI(SI != SE ? SI->atoms().begin() : Section::atom_iterator()) {
      moveToNextAtomOrEnd();
    }

    bool operator==(const defined_atom_iterator_impl &RHS) const {
      return (SI == RHS.SI) && (AI == RHS.AI);
    }

    T operator*() const {
      assert(AI != SI->atoms().end() && "Dereferencing end?");
      return *AI;
    }

    defined_atom_iterator_impl operator++() {
      ++AI;
      moveToNextAtomOrEnd();
      return *this;
    }

  private:
    void moveToNextAtomOrEnd() {
      while (SI != SE && AI == SI->atoms().end()) {
        ++SI;
        if (SI == SE)
          AI = Section::atom_iterator();
        else
          AI = SI->atoms().begin();
      }
    }

    SecItrT SI, SE;
    AtomItrT AI;
  };

  using defined_atom_iterator =
      defined_atom_iterator_impl<section_iterator, Section::atom_iterator,
                                 DefinedAtom *>;

  using const_defined_atom_iterator =
      defined_atom_iterator_impl<const_section_iterator,
                                 Section::const_atom_iterator,
                                 const DefinedAtom *>;

  AtomGraph(std::string Name, unsigned PointerSize,
            support::endianness Endianness)
      : Name(std::move(Name)), PointerSize(PointerSize),
        Endianness(Endianness) {}

  /// Returns the name of this graph (usually the name of the original
  /// underlying MemoryBuffer).
  const std::string &getName() { return Name; }

  /// Returns the pointer size for use in this graph.
  unsigned getPointerSize() const { return PointerSize; }

  /// Returns the endianness of atom-content in this graph.
  support::endianness getEndianness() const { return Endianness; }

  /// Create a section with the given name, protection flags, and alignment.
  Section &createSection(StringRef Name, sys::Memory::ProtectionFlags Prot,
                         bool IsZeroFill) {
    std::unique_ptr<Section> Sec(
        new Section(Name, Prot, Sections.size(), IsZeroFill));
    Sections.push_back(std::move(Sec));
    return *Sections.back();
  }

  /// Add an external atom representing an undefined symbol in this graph.
  Atom &addExternalAtom(StringRef Name) {
    assert(!NamedAtoms.count(Name) && "Duplicate named atom inserted");
    Atom *A = reinterpret_cast<Atom *>(
        AtomAllocator.Allocate(sizeof(Atom), alignof(Atom)));
    new (A) Atom(Name);
    ExternalAtoms.insert(A);
    NamedAtoms[Name] = A;
    return *A;
  }

  /// Add an external atom representing an absolute symbol.
  Atom &addAbsoluteAtom(StringRef Name, JITTargetAddress Addr) {
    assert(!NamedAtoms.count(Name) && "Duplicate named atom inserted");
    Atom *A = reinterpret_cast<Atom *>(
        AtomAllocator.Allocate(sizeof(Atom), alignof(Atom)));
    new (A) Atom(Name, Addr);
    AbsoluteAtoms.insert(A);
    NamedAtoms[Name] = A;
    return *A;
  }

  /// Add an anonymous defined atom to the graph.
  ///
  /// Anonymous atoms have content but no name. They must have an address.
  DefinedAtom &addAnonymousAtom(Section &Parent, JITTargetAddress Address,
                                uint32_t Alignment) {
    DefinedAtom *A = reinterpret_cast<DefinedAtom *>(
        AtomAllocator.Allocate(sizeof(DefinedAtom), alignof(DefinedAtom)));
    new (A) DefinedAtom(Parent, Address, Alignment);
    Parent.addAtom(*A);
    getAddrToAtomMap()[A->getAddress()] = A;
    return *A;
  }

  /// Add a defined atom to the graph.
  ///
  /// Allocates and constructs a DefinedAtom instance with the given parent,
  /// name, address, and alignment.
  DefinedAtom &addDefinedAtom(Section &Parent, StringRef Name,
                              JITTargetAddress Address, uint32_t Alignment) {
    assert(!NamedAtoms.count(Name) && "Duplicate named atom inserted");
    DefinedAtom *A = reinterpret_cast<DefinedAtom *>(
        AtomAllocator.Allocate(sizeof(DefinedAtom), alignof(DefinedAtom)));
    new (A) DefinedAtom(Parent, Name, Address, Alignment);
    Parent.addAtom(*A);
    getAddrToAtomMap()[A->getAddress()] = A;
    NamedAtoms[Name] = A;
    return *A;
  }

  /// Add a common symbol atom to the graph.
  ///
  /// Adds a common-symbol atom to the graph with the given parent, name,
  /// address, alignment and size.
  DefinedAtom &addCommonAtom(Section &Parent, StringRef Name,
                             JITTargetAddress Address, uint32_t Alignment,
                             uint64_t Size) {
    assert(!NamedAtoms.count(Name) && "Duplicate named atom inserted");
    DefinedAtom *A = reinterpret_cast<DefinedAtom *>(
        AtomAllocator.Allocate(sizeof(DefinedAtom), alignof(DefinedAtom)));
    new (A) DefinedAtom(Parent, Name, Address, Alignment);
    A->setCommon(Size);
    Parent.addAtom(*A);
    NamedAtoms[Name] = A;
    return *A;
  }

  iterator_range<section_iterator> sections() {
    return make_range(section_iterator(Sections.begin()),
                      section_iterator(Sections.end()));
  }

  iterator_range<external_atom_iterator> external_atoms() {
    return make_range(ExternalAtoms.begin(), ExternalAtoms.end());
  }

  iterator_range<external_atom_iterator> absolute_atoms() {
    return make_range(AbsoluteAtoms.begin(), AbsoluteAtoms.end());
  }

  iterator_range<defined_atom_iterator> defined_atoms() {
    return make_range(defined_atom_iterator(Sections.begin(), Sections.end()),
                      defined_atom_iterator(Sections.end(), Sections.end()));
  }

  iterator_range<const_defined_atom_iterator> defined_atoms() const {
    return make_range(
        const_defined_atom_iterator(Sections.begin(), Sections.end()),
        const_defined_atom_iterator(Sections.end(), Sections.end()));
  }

  /// Returns the atom with the given name, which must exist in this graph.
  Atom &getAtomByName(StringRef Name) {
    auto I = NamedAtoms.find(Name);
    assert(I != NamedAtoms.end() && "Name not in NamedAtoms map");
    return *I->second;
  }

  /// Returns the atom with the given name, which must exist in this graph and
  /// be a DefinedAtom.
  DefinedAtom &getDefinedAtomByName(StringRef Name) {
    auto &A = getAtomByName(Name);
    assert(A.isDefined() && "Atom is not a defined atom");
    return static_cast<DefinedAtom &>(A);
  }

  /// Search for the given atom by name.
  /// Returns the atom (if found) or an error (if no atom with this name
  /// exists).
  Expected<Atom &> findAtomByName(StringRef Name) {
    auto I = NamedAtoms.find(Name);
    if (I == NamedAtoms.end())
      return make_error<JITLinkError>("No atom named " + Name);
    return *I->second;
  }

  /// Search for the given defined atom by name.
  /// Returns the defined atom (if found) or an error (if no atom with this
  /// name exists, or if one exists but is not a defined atom).
  Expected<DefinedAtom &> findDefinedAtomByName(StringRef Name) {
    auto I = NamedAtoms.find(Name);
    if (I == NamedAtoms.end())
      return make_error<JITLinkError>("No atom named " + Name);
    if (!I->second->isDefined())
      return make_error<JITLinkError>("Atom " + Name +
                                      " exists but is not a "
                                      "defined atom");
    return static_cast<DefinedAtom &>(*I->second);
  }

  /// Returns the atom covering the given address, or an error if no such atom
  /// exists.
  ///
  /// Returns null if no atom exists at the given address.
  DefinedAtom *getAtomByAddress(JITTargetAddress Address) {
    refreshAddrToAtomCache();

    // If there are no defined atoms, bail out early.
    if (AddrToAtomCache->empty())
      return nullptr;

    // Find the atom *after* the given address.
    auto I = AddrToAtomCache->upper_bound(Address);

    // If this address falls before any known atom, bail out.
    if (I == AddrToAtomCache->begin())
      return nullptr;

    // The atom we're looking for is the one before the atom we found.
    --I;

    // Otherwise range check the atom that was found.
    assert(!I->second->getContent().empty() && "Atom content not set");
    if (Address >= I->second->getAddress() + I->second->getContent().size())
      return nullptr;

    return I->second;
  }

  /// Like getAtomByAddress, but returns an Error if the given address is not
  /// covered by an atom, rather than a null pointer.
  Expected<DefinedAtom &> findAtomByAddress(JITTargetAddress Address) {
    if (auto *DA = getAtomByAddress(Address))
      return *DA;
    return make_error<JITLinkError>("No atom at address " +
                                    formatv("{0:x16}", Address));
  }

  // Remove the given external atom from the graph.
  void removeExternalAtom(Atom &A) {
    assert(!A.isDefined() && !A.isAbsolute() && "A is not an external atom");
    assert(ExternalAtoms.count(&A) && "A is not in the external atoms set");
    ExternalAtoms.erase(&A);
    A.~Atom();
  }

  /// Remove the given absolute atom from the graph.
  void removeAbsoluteAtom(Atom &A) {
    assert(A.isAbsolute() && "A is not an absolute atom");
    assert(AbsoluteAtoms.count(&A) && "A is not in the absolute atoms set");
    AbsoluteAtoms.erase(&A);
    A.~Atom();
  }

  /// Remove the given defined atom from the graph.
  void removeDefinedAtom(DefinedAtom &DA) {
    if (AddrToAtomCache) {
      assert(AddrToAtomCache->count(DA.getAddress()) &&
             "Cache exists, but does not contain atom");
      AddrToAtomCache->erase(DA.getAddress());
    }
    if (DA.hasName()) {
      assert(NamedAtoms.count(DA.getName()) && "Named atom not in map");
      NamedAtoms.erase(DA.getName());
    }
    DA.getSection().removeAtom(DA);
    DA.~DefinedAtom();
  }

  /// Invalidate the atom-to-address map.
  void invalidateAddrToAtomMap() { AddrToAtomCache = None; }

  /// Dump the graph.
  ///
  /// If supplied, the EdgeKindToName function will be used to name edge
  /// kinds in the debug output. Otherwise raw edge kind numbers will be
  /// displayed.
  void dump(raw_ostream &OS,
            std::function<StringRef(Edge::Kind)> EdegKindToName =
                std::function<StringRef(Edge::Kind)>());

private:
  AddressToAtomMap &getAddrToAtomMap() {
    refreshAddrToAtomCache();
    return *AddrToAtomCache;
  }

  const AddressToAtomMap &getAddrToAtomMap() const {
    refreshAddrToAtomCache();
    return *AddrToAtomCache;
  }

  void refreshAddrToAtomCache() const {
    if (!AddrToAtomCache) {
      AddrToAtomCache = AddressToAtomMap();
      for (auto *DA : defined_atoms())
        (*AddrToAtomCache)[DA->getAddress()] = const_cast<DefinedAtom *>(DA);
    }
  }

  // Put the BumpPtrAllocator first so that we don't free any of the atoms in
  // it until all of their destructors have been run.
  BumpPtrAllocator AtomAllocator;

  std::string Name;
  unsigned PointerSize;
  support::endianness Endianness;
  SectionList Sections;
  NamedAtomMap NamedAtoms;
  ExternalAtomSet ExternalAtoms;
  ExternalAtomSet AbsoluteAtoms;
  mutable Optional<AddressToAtomMap> AddrToAtomCache;
};

/// A function for mutating AtomGraphs.
using AtomGraphPassFunction = std::function<Error(AtomGraph &)>;

/// A list of atom graph passes.
using AtomGraphPassList = std::vector<AtomGraphPassFunction>;

/// An atom graph pass configuration, consisting of a list of pre-prune,
/// post-prune, and post-fixup passes.
struct PassConfiguration {

  /// Pre-prune passes.
  ///
  /// These passes are called on the graph after it is built, and before any
  /// atoms have been pruned.
  ///
  /// Notable use cases: Marking atoms live or should-discard.
  AtomGraphPassList PrePrunePasses;

  /// Post-prune passes.
  ///
  /// These passes are called on the graph after dead and should-discard atoms
  /// have been removed, but before fixups are applied.
  ///
  /// Notable use cases: Building GOT, stub, and TLV atoms.
  AtomGraphPassList PostPrunePasses;

  /// Post-fixup passes.
  ///
  /// These passes are called on the graph after atom contents has been copied
  /// to working memory, and fixups applied.
  ///
  /// Notable use cases: Testing and validation.
  AtomGraphPassList PostFixupPasses;
};

/// A JITLinkMemoryManager that allocates in-process memory.
class InProcessMemoryManager : public JITLinkMemoryManager {
public:
  Expected<std::unique_ptr<Allocation>>
  allocate(const SegmentsRequestMap &Request) override;
};

/// A map of symbol names to resolved addresses.
using AsyncLookupResult = DenseMap<StringRef, JITEvaluatedSymbol>;

/// A function to call with a resolved symbol map (See AsyncLookupResult) or an
/// error if resolution failed.
using JITLinkAsyncLookupContinuation =
    std::function<void(Expected<AsyncLookupResult> LR)>;

/// An asynchronous symbol lookup. Performs a search (possibly asynchronously)
/// for the given symbols, calling the given continuation with either the result
/// (if the lookup succeeds), or an error (if the lookup fails).
using JITLinkAsyncLookupFunction =
    std::function<void(const DenseSet<StringRef> &Symbols,
                       JITLinkAsyncLookupContinuation LookupContinuation)>;

/// Holds context for a single jitLink invocation.
class JITLinkContext {
public:
  /// Destroy a JITLinkContext.
  virtual ~JITLinkContext();

  /// Return the MemoryManager to be used for this link.
  virtual JITLinkMemoryManager &getMemoryManager() = 0;

  /// Returns a StringRef for the object buffer.
  /// This method can not be called once takeObjectBuffer has been called.
  virtual MemoryBufferRef getObjectBuffer() const = 0;

  /// Notify this context that linking failed.
  /// Called by JITLink if linking cannot be completed.
  virtual void notifyFailed(Error Err) = 0;

  /// Called by JITLink to resolve external symbols. This method is passed a
  /// lookup continutation which it must call with a result to continue the
  /// linking process.
  virtual void lookup(const DenseSet<StringRef> &Symbols,
                      JITLinkAsyncLookupContinuation LookupContinuation) = 0;

  /// Called by JITLink once all defined atoms in the graph have been assigned
  /// their final memory locations in the target process. At this point he
  /// atom graph can be, inspected to build a symbol table however the atom
  /// content will not generally have been copied to the target location yet.
  virtual void notifyResolved(AtomGraph &G) = 0;

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
  /// choose a conservative default (usually marking all atoms live).
  /// This function is only called if shouldAddDefaultTargetPasses returns true,
  /// otherwise the JITContext is responsible for adding a mark-live pass in
  /// modifyPassConfig.
  virtual AtomGraphPassFunction getMarkLivePass(const Triple &TT) const;

  /// Called by JITLink to modify the pass pipeline prior to linking.
  /// The default version performs no modification.
  virtual Error modifyPassConfig(const Triple &TT, PassConfiguration &Config);
};

/// Marks all atoms in a graph live. This can be used as a default, conservative
/// mark-live implementation.
Error markAllAtomsLive(AtomGraph &G);

/// Basic JITLink implementation.
///
/// This function will use sensible defaults for GOT and Stub handling.
void jitLink(std::unique_ptr<JITLinkContext> Ctx);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINK_H
