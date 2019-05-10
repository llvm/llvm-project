//===----- MachOAtomGraphBuilder.h - MachO AtomGraph builder ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic MachO AtomGraph building code.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_EXECUTIONENGINE_JITLINK_MACHOATOMGRAPHBUILDER_H
#define LIB_EXECUTIONENGINE_JITLINK_MACHOATOMGRAPHBUILDER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#include "JITLinkGeneric.h"

#include "llvm/Object/MachO.h"

namespace llvm {
namespace jitlink {

class MachOAtomGraphBuilder {
public:
  virtual ~MachOAtomGraphBuilder();
  Expected<std::unique_ptr<AtomGraph>> buildGraph();

protected:
  using OffsetToAtomMap = std::map<JITTargetAddress, DefinedAtom *>;

  class MachOSection {
  public:
    MachOSection() = default;

    /// Create a MachO section with the given address and alignment.
    MachOSection(Section &GenericSection, JITTargetAddress Address,
                 unsigned Alignment)
        : Address(Address), GenericSection(&GenericSection),
          Alignment(Alignment) {}

    /// Create a section without address, content or size (used for common
    /// symbol sections).
    MachOSection(Section &GenericSection) : GenericSection(&GenericSection) {}

    Section &getGenericSection() const {
      assert(GenericSection && "Section is null");
      return *GenericSection;
    }

    StringRef getName() const {
      assert(GenericSection && "No generic section attached");
      return GenericSection->getName();
    }

    MachOSection &setContent(StringRef Content) {
      assert(!ContentPtr && !Size && "Content/zeroFill already set");
      ContentPtr = Content.data();
      Size = Content.size();
      return *this;
    }

    MachOSection &setZeroFill(uint64_t Size) {
      assert(!ContentPtr && !Size && "Content/zeroFill already set");
      this->Size = Size;
      return *this;
    }

    bool isZeroFill() const { return !ContentPtr; }

    bool empty() const { return getSize() == 0; }

    size_t getSize() const { return Size; }

    StringRef getContent() const {
      assert(ContentPtr && "getContent() called on zero-fill section");
      return {ContentPtr, static_cast<size_t>(Size)};
    }

    JITTargetAddress getAddress() const { return Address; }

    unsigned getAlignment() const { return Alignment; }

    MachOSection &setNoDeadStrip(bool NoDeadStrip) {
      this->NoDeadStrip = NoDeadStrip;
      return *this;
    }

    bool isNoDeadStrip() const { return NoDeadStrip; }

  private:
    JITTargetAddress Address = 0;
    Section *GenericSection = nullptr;
    const char *ContentPtr = nullptr;
    uint64_t Size = 0;
    unsigned Alignment = 0;
    bool NoDeadStrip = false;
  };

  using CustomAtomizeFunction = std::function<Error(MachOSection &S)>;

  MachOAtomGraphBuilder(const object::MachOObjectFile &Obj);

  AtomGraph &getGraph() const { return *G; }

  const object::MachOObjectFile &getObject() const { return Obj; }

  void addCustomAtomizer(StringRef SectionName, CustomAtomizeFunction Atomizer);

  virtual Error addRelocations() = 0;

  /// Returns true if Atom A and Atom B are at a fixed offset from one another
  /// (i.e. if they're part of the same alt-entry chain).
  bool areLayoutLocked(const Atom &A, const Atom &B);

private:
  static unsigned getPointerSize(const object::MachOObjectFile &Obj);
  static support::endianness getEndianness(const object::MachOObjectFile &Obj);

  MachOSection &getCommonSection();

  Error parseSections();
  Error addNonCustomAtoms();
  Error addAtoms();

  const object::MachOObjectFile &Obj;
  std::unique_ptr<AtomGraph> G;
  DenseMap<const DefinedAtom *, const DefinedAtom *> AltEntryStarts;
  DenseMap<unsigned, MachOSection> Sections;
  StringMap<CustomAtomizeFunction> CustomAtomizeFunctions;
  Optional<MachOSection> CommonSymbolsSection;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LIB_EXECUTIONENGINE_JITLINK_MACHOATOMGRAPHBUILDER_H
