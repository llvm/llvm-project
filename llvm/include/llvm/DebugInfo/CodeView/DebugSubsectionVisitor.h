//===- DebugSubsectionVisitor.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {

namespace codeview {

class DebugChecksumsSubsectionRef;
class DebugSubsectionRecord;
class DebugInlineeLinesSubsectionRef;
class DebugCrossModuleExportsSubsectionRef;
class DebugCrossModuleImportsSubsectionRef;
class DebugFrameDataSubsectionRef;
class DebugLinesSubsectionRef;
class DebugStringTableSubsectionRef;
class DebugSymbolRVASubsectionRef;
class DebugSymbolsSubsectionRef;
class DebugUnknownSubsectionRef;

struct DebugSubsectionState {
public:
  // If no subsections are known about initially, we find as much as we can.
  DebugSubsectionState();

  // If only a string table subsection is given, we find a checksums subsection.
  explicit DebugSubsectionState(const DebugStringTableSubsectionRef &Strings);

  // If both subsections are given, we don't need to find anything.
  DebugSubsectionState(const DebugStringTableSubsectionRef &Strings,
                       const DebugChecksumsSubsectionRef &Checksums);

  template <typename T> void initialize(T &&FragmentRange) {
    for (const DebugSubsectionRecord &R : FragmentRange) {
      if (Strings && Checksums)
        return;
      if (R.kind() == DebugSubsectionKind::FileChecksums) {
        initializeChecksums(R);
        continue;
      }
      if (R.kind() == DebugSubsectionKind::StringTable && !Strings) {
        // While in practice we should never encounter a string table even
        // though the string table is already initialized, in theory it's
        // possible.  PDBs are supposed to have one global string table and
        // then this subsection should not appear.  Whereas object files are
        // supposed to have this subsection appear exactly once.  However,
        // for testing purposes it's nice to be able to test this subsection
        // independently of one format or the other, so for some tests we
        // manually construct a PDB that contains this subsection in addition
        // to a global string table.
        initializeStrings(R);
        continue;
      }
    }
  }

  const DebugStringTableSubsectionRef &strings() const { return *Strings; }
  const DebugChecksumsSubsectionRef &checksums() const { return *Checksums; }

private:
  void initializeStrings(const DebugSubsectionRecord &SR);
  void initializeChecksums(const DebugSubsectionRecord &FCR);

  std::unique_ptr<DebugStringTableSubsectionRef> OwnedStrings;
  std::unique_ptr<DebugChecksumsSubsectionRef> OwnedChecksums;

  const DebugStringTableSubsectionRef *Strings = nullptr;
  const DebugChecksumsSubsectionRef *Checksums = nullptr;
};

class DebugSubsectionVisitor {
public:
  virtual ~DebugSubsectionVisitor() = default;

  virtual Error visitUnknown(DebugUnknownSubsectionRef &Unknown) {
    return Error::success();
  }
  virtual Error visitLines(DebugLinesSubsectionRef &Lines,
                           const DebugSubsectionState &State) = 0;
  virtual Error visitFileChecksums(DebugChecksumsSubsectionRef &Checksums,
                                   const DebugSubsectionState &State) = 0;
  virtual Error visitInlineeLines(DebugInlineeLinesSubsectionRef &Inlinees,
                                  const DebugSubsectionState &State) = 0;
  virtual Error
  visitCrossModuleExports(DebugCrossModuleExportsSubsectionRef &CSE,
                          const DebugSubsectionState &State) = 0;
  virtual Error
  visitCrossModuleImports(DebugCrossModuleImportsSubsectionRef &CSE,
                          const DebugSubsectionState &State) = 0;

  virtual Error visitStringTable(DebugStringTableSubsectionRef &ST,
                                 const DebugSubsectionState &State) = 0;

  virtual Error visitSymbols(DebugSymbolsSubsectionRef &CSE,
                             const DebugSubsectionState &State) = 0;

  virtual Error visitFrameData(DebugFrameDataSubsectionRef &FD,
                               const DebugSubsectionState &State) = 0;
  virtual Error visitCOFFSymbolRVAs(DebugSymbolRVASubsectionRef &RVAs,
                                    const DebugSubsectionState &State) = 0;
};

Error visitDebugSubsection(const DebugSubsectionRecord &R,
                           DebugSubsectionVisitor &V,
                           const DebugSubsectionState &State);

namespace detail {
template <typename T>
Error visitDebugSubsections(T &&FragmentRange, DebugSubsectionVisitor &V,
                            DebugSubsectionState &State) {
  State.initialize(std::forward<T>(FragmentRange));

  for (const DebugSubsectionRecord &L : FragmentRange) {
    if (auto EC = visitDebugSubsection(L, V, State))
      return EC;
  }
  return Error::success();
}
} // namespace detail

template <typename T>
Error visitDebugSubsections(T &&FragmentRange, DebugSubsectionVisitor &V) {
  DebugSubsectionState State;
  return detail::visitDebugSubsections(std::forward<T>(FragmentRange), V,
                                       State);
}

template <typename T>
Error visitDebugSubsections(T &&FragmentRange, DebugSubsectionVisitor &V,
                            const DebugStringTableSubsectionRef &Strings) {
  DebugSubsectionState State(Strings);
  return detail::visitDebugSubsections(std::forward<T>(FragmentRange), V,
                                       State);
}

template <typename T>
Error visitDebugSubsections(T &&FragmentRange, DebugSubsectionVisitor &V,
                            const DebugStringTableSubsectionRef &Strings,
                            const DebugChecksumsSubsectionRef &Checksums) {
  DebugSubsectionState State(Strings, Checksums);
  return detail::visitDebugSubsections(std::forward<T>(FragmentRange), V,
                                       State);
}

} // end namespace codeview

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
