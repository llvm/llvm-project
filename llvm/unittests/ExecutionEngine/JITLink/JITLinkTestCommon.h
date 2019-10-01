//===---- JITLinkTestCommon.h - Utilities for Orc Unit Tests ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities for JITLink unit tests.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_JITLINK_JITLINKTESTCOMMON_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_JITLINK_JITLINKTESTCOMMON_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"

#include "gtest/gtest.h"

namespace llvm {

class JITLinkTestCommon {
public:

  class TestResources {
  public:
    static Expected<std::unique_ptr<TestResources>>
    Create(StringRef AsmSrc, StringRef TripleStr, bool PIC, bool LargeCodeModel,
           MCTargetOptions Options);

    MemoryBufferRef getTestObjectBufferRef() const;

    const MCDisassembler &getDisassembler() const { return *Dis; }

  private:
    TestResources(StringRef AsmSrc, StringRef TripleStr, bool PIC,
                  bool LargeCodeModel, MCTargetOptions Options, Error &Err);

    Error initializeTripleSpecifics(Triple &TT);
    void initializeTestSpecifics(StringRef AsmSource, const Triple &TT,
                                 bool PIC, bool LargeCodeModel);

    const Target *TheTarget = nullptr;
    SourceMgr SrcMgr;
    SmallVector<char, 0> ObjBuffer;
    raw_svector_ostream ObjStream;

    MCTargetOptions Options;
    std::unique_ptr<MCRegisterInfo> MRI;
    std::unique_ptr<MCAsmInfo> MAI;
    std::unique_ptr<MCInstrInfo> MCII;
    std::unique_ptr<MCSubtargetInfo> STI;

    MCObjectFileInfo MOFI;
    std::unique_ptr<MCContext> AsCtx;
    std::unique_ptr<MCStreamer> MOS;

    std::unique_ptr<MCContext> DisCtx;
    std::unique_ptr<const MCDisassembler> Dis;
  };

  class TestJITLinkContext : public jitlink::JITLinkContext {
  public:
    using TestCaseFunction = std::function<void(jitlink::AtomGraph &)>;

    using NotifyResolvedFunction = std::function<void(jitlink::AtomGraph &G)>;

    using NotifyFinalizedFunction = std::function<void(
        std::unique_ptr<jitlink::JITLinkMemoryManager::Allocation>)>;

    TestJITLinkContext(TestResources &TR, TestCaseFunction TestCase);

    StringMap<JITEvaluatedSymbol> &externals() { return Externals; }

    TestJITLinkContext &
    setNotifyResolved(NotifyResolvedFunction NotifyResolved);

    TestJITLinkContext &
    setNotifyFinalized(NotifyFinalizedFunction NotifyFinalized);

    TestJITLinkContext &
    setMemoryManager(std::unique_ptr<jitlink::JITLinkMemoryManager> MM);

    jitlink::JITLinkMemoryManager &getMemoryManager() override;

    MemoryBufferRef getObjectBuffer() const override;

    void notifyFailed(Error Err) override;

    void
    lookup(const DenseSet<StringRef> &Symbols,
           jitlink::JITLinkAsyncLookupContinuation LookupContinuation) override;

    void notifyResolved(jitlink::AtomGraph &G) override;

    void notifyFinalized(
        std::unique_ptr<jitlink::JITLinkMemoryManager::Allocation> A) override;

    Error modifyPassConfig(const Triple &TT,
                           jitlink::PassConfiguration &Config) override;

  private:
    TestResources &TR;
    TestCaseFunction TestCase;
    NotifyResolvedFunction NotifyResolved;
    NotifyFinalizedFunction NotifyFinalized;
    std::unique_ptr<MemoryBuffer> ObjBuffer;
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr;
    StringMap<JITEvaluatedSymbol> Externals;
  };

  JITLinkTestCommon();

  /// Get TestResources for this target/test.
  ///
  /// If this method fails it is likely because the target is not supported in
  /// this build. The test should bail out without failing (possibly logging a
  /// diagnostic).
  Expected<std::unique_ptr<TestResources>>
  getTestResources(StringRef AsmSrc, StringRef Triple, bool PIC,
                   bool LargeCodeModel, MCTargetOptions Options) const {
    return TestResources::Create(AsmSrc, Triple, PIC, LargeCodeModel,
                                 std::move(Options));
  }

  template <typename T>
  static Expected<T> readInt(jitlink::AtomGraph &G, jitlink::DefinedAtom &A,
                             size_t Offset = 0) {
    if (Offset + sizeof(T) > A.getContent().size())
      return make_error<StringError>("Reading past end of atom content",
                                     inconvertibleErrorCode());
    return support::endian::read<T, 1>(A.getContent().data() + Offset,
                                       G.getEndianness());
  }

  template <typename T>
  static Expected<T> readInt(jitlink::AtomGraph &G, StringRef AtomName,
                             size_t Offset = 0) {
    auto DA = G.findDefinedAtomByName(AtomName);
    if (!DA)
      return DA.takeError();
    return readInt<T>(G, *DA);
  }

  static Expected<std::pair<MCInst, size_t>>
  disassemble(const MCDisassembler &Dis, jitlink::DefinedAtom &Atom,
              size_t Offset = 0);

  static Expected<int64_t> decodeImmediateOperand(const MCDisassembler &Dis,
                                                  jitlink::DefinedAtom &Atom,
                                                  size_t OpIdx,
                                                  size_t Offset = 0);

  static jitlink::Atom &atom(jitlink::AtomGraph &G, StringRef Name) {
    return G.getAtomByName(Name);
  }

  static jitlink::DefinedAtom &definedAtom(jitlink::AtomGraph &G,
                                           StringRef Name) {
    return G.getDefinedAtomByName(Name);
  }

  static JITTargetAddress atomAddr(jitlink::AtomGraph &G, StringRef Name) {
    return atom(G, Name).getAddress();
  }

  template <typename PredT>
  static size_t countEdgesMatching(jitlink::DefinedAtom &DA,
                                   const PredT &Pred) {
    return std::count_if(DA.edges().begin(), DA.edges().end(), Pred);
  }

  template <typename PredT>
  static size_t countEdgesMatching(jitlink::AtomGraph &G, StringRef Name,
                                   const PredT &Pred) {
    return countEdgesMatching(definedAtom(G, Name), Pred);
  }

private:

  static bool AreTargetsInitialized;
  void initializeLLVMTargets();

  DenseMap<StringRef, JITEvaluatedSymbol> Externals;
};

} // end namespace llvm

#endif
