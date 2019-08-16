//===------- JITLinkTestCommon.cpp - Common code for JITLink tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestCommon.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm::jitlink;
namespace llvm {

Expected<std::unique_ptr<JITLinkTestCommon::TestResources>>
JITLinkTestCommon::TestResources::Create(StringRef AsmSrc, StringRef TripleStr,
                                         bool PIC, bool LargeCodeModel,
                                         MCTargetOptions Options) {
  Error Err = Error::success();
  auto R = std::unique_ptr<TestResources>(new TestResources(
      AsmSrc, TripleStr, PIC, LargeCodeModel, std::move(Options), Err));
  if (Err)
    return std::move(Err);
  return std::move(R);
}

MemoryBufferRef
JITLinkTestCommon::TestResources::getTestObjectBufferRef() const {
  return MemoryBufferRef(StringRef(ObjBuffer.data(), ObjBuffer.size()),
                         "Test object");
}

JITLinkTestCommon::TestResources::TestResources(StringRef AsmSrc,
                                                StringRef TripleStr, bool PIC,
                                                bool LargeCodeModel,
                                                MCTargetOptions Options,
                                                Error &Err)
    : ObjStream(ObjBuffer), Options(std::move(Options)) {
  ErrorAsOutParameter _(&Err);
  Triple TT(Triple::normalize(TripleStr));
  if (auto Err2 = initializeTripleSpecifics(TT)) {
    Err = std::move(Err2);
    return;
  }
  initializeTestSpecifics(AsmSrc, TT, PIC, LargeCodeModel);
}

Error JITLinkTestCommon::TestResources::initializeTripleSpecifics(Triple &TT) {
  std::string ErrorMsg;
  TheTarget = TargetRegistry::lookupTarget("", TT, ErrorMsg);

  if (!TheTarget)
    return make_error<StringError>(ErrorMsg, inconvertibleErrorCode());

  MRI.reset(TheTarget->createMCRegInfo(TT.getTriple()));
  if (!MRI)
    report_fatal_error("Could not build MCRegisterInfo for triple");

  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT.getTriple()));
  if (!MAI)
    report_fatal_error("Could not build MCAsmInfo for triple");

  MCII.reset(TheTarget->createMCInstrInfo());
  if (!MCII)
    report_fatal_error("Could not build MCInstrInfo for triple");

  STI.reset(TheTarget->createMCSubtargetInfo(TT.getTriple(), "", ""));
  if (!STI)
    report_fatal_error("Could not build MCSubtargetInfo for triple");

  DisCtx = llvm::make_unique<MCContext>(MAI.get(), MRI.get(), nullptr);
  Dis.reset(TheTarget->createMCDisassembler(*STI, *DisCtx));

  if (!Dis)
    report_fatal_error("Could not build MCDisassembler");

  return Error::success();
}

void JITLinkTestCommon::TestResources::initializeTestSpecifics(
    StringRef AsmSrc, const Triple &TT, bool PIC, bool LargeCodeModel) {
  SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(AsmSrc), SMLoc());
  AsCtx = llvm::make_unique<MCContext>(MAI.get(), MRI.get(), &MOFI, &SrcMgr);
  MOFI.InitMCObjectFileInfo(TT, PIC, *AsCtx, LargeCodeModel);

  std::unique_ptr<MCCodeEmitter> CE(
      TheTarget->createMCCodeEmitter(*MCII, *MRI, *AsCtx));
  if (!CE)
    report_fatal_error("Could not build MCCodeEmitter");

  std::unique_ptr<MCAsmBackend> MAB(
      TheTarget->createMCAsmBackend(*STI, *MRI, Options));
  if (!MAB)
    report_fatal_error("Could not build MCAsmBackend for test");

  std::unique_ptr<MCObjectWriter> MOW(MAB->createObjectWriter(ObjStream));

  MOS.reset(TheTarget->createMCObjectStreamer(
      TT, *AsCtx, std::move(MAB), std::move(MOW), std::move(CE), *STI,
      Options.MCRelaxAll, Options.MCIncrementalLinkerCompatible, false));

  std::unique_ptr<MCAsmParser> MAP(
      createMCAsmParser(SrcMgr, *AsCtx, *MOS, *MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *MAP, *MCII, Options));

  if (!TAP)
    report_fatal_error("Could not build MCTargetAsmParser for test");

  MAP->setTargetParser(*TAP);

  if (MAP->Run(false))
    report_fatal_error("Failed to parse test case");
}

JITLinkTestCommon::TestJITLinkContext::TestJITLinkContext(
    TestResources &TR, TestCaseFunction TestCase)
    : TR(TR), TestCase(std::move(TestCase)) {}

JITLinkTestCommon::TestJITLinkContext &
JITLinkTestCommon::TestJITLinkContext::setMemoryManager(
    std::unique_ptr<JITLinkMemoryManager> MM) {
  assert(!MemMgr && "Memory manager already set");
  MemMgr = std::move(MM);
  return *this;
}

JITLinkMemoryManager &
JITLinkTestCommon::TestJITLinkContext::getMemoryManager() {
  if (!MemMgr)
    MemMgr = llvm::make_unique<InProcessMemoryManager>();
  return *MemMgr;
}

MemoryBufferRef JITLinkTestCommon::TestJITLinkContext::getObjectBuffer() const {
  return TR.getTestObjectBufferRef();
}

void JITLinkTestCommon::TestJITLinkContext::notifyFailed(Error Err) {
  ADD_FAILURE() << "Unexpected failure: " << toString(std::move(Err));
}

void JITLinkTestCommon::TestJITLinkContext::lookup(
    const DenseSet<StringRef> &Symbols,
    JITLinkAsyncLookupContinuation LookupContinuation) {
  jitlink::AsyncLookupResult LookupResult;
  DenseSet<StringRef> MissingSymbols;
  for (const auto &Symbol : Symbols) {
    auto I = Externals.find(Symbol);
    if (I != Externals.end())
      LookupResult[Symbol] = I->second;
    else
      MissingSymbols.insert(Symbol);
  }

  if (MissingSymbols.empty())
    LookupContinuation(std::move(LookupResult));
  else {
    std::string ErrMsg;
    {
      raw_string_ostream ErrMsgStream(ErrMsg);
      ErrMsgStream << "Failed to resolve external symbols: [";
      for (auto &Sym : MissingSymbols)
        ErrMsgStream << " " << Sym;
      ErrMsgStream << " ]\n";
    }
    LookupContinuation(
        make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode()));
  }
}

void JITLinkTestCommon::TestJITLinkContext::notifyResolved(AtomGraph &G) {
  if (NotifyResolved)
    NotifyResolved(G);
}

void JITLinkTestCommon::TestJITLinkContext::notifyFinalized(
    std::unique_ptr<JITLinkMemoryManager::Allocation> A) {
  if (NotifyFinalized)
    NotifyFinalized(std::move(A));
}

Error JITLinkTestCommon::TestJITLinkContext::modifyPassConfig(
    const Triple &TT, PassConfiguration &Config) {
  if (TestCase)
    Config.PostFixupPasses.push_back([&](AtomGraph &G) -> Error {
      TestCase(G);
      return Error::success();
    });
  return Error::success();
}

JITLinkTestCommon::JITLinkTestCommon() { initializeLLVMTargets(); }

Expected<std::pair<MCInst, size_t>>
JITLinkTestCommon::disassemble(const MCDisassembler &Dis,
                               jitlink::DefinedAtom &Atom, size_t Offset) {
  ArrayRef<uint8_t> InstBuffer(
      reinterpret_cast<const uint8_t *>(Atom.getContent().data()) + Offset,
      Atom.getContent().size() - Offset);

  MCInst Inst;
  uint64_t InstSize;
  auto Status =
      Dis.getInstruction(Inst, InstSize, InstBuffer, 0, nulls(), nulls());

  if (Status != MCDisassembler::Success)
    return make_error<StringError>("Could not disassemble instruction",
                                   inconvertibleErrorCode());

  return std::make_pair(Inst, InstSize);
}

Expected<int64_t>
JITLinkTestCommon::decodeImmediateOperand(const MCDisassembler &Dis,
                                          jitlink::DefinedAtom &Atom,
                                          size_t OpIdx, size_t Offset) {
  auto InstAndSize = disassemble(Dis, Atom, Offset);
  if (!InstAndSize)
    return InstAndSize.takeError();

  if (OpIdx >= InstAndSize->first.getNumOperands())
    return make_error<StringError>("Invalid operand index",
                                   inconvertibleErrorCode());

  auto &Op = InstAndSize->first.getOperand(OpIdx);

  if (!Op.isImm())
    return make_error<StringError>("Operand at index is not immediate",
                                   inconvertibleErrorCode());

  return Op.getImm();
}

bool JITLinkTestCommon::AreTargetsInitialized = false;

void JITLinkTestCommon::initializeLLVMTargets() {
  if (!AreTargetsInitialized) {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
    InitializeAllDisassemblers();
    AreTargetsInitialized = true;
  }
}

} // end namespace llvm
