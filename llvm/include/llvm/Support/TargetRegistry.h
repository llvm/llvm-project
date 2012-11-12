//===-- Support/TargetRegistry.h - Target Registration ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the TargetRegistry interface, which tools can use to access
// the appropriate target specific classes (TargetMachine, AsmPrinter, etc.)
// which have been registered.
//
// Target specific class implementations should register themselves using the
// appropriate TargetRegistry interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TARGETREGISTRY_H
#define LLVM_SUPPORT_TARGETREGISTRY_H

#include "llvm/Support/CodeGen.h"
#include "llvm/ADT/Triple.h"
#include <string>
#include <cassert>

namespace llvm {
  class AsmPrinter;
  class Module;
  class MCAssembler;
  class MCAsmBackend;
  class MCAsmInfo;
  class MCAsmParser;
  class MCCodeEmitter;
  class MCCodeGenInfo;
  class MCContext;
  class MCDisassembler;
  class MCInstrAnalysis;
  class MCInstPrinter;
  class MCInstrInfo;
  class MCRegisterInfo;
  class MCStreamer;
  class MCSubtargetInfo;
  class MCTargetAsmLexer;
  class MCTargetAsmParser;
  class TargetMachine;
  class TargetOptions;
  class raw_ostream;
  class formatted_raw_ostream;

  MCStreamer *createAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                bool isVerboseAsm,
                                bool useLoc, bool useCFI,
                                bool useDwarfDirectory,
                                MCInstPrinter *InstPrint,
                                MCCodeEmitter *CE,
                                MCAsmBackend *TAB,
                                bool ShowInst);

  /// Target - Wrapper for Target specific information.
  ///
  /// For registration purposes, this is a POD type so that targets can be
  /// registered without the use of static constructors.
  ///
  /// Targets should implement a single global instance of this class (which
  /// will be zero initialized), and pass that instance to the TargetRegistry as
  /// part of their initialization.
  class Target {
  public:
    friend struct TargetRegistry;

    typedef unsigned (*TripleMatchQualityFnTy)(const std::string &TT);

    typedef MCAsmInfo *(*MCAsmInfoCtorFnTy)(const Target &T,
                                            StringRef TT);
    typedef MCCodeGenInfo *(*MCCodeGenInfoCtorFnTy)(StringRef TT,
                                                    Reloc::Model RM,
                                                    CodeModel::Model CM,
                                                    CodeGenOpt::Level OL);
    typedef MCInstrInfo *(*MCInstrInfoCtorFnTy)(void);
    typedef MCInstrAnalysis *(*MCInstrAnalysisCtorFnTy)(const MCInstrInfo*Info);
    typedef MCRegisterInfo *(*MCRegInfoCtorFnTy)(StringRef TT);
    typedef MCSubtargetInfo *(*MCSubtargetInfoCtorFnTy)(StringRef TT,
                                                        StringRef CPU,
                                                        StringRef Features);
    typedef TargetMachine *(*TargetMachineCtorTy)(const Target &T,
                                                  StringRef TT,
                                                  StringRef CPU,
                                                  StringRef Features,
                                                  const TargetOptions &Options,
                                                  Reloc::Model RM,
                                                  CodeModel::Model CM,
                                                  CodeGenOpt::Level OL);
    typedef AsmPrinter *(*AsmPrinterCtorTy)(TargetMachine &TM,
                                            MCStreamer &Streamer);
    typedef MCAsmBackend *(*MCAsmBackendCtorTy)(const Target &T,
                                                StringRef TT,
                                                StringRef CPU);
    typedef MCTargetAsmLexer *(*MCAsmLexerCtorTy)(const Target &T,
                                                  const MCRegisterInfo &MRI,
                                                  const MCAsmInfo &MAI);
    typedef MCTargetAsmParser *(*MCAsmParserCtorTy)(MCSubtargetInfo &STI,
                                                    MCAsmParser &P);
    typedef MCDisassembler *(*MCDisassemblerCtorTy)(const Target &T,
                                                    const MCSubtargetInfo &STI);
    typedef MCInstPrinter *(*MCInstPrinterCtorTy)(const Target &T,
                                                  unsigned SyntaxVariant,
                                                  const MCAsmInfo &MAI,
                                                  const MCInstrInfo &MII,
                                                  const MCRegisterInfo &MRI,
                                                  const MCSubtargetInfo &STI);
    typedef MCCodeEmitter *(*MCCodeEmitterCtorTy)(const MCInstrInfo &II,
                                                  const MCRegisterInfo &MRI,
                                                  const MCSubtargetInfo &STI,
                                                  MCContext &Ctx);
    typedef MCStreamer *(*MCObjectStreamerCtorTy)(const Target &T,
                                                  StringRef TT,
                                                  MCContext &Ctx,
                                                  MCAsmBackend &TAB,
                                                  raw_ostream &_OS,
                                                  MCCodeEmitter *_Emitter,
                                                  bool RelaxAll,
                                                  bool NoExecStack);
    typedef MCStreamer *(*AsmStreamerCtorTy)(MCContext &Ctx,
                                             formatted_raw_ostream &OS,
                                             bool isVerboseAsm,
                                             bool useLoc,
                                             bool useCFI,
                                             bool useDwarfDirectory,
                                             MCInstPrinter *InstPrint,
                                             MCCodeEmitter *CE,
                                             MCAsmBackend *TAB,
                                             bool ShowInst);

  private:
    /// Next - The next registered target in the linked list, maintained by the
    /// TargetRegistry.
    Target *Next;

    /// TripleMatchQualityFn - The target function for rating the match quality
    /// of a triple.
    TripleMatchQualityFnTy TripleMatchQualityFn;

    /// Name - The target name.
    const char *Name;

    /// ShortDesc - A short description of the target.
    const char *ShortDesc;

    /// HasJIT - Whether this target supports the JIT.
    bool HasJIT;

    /// MCAsmInfoCtorFn - Constructor function for this target's MCAsmInfo, if
    /// registered.
    MCAsmInfoCtorFnTy MCAsmInfoCtorFn;

    /// MCCodeGenInfoCtorFn - Constructor function for this target's
    /// MCCodeGenInfo, if registered.
    MCCodeGenInfoCtorFnTy MCCodeGenInfoCtorFn;

    /// MCInstrInfoCtorFn - Constructor function for this target's MCInstrInfo,
    /// if registered.
    MCInstrInfoCtorFnTy MCInstrInfoCtorFn;

    /// MCInstrAnalysisCtorFn - Constructor function for this target's
    /// MCInstrAnalysis, if registered.
    MCInstrAnalysisCtorFnTy MCInstrAnalysisCtorFn;

    /// MCRegInfoCtorFn - Constructor function for this target's MCRegisterInfo,
    /// if registered.
    MCRegInfoCtorFnTy MCRegInfoCtorFn;

    /// MCSubtargetInfoCtorFn - Constructor function for this target's
    /// MCSubtargetInfo, if registered.
    MCSubtargetInfoCtorFnTy MCSubtargetInfoCtorFn;

    /// TargetMachineCtorFn - Construction function for this target's
    /// TargetMachine, if registered.
    TargetMachineCtorTy TargetMachineCtorFn;

    /// MCAsmBackendCtorFn - Construction function for this target's
    /// MCAsmBackend, if registered.
    MCAsmBackendCtorTy MCAsmBackendCtorFn;

    /// MCAsmLexerCtorFn - Construction function for this target's
    /// MCTargetAsmLexer, if registered.
    MCAsmLexerCtorTy MCAsmLexerCtorFn;

    /// MCAsmParserCtorFn - Construction function for this target's
    /// MCTargetAsmParser, if registered.
    MCAsmParserCtorTy MCAsmParserCtorFn;

    /// AsmPrinterCtorFn - Construction function for this target's AsmPrinter,
    /// if registered.
    AsmPrinterCtorTy AsmPrinterCtorFn;

    /// MCDisassemblerCtorFn - Construction function for this target's
    /// MCDisassembler, if registered.
    MCDisassemblerCtorTy MCDisassemblerCtorFn;

    /// MCInstPrinterCtorFn - Construction function for this target's
    /// MCInstPrinter, if registered.
    MCInstPrinterCtorTy MCInstPrinterCtorFn;

    /// MCCodeEmitterCtorFn - Construction function for this target's
    /// CodeEmitter, if registered.
    MCCodeEmitterCtorTy MCCodeEmitterCtorFn;

    /// MCObjectStreamerCtorFn - Construction function for this target's
    /// MCObjectStreamer, if registered.
    MCObjectStreamerCtorTy MCObjectStreamerCtorFn;

    /// AsmStreamerCtorFn - Construction function for this target's
    /// AsmStreamer, if registered (default = llvm::createAsmStreamer).
    AsmStreamerCtorTy AsmStreamerCtorFn;

  public:
    Target() : AsmStreamerCtorFn(llvm::createAsmStreamer) {}

    /// @name Target Information
    /// @{

    // getNext - Return the next registered target.
    const Target *getNext() const { return Next; }

    /// getName - Get the target name.
    const char *getName() const { return Name; }

    /// getShortDescription - Get a short description of the target.
    const char *getShortDescription() const { return ShortDesc; }

    /// @}
    /// @name Feature Predicates
    /// @{

    /// hasJIT - Check if this targets supports the just-in-time compilation.
    bool hasJIT() const { return HasJIT; }

    /// hasTargetMachine - Check if this target supports code generation.
    bool hasTargetMachine() const { return TargetMachineCtorFn != 0; }

    /// hasMCAsmBackend - Check if this target supports .o generation.
    bool hasMCAsmBackend() const { return MCAsmBackendCtorFn != 0; }

    /// hasMCAsmLexer - Check if this target supports .s lexing.
    bool hasMCAsmLexer() const { return MCAsmLexerCtorFn != 0; }

    /// hasAsmParser - Check if this target supports .s parsing.
    bool hasMCAsmParser() const { return MCAsmParserCtorFn != 0; }

    /// hasAsmPrinter - Check if this target supports .s printing.
    bool hasAsmPrinter() const { return AsmPrinterCtorFn != 0; }

    /// hasMCDisassembler - Check if this target has a disassembler.
    bool hasMCDisassembler() const { return MCDisassemblerCtorFn != 0; }

    /// hasMCInstPrinter - Check if this target has an instruction printer.
    bool hasMCInstPrinter() const { return MCInstPrinterCtorFn != 0; }

    /// hasMCCodeEmitter - Check if this target supports instruction encoding.
    bool hasMCCodeEmitter() const { return MCCodeEmitterCtorFn != 0; }

    /// hasMCObjectStreamer - Check if this target supports streaming to files.
    bool hasMCObjectStreamer() const { return MCObjectStreamerCtorFn != 0; }

    /// hasAsmStreamer - Check if this target supports streaming to files.
    bool hasAsmStreamer() const { return AsmStreamerCtorFn != 0; }

    /// @}
    /// @name Feature Constructors
    /// @{

    /// createMCAsmInfo - Create a MCAsmInfo implementation for the specified
    /// target triple.
    ///
    /// \param Triple This argument is used to determine the target machine
    /// feature set; it should always be provided. Generally this should be
    /// either the target triple from the module, or the target triple of the
    /// host if that does not exist.
    MCAsmInfo *createMCAsmInfo(StringRef Triple) const {
      if (!MCAsmInfoCtorFn)
        return 0;
      return MCAsmInfoCtorFn(*this, Triple);
    }

    /// createMCCodeGenInfo - Create a MCCodeGenInfo implementation.
    ///
    MCCodeGenInfo *createMCCodeGenInfo(StringRef Triple, Reloc::Model RM,
                                       CodeModel::Model CM,
                                       CodeGenOpt::Level OL) const {
      if (!MCCodeGenInfoCtorFn)
        return 0;
      return MCCodeGenInfoCtorFn(Triple, RM, CM, OL);
    }

    /// createMCInstrInfo - Create a MCInstrInfo implementation.
    ///
    MCInstrInfo *createMCInstrInfo() const {
      if (!MCInstrInfoCtorFn)
        return 0;
      return MCInstrInfoCtorFn();
    }

    /// createMCInstrAnalysis - Create a MCInstrAnalysis implementation.
    ///
    MCInstrAnalysis *createMCInstrAnalysis(const MCInstrInfo *Info) const {
      if (!MCInstrAnalysisCtorFn)
        return 0;
      return MCInstrAnalysisCtorFn(Info);
    }

    /// createMCRegInfo - Create a MCRegisterInfo implementation.
    ///
    MCRegisterInfo *createMCRegInfo(StringRef Triple) const {
      if (!MCRegInfoCtorFn)
        return 0;
      return MCRegInfoCtorFn(Triple);
    }

    /// createMCSubtargetInfo - Create a MCSubtargetInfo implementation.
    ///
    /// \param Triple This argument is used to determine the target machine
    /// feature set; it should always be provided. Generally this should be
    /// either the target triple from the module, or the target triple of the
    /// host if that does not exist.
    /// \param CPU This specifies the name of the target CPU.
    /// \param Features This specifies the string representation of the
    /// additional target features.
    MCSubtargetInfo *createMCSubtargetInfo(StringRef Triple, StringRef CPU,
                                           StringRef Features) const {
      if (!MCSubtargetInfoCtorFn)
        return 0;
      return MCSubtargetInfoCtorFn(Triple, CPU, Features);
    }

    /// createTargetMachine - Create a target specific machine implementation
    /// for the specified \p Triple.
    ///
    /// \param Triple This argument is used to determine the target machine
    /// feature set; it should always be provided. Generally this should be
    /// either the target triple from the module, or the target triple of the
    /// host if that does not exist.
    TargetMachine *createTargetMachine(StringRef Triple, StringRef CPU,
                             StringRef Features, const TargetOptions &Options,
                             Reloc::Model RM = Reloc::Default,
                             CodeModel::Model CM = CodeModel::Default,
                             CodeGenOpt::Level OL = CodeGenOpt::Default) const {
      if (!TargetMachineCtorFn)
        return 0;
      return TargetMachineCtorFn(*this, Triple, CPU, Features, Options,
                                 RM, CM, OL);
    }

    /// createMCAsmBackend - Create a target specific assembly parser.
    ///
    /// \param Triple The target triple string.
    MCAsmBackend *createMCAsmBackend(StringRef Triple, StringRef CPU) const {
      if (!MCAsmBackendCtorFn)
        return 0;
      return MCAsmBackendCtorFn(*this, Triple, CPU);
    }

    /// createMCAsmLexer - Create a target specific assembly lexer.
    ///
    MCTargetAsmLexer *createMCAsmLexer(const MCRegisterInfo &MRI,
                                       const MCAsmInfo &MAI) const {
      if (!MCAsmLexerCtorFn)
        return 0;
      return MCAsmLexerCtorFn(*this, MRI, MAI);
    }

    /// createMCAsmParser - Create a target specific assembly parser.
    ///
    /// \param Parser The target independent parser implementation to use for
    /// parsing and lexing.
    MCTargetAsmParser *createMCAsmParser(MCSubtargetInfo &STI,
                                         MCAsmParser &Parser) const {
      if (!MCAsmParserCtorFn)
        return 0;
      return MCAsmParserCtorFn(STI, Parser);
    }

    /// createAsmPrinter - Create a target specific assembly printer pass.  This
    /// takes ownership of the MCStreamer object.
    AsmPrinter *createAsmPrinter(TargetMachine &TM, MCStreamer &Streamer) const{
      if (!AsmPrinterCtorFn)
        return 0;
      return AsmPrinterCtorFn(TM, Streamer);
    }

    MCDisassembler *createMCDisassembler(const MCSubtargetInfo &STI) const {
      if (!MCDisassemblerCtorFn)
        return 0;
      return MCDisassemblerCtorFn(*this, STI);
    }

    MCInstPrinter *createMCInstPrinter(unsigned SyntaxVariant,
                                       const MCAsmInfo &MAI,
                                       const MCInstrInfo &MII,
                                       const MCRegisterInfo &MRI,
                                       const MCSubtargetInfo &STI) const {
      if (!MCInstPrinterCtorFn)
        return 0;
      return MCInstPrinterCtorFn(*this, SyntaxVariant, MAI, MII, MRI, STI);
    }


    /// createMCCodeEmitter - Create a target specific code emitter.
    MCCodeEmitter *createMCCodeEmitter(const MCInstrInfo &II,
                                       const MCRegisterInfo &MRI,
                                       const MCSubtargetInfo &STI,
                                       MCContext &Ctx) const {
      if (!MCCodeEmitterCtorFn)
        return 0;
      return MCCodeEmitterCtorFn(II, MRI, STI, Ctx);
    }

    /// createMCObjectStreamer - Create a target specific MCStreamer.
    ///
    /// \param TT The target triple.
    /// \param Ctx The target context.
    /// \param TAB The target assembler backend object. Takes ownership.
    /// \param _OS The stream object.
    /// \param _Emitter The target independent assembler object.Takes ownership.
    /// \param RelaxAll Relax all fixups?
    /// \param NoExecStack Mark file as not needing a executable stack.
    MCStreamer *createMCObjectStreamer(StringRef TT, MCContext &Ctx,
                                       MCAsmBackend &TAB,
                                       raw_ostream &_OS,
                                       MCCodeEmitter *_Emitter,
                                       bool RelaxAll,
                                       bool NoExecStack) const {
      if (!MCObjectStreamerCtorFn)
        return 0;
      return MCObjectStreamerCtorFn(*this, TT, Ctx, TAB, _OS, _Emitter,
                                    RelaxAll, NoExecStack);
    }

    /// createAsmStreamer - Create a target specific MCStreamer.
    MCStreamer *createAsmStreamer(MCContext &Ctx,
                                  formatted_raw_ostream &OS,
                                  bool isVerboseAsm,
                                  bool useLoc,
                                  bool useCFI,
                                  bool useDwarfDirectory,
                                  MCInstPrinter *InstPrint,
                                  MCCodeEmitter *CE,
                                  MCAsmBackend *TAB,
                                  bool ShowInst) const {
      // AsmStreamerCtorFn is default to llvm::createAsmStreamer
      return AsmStreamerCtorFn(Ctx, OS, isVerboseAsm, useLoc, useCFI,
                               useDwarfDirectory, InstPrint, CE, TAB, ShowInst);
    }

    /// @}
  };

  /// TargetRegistry - Generic interface to target specific features.
  struct TargetRegistry {
    class iterator {
      const Target *Current;
      explicit iterator(Target *T) : Current(T) {}
      friend struct TargetRegistry;
    public:
      iterator(const iterator &I) : Current(I.Current) {}
      iterator() : Current(0) {}

      bool operator==(const iterator &x) const {
        return Current == x.Current;
      }
      bool operator!=(const iterator &x) const {
        return !operator==(x);
      }

      // Iterator traversal: forward iteration only
      iterator &operator++() {          // Preincrement
        assert(Current && "Cannot increment end iterator!");
        Current = Current->getNext();
        return *this;
      }
      iterator operator++(int) {        // Postincrement
        iterator tmp = *this;
        ++*this;
        return tmp;
      }

      const Target &operator*() const {
        assert(Current && "Cannot dereference end iterator!");
        return *Current;
      }

      const Target *operator->() const {
        return &operator*();
      }
    };

    /// printRegisteredTargetsForVersion - Print the registered targets
    /// appropriately for inclusion in a tool's version output.
    static void printRegisteredTargetsForVersion();

    /// @name Registry Access
    /// @{

    static iterator begin();

    static iterator end() { return iterator(); }

    /// lookupTarget - Lookup a target based on a target triple.
    ///
    /// \param Triple - The triple to use for finding a target.
    /// \param Error - On failure, an error string describing why no target was
    /// found.
    static const Target *lookupTarget(const std::string &Triple,
                                      std::string &Error);

    /// lookupTarget - Lookup a target based on an architecture name
    /// and a target triple.  If the architecture name is non-empty,
    /// then the lookup is done by architecture.  Otherwise, the target
    /// triple is used.
    ///
    /// \param ArchName - The architecture to use for finding a target.
    /// \param TheTriple - The triple to use for finding a target.  The
    /// triple is updated with canonical architecture name if a lookup
    /// by architecture is done.
    /// \param Error - On failure, an error string describing why no target was
    /// found.
    static const Target *lookupTarget(const std::string &ArchName,
                                      Triple &TheTriple,
                                      std::string &Error);

    /// getClosestTargetForJIT - Pick the best target that is compatible with
    /// the current host.  If no close target can be found, this returns null
    /// and sets the Error string to a reason.
    ///
    /// Maintained for compatibility through 2.6.
    static const Target *getClosestTargetForJIT(std::string &Error);

    /// @}
    /// @name Target Registration
    /// @{

    /// RegisterTarget - Register the given target. Attempts to register a
    /// target which has already been registered will be ignored.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Name - The target name. This should be a static string.
    /// @param ShortDesc - A short target description. This should be a static
    /// string.
    /// @param TQualityFn - The triple match quality computation function for
    /// this target.
    /// @param HasJIT - Whether the target supports JIT code
    /// generation.
    static void RegisterTarget(Target &T,
                               const char *Name,
                               const char *ShortDesc,
                               Target::TripleMatchQualityFnTy TQualityFn,
                               bool HasJIT = false);

    /// RegisterMCAsmInfo - Register a MCAsmInfo implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a MCAsmInfo for the target.
    static void RegisterMCAsmInfo(Target &T, Target::MCAsmInfoCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.MCAsmInfoCtorFn)
        T.MCAsmInfoCtorFn = Fn;
    }

    /// RegisterMCCodeGenInfo - Register a MCCodeGenInfo implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a MCCodeGenInfo for the target.
    static void RegisterMCCodeGenInfo(Target &T,
                                     Target::MCCodeGenInfoCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.MCCodeGenInfoCtorFn)
        T.MCCodeGenInfoCtorFn = Fn;
    }

    /// RegisterMCInstrInfo - Register a MCInstrInfo implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a MCInstrInfo for the target.
    static void RegisterMCInstrInfo(Target &T, Target::MCInstrInfoCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.MCInstrInfoCtorFn)
        T.MCInstrInfoCtorFn = Fn;
    }

    /// RegisterMCInstrAnalysis - Register a MCInstrAnalysis implementation for
    /// the given target.
    static void RegisterMCInstrAnalysis(Target &T,
                                        Target::MCInstrAnalysisCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.MCInstrAnalysisCtorFn)
        T.MCInstrAnalysisCtorFn = Fn;
    }

    /// RegisterMCRegInfo - Register a MCRegisterInfo implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a MCRegisterInfo for the target.
    static void RegisterMCRegInfo(Target &T, Target::MCRegInfoCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.MCRegInfoCtorFn)
        T.MCRegInfoCtorFn = Fn;
    }

    /// RegisterMCSubtargetInfo - Register a MCSubtargetInfo implementation for
    /// the given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a MCSubtargetInfo for the target.
    static void RegisterMCSubtargetInfo(Target &T,
                                        Target::MCSubtargetInfoCtorFnTy Fn) {
      // Ignore duplicate registration.
      if (!T.MCSubtargetInfoCtorFn)
        T.MCSubtargetInfoCtorFn = Fn;
    }

    /// RegisterTargetMachine - Register a TargetMachine implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct a TargetMachine for the target.
    static void RegisterTargetMachine(Target &T,
                                      Target::TargetMachineCtorTy Fn) {
      // Ignore duplicate registration.
      if (!T.TargetMachineCtorFn)
        T.TargetMachineCtorFn = Fn;
    }

    /// RegisterMCAsmBackend - Register a MCAsmBackend implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an AsmBackend for the target.
    static void RegisterMCAsmBackend(Target &T, Target::MCAsmBackendCtorTy Fn) {
      if (!T.MCAsmBackendCtorFn)
        T.MCAsmBackendCtorFn = Fn;
    }

    /// RegisterMCAsmLexer - Register a MCTargetAsmLexer implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCAsmLexer for the target.
    static void RegisterMCAsmLexer(Target &T, Target::MCAsmLexerCtorTy Fn) {
      if (!T.MCAsmLexerCtorFn)
        T.MCAsmLexerCtorFn = Fn;
    }

    /// RegisterMCAsmParser - Register a MCTargetAsmParser implementation for
    /// the given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCTargetAsmParser for the target.
    static void RegisterMCAsmParser(Target &T, Target::MCAsmParserCtorTy Fn) {
      if (!T.MCAsmParserCtorFn)
        T.MCAsmParserCtorFn = Fn;
    }

    /// RegisterAsmPrinter - Register an AsmPrinter implementation for the given
    /// target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an AsmPrinter for the target.
    static void RegisterAsmPrinter(Target &T, Target::AsmPrinterCtorTy Fn) {
      // Ignore duplicate registration.
      if (!T.AsmPrinterCtorFn)
        T.AsmPrinterCtorFn = Fn;
    }

    /// RegisterMCDisassembler - Register a MCDisassembler implementation for
    /// the given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCDisassembler for the target.
    static void RegisterMCDisassembler(Target &T,
                                       Target::MCDisassemblerCtorTy Fn) {
      if (!T.MCDisassemblerCtorFn)
        T.MCDisassemblerCtorFn = Fn;
    }

    /// RegisterMCInstPrinter - Register a MCInstPrinter implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCInstPrinter for the target.
    static void RegisterMCInstPrinter(Target &T,
                                      Target::MCInstPrinterCtorTy Fn) {
      if (!T.MCInstPrinterCtorFn)
        T.MCInstPrinterCtorFn = Fn;
    }

    /// RegisterMCCodeEmitter - Register a MCCodeEmitter implementation for the
    /// given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCCodeEmitter for the target.
    static void RegisterMCCodeEmitter(Target &T,
                                      Target::MCCodeEmitterCtorTy Fn) {
      if (!T.MCCodeEmitterCtorFn)
        T.MCCodeEmitterCtorFn = Fn;
    }

    /// RegisterMCObjectStreamer - Register a object code MCStreamer
    /// implementation for the given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCStreamer for the target.
    static void RegisterMCObjectStreamer(Target &T,
                                         Target::MCObjectStreamerCtorTy Fn) {
      if (!T.MCObjectStreamerCtorFn)
        T.MCObjectStreamerCtorFn = Fn;
    }

    /// RegisterAsmStreamer - Register an assembly MCStreamer implementation
    /// for the given target.
    ///
    /// Clients are responsible for ensuring that registration doesn't occur
    /// while another thread is attempting to access the registry. Typically
    /// this is done by initializing all targets at program startup.
    ///
    /// @param T - The target being registered.
    /// @param Fn - A function to construct an MCStreamer for the target.
    static void RegisterAsmStreamer(Target &T, Target::AsmStreamerCtorTy Fn) {
      if (T.AsmStreamerCtorFn == createAsmStreamer)
        T.AsmStreamerCtorFn = Fn;
    }

    /// @}
  };


  //===--------------------------------------------------------------------===//

  /// RegisterTarget - Helper template for registering a target, for use in the
  /// target's initialization function. Usage:
  ///
  ///
  /// Target TheFooTarget; // The global target instance.
  ///
  /// extern "C" void LLVMInitializeFooTargetInfo() {
  ///   RegisterTarget<Triple::foo> X(TheFooTarget, "foo", "Foo description");
  /// }
  template<Triple::ArchType TargetArchType = Triple::UnknownArch,
           bool HasJIT = false>
  struct RegisterTarget {
    RegisterTarget(Target &T, const char *Name, const char *Desc) {
      TargetRegistry::RegisterTarget(T, Name, Desc,
                                     &getTripleMatchQuality,
                                     HasJIT);
    }

    static unsigned getTripleMatchQuality(const std::string &TT) {
      if (Triple(TT).getArch() == TargetArchType)
        return 20;
      return 0;
    }
  };

  /// RegisterMCAsmInfo - Helper template for registering a target assembly info
  /// implementation.  This invokes the static "Create" method on the class to
  /// actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCAsmInfo<FooMCAsmInfo> X(TheFooTarget);
  /// }
  template<class MCAsmInfoImpl>
  struct RegisterMCAsmInfo {
    RegisterMCAsmInfo(Target &T) {
      TargetRegistry::RegisterMCAsmInfo(T, &Allocator);
    }
  private:
    static MCAsmInfo *Allocator(const Target &T, StringRef TT) {
      return new MCAsmInfoImpl(T, TT);
    }

  };

  /// RegisterMCAsmInfoFn - Helper template for registering a target assembly info
  /// implementation.  This invokes the specified function to do the
  /// construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCAsmInfoFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterMCAsmInfoFn {
    RegisterMCAsmInfoFn(Target &T, Target::MCAsmInfoCtorFnTy Fn) {
      TargetRegistry::RegisterMCAsmInfo(T, Fn);
    }
  };

  /// RegisterMCCodeGenInfo - Helper template for registering a target codegen info
  /// implementation.  This invokes the static "Create" method on the class
  /// to actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCCodeGenInfo<FooMCCodeGenInfo> X(TheFooTarget);
  /// }
  template<class MCCodeGenInfoImpl>
  struct RegisterMCCodeGenInfo {
    RegisterMCCodeGenInfo(Target &T) {
      TargetRegistry::RegisterMCCodeGenInfo(T, &Allocator);
    }
  private:
    static MCCodeGenInfo *Allocator(StringRef TT, Reloc::Model RM,
                                    CodeModel::Model CM, CodeGenOpt::Level OL) {
      return new MCCodeGenInfoImpl();
    }
  };

  /// RegisterMCCodeGenInfoFn - Helper template for registering a target codegen
  /// info implementation.  This invokes the specified function to do the
  /// construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCCodeGenInfoFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterMCCodeGenInfoFn {
    RegisterMCCodeGenInfoFn(Target &T, Target::MCCodeGenInfoCtorFnTy Fn) {
      TargetRegistry::RegisterMCCodeGenInfo(T, Fn);
    }
  };

  /// RegisterMCInstrInfo - Helper template for registering a target instruction
  /// info implementation.  This invokes the static "Create" method on the class
  /// to actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCInstrInfo<FooMCInstrInfo> X(TheFooTarget);
  /// }
  template<class MCInstrInfoImpl>
  struct RegisterMCInstrInfo {
    RegisterMCInstrInfo(Target &T) {
      TargetRegistry::RegisterMCInstrInfo(T, &Allocator);
    }
  private:
    static MCInstrInfo *Allocator() {
      return new MCInstrInfoImpl();
    }
  };

  /// RegisterMCInstrInfoFn - Helper template for registering a target
  /// instruction info implementation.  This invokes the specified function to
  /// do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCInstrInfoFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterMCInstrInfoFn {
    RegisterMCInstrInfoFn(Target &T, Target::MCInstrInfoCtorFnTy Fn) {
      TargetRegistry::RegisterMCInstrInfo(T, Fn);
    }
  };

  /// RegisterMCInstrAnalysis - Helper template for registering a target
  /// instruction analyzer implementation.  This invokes the static "Create"
  /// method on the class to actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCInstrAnalysis<FooMCInstrAnalysis> X(TheFooTarget);
  /// }
  template<class MCInstrAnalysisImpl>
  struct RegisterMCInstrAnalysis {
    RegisterMCInstrAnalysis(Target &T) {
      TargetRegistry::RegisterMCInstrAnalysis(T, &Allocator);
    }
  private:
    static MCInstrAnalysis *Allocator(const MCInstrInfo *Info) {
      return new MCInstrAnalysisImpl(Info);
    }
  };

  /// RegisterMCInstrAnalysisFn - Helper template for registering a target
  /// instruction analyzer implementation.  This invokes the specified function
  /// to do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCInstrAnalysisFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterMCInstrAnalysisFn {
    RegisterMCInstrAnalysisFn(Target &T, Target::MCInstrAnalysisCtorFnTy Fn) {
      TargetRegistry::RegisterMCInstrAnalysis(T, Fn);
    }
  };

  /// RegisterMCRegInfo - Helper template for registering a target register info
  /// implementation.  This invokes the static "Create" method on the class to
  /// actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCRegInfo<FooMCRegInfo> X(TheFooTarget);
  /// }
  template<class MCRegisterInfoImpl>
  struct RegisterMCRegInfo {
    RegisterMCRegInfo(Target &T) {
      TargetRegistry::RegisterMCRegInfo(T, &Allocator);
    }
  private:
    static MCRegisterInfo *Allocator(StringRef TT) {
      return new MCRegisterInfoImpl();
    }
  };

  /// RegisterMCRegInfoFn - Helper template for registering a target register
  /// info implementation.  This invokes the specified function to do the
  /// construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCRegInfoFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterMCRegInfoFn {
    RegisterMCRegInfoFn(Target &T, Target::MCRegInfoCtorFnTy Fn) {
      TargetRegistry::RegisterMCRegInfo(T, Fn);
    }
  };

  /// RegisterMCSubtargetInfo - Helper template for registering a target
  /// subtarget info implementation.  This invokes the static "Create" method
  /// on the class to actually do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCSubtargetInfo<FooMCSubtargetInfo> X(TheFooTarget);
  /// }
  template<class MCSubtargetInfoImpl>
  struct RegisterMCSubtargetInfo {
    RegisterMCSubtargetInfo(Target &T) {
      TargetRegistry::RegisterMCSubtargetInfo(T, &Allocator);
    }
  private:
    static MCSubtargetInfo *Allocator(StringRef TT, StringRef CPU,
                                      StringRef FS) {
      return new MCSubtargetInfoImpl();
    }
  };

  /// RegisterMCSubtargetInfoFn - Helper template for registering a target
  /// subtarget info implementation.  This invokes the specified function to
  /// do the construction.  Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCSubtargetInfoFn X(TheFooTarget, TheFunction);
  /// }
  struct RegisterMCSubtargetInfoFn {
    RegisterMCSubtargetInfoFn(Target &T, Target::MCSubtargetInfoCtorFnTy Fn) {
      TargetRegistry::RegisterMCSubtargetInfo(T, Fn);
    }
  };

  /// RegisterTargetMachine - Helper template for registering a target machine
  /// implementation, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooTarget() {
  ///   extern Target TheFooTarget;
  ///   RegisterTargetMachine<FooTargetMachine> X(TheFooTarget);
  /// }
  template<class TargetMachineImpl>
  struct RegisterTargetMachine {
    RegisterTargetMachine(Target &T) {
      TargetRegistry::RegisterTargetMachine(T, &Allocator);
    }

  private:
    static TargetMachine *Allocator(const Target &T, StringRef TT,
                                    StringRef CPU, StringRef FS,
                                    const TargetOptions &Options,
                                    Reloc::Model RM,
                                    CodeModel::Model CM,
                                    CodeGenOpt::Level OL) {
      return new TargetMachineImpl(T, TT, CPU, FS, Options, RM, CM, OL);
    }
  };

  /// RegisterMCAsmBackend - Helper template for registering a target specific
  /// assembler backend. Usage:
  ///
  /// extern "C" void LLVMInitializeFooMCAsmBackend() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCAsmBackend<FooAsmLexer> X(TheFooTarget);
  /// }
  template<class MCAsmBackendImpl>
  struct RegisterMCAsmBackend {
    RegisterMCAsmBackend(Target &T) {
      TargetRegistry::RegisterMCAsmBackend(T, &Allocator);
    }

  private:
    static MCAsmBackend *Allocator(const Target &T, StringRef Triple,
                                   StringRef CPU) {
      return new MCAsmBackendImpl(T, Triple, CPU);
    }
  };

  /// RegisterMCAsmLexer - Helper template for registering a target specific
  /// assembly lexer, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooMCAsmLexer() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCAsmLexer<FooMCAsmLexer> X(TheFooTarget);
  /// }
  template<class MCAsmLexerImpl>
  struct RegisterMCAsmLexer {
    RegisterMCAsmLexer(Target &T) {
      TargetRegistry::RegisterMCAsmLexer(T, &Allocator);
    }

  private:
    static MCTargetAsmLexer *Allocator(const Target &T,
                                       const MCRegisterInfo &MRI,
                                       const MCAsmInfo &MAI) {
      return new MCAsmLexerImpl(T, MRI, MAI);
    }
  };

  /// RegisterMCAsmParser - Helper template for registering a target specific
  /// assembly parser, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooMCAsmParser() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCAsmParser<FooAsmParser> X(TheFooTarget);
  /// }
  template<class MCAsmParserImpl>
  struct RegisterMCAsmParser {
    RegisterMCAsmParser(Target &T) {
      TargetRegistry::RegisterMCAsmParser(T, &Allocator);
    }

  private:
    static MCTargetAsmParser *Allocator(MCSubtargetInfo &STI, MCAsmParser &P) {
      return new MCAsmParserImpl(STI, P);
    }
  };

  /// RegisterAsmPrinter - Helper template for registering a target specific
  /// assembly printer, for use in the target machine initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooAsmPrinter() {
  ///   extern Target TheFooTarget;
  ///   RegisterAsmPrinter<FooAsmPrinter> X(TheFooTarget);
  /// }
  template<class AsmPrinterImpl>
  struct RegisterAsmPrinter {
    RegisterAsmPrinter(Target &T) {
      TargetRegistry::RegisterAsmPrinter(T, &Allocator);
    }

  private:
    static AsmPrinter *Allocator(TargetMachine &TM, MCStreamer &Streamer) {
      return new AsmPrinterImpl(TM, Streamer);
    }
  };

  /// RegisterMCCodeEmitter - Helper template for registering a target specific
  /// machine code emitter, for use in the target initialization
  /// function. Usage:
  ///
  /// extern "C" void LLVMInitializeFooMCCodeEmitter() {
  ///   extern Target TheFooTarget;
  ///   RegisterMCCodeEmitter<FooCodeEmitter> X(TheFooTarget);
  /// }
  template<class MCCodeEmitterImpl>
  struct RegisterMCCodeEmitter {
    RegisterMCCodeEmitter(Target &T) {
      TargetRegistry::RegisterMCCodeEmitter(T, &Allocator);
    }

  private:
    static MCCodeEmitter *Allocator(const MCInstrInfo &II,
                                    const MCRegisterInfo &MRI,
                                    const MCSubtargetInfo &STI,
                                    MCContext &Ctx) {
      return new MCCodeEmitterImpl();
    }
  };

}

#endif
