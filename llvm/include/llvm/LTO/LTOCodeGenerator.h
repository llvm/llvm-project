//===-LTOCodeGenerator.h - LLVM Link Time Optimizer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LTOCodeGenerator class.
//
//   LTO compilation consists of three phases: Pre-IPO, IPO and Post-IPO.
//
//   The Pre-IPO phase compiles source code into bitcode file. The resulting
// bitcode files, along with object files and libraries, will be fed to the
// linker to through the IPO and Post-IPO phases. By using obj-file extension,
// the resulting bitcode file disguises itself as an object file, and therefore
// obviates the need of writing a special set of the make-rules only for LTO
// compilation.
//
//   The IPO phase perform inter-procedural analyses and optimizations, and
// the Post-IPO consists two sub-phases: intra-procedural scalar optimizations
// (SOPT), and intra-procedural target-dependent code generator (CG).
//
//   As of this writing, we don't separate IPO and the Post-IPO SOPT. They
// are intermingled together, and are driven by a single pass manager (see
// PassManagerBuilder::populateLTOPassManager()).
//
//   The "LTOCodeGenerator" is the driver for the IPO and Post-IPO stages.
// The "CodeGenerator" here is bit confusing. Don't confuse the "CodeGenerator"
// with the machine specific code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_LTOCODEGENERATOR_H
#define LLVM_LTO_LTOCODEGENERATOR_H

#include "llvm-c/lto.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <string>
#include <vector>

namespace llvm {
  class LLVMContext;
  class DiagnosticInfo;
  class GlobalValue;
  class Linker;
  class Mangler;
  class MemoryBuffer;
  class TargetLibraryInfo;
  class TargetMachine;
  class raw_ostream;
  class raw_pwrite_stream;

//===----------------------------------------------------------------------===//
/// C++ class which implements the opaque lto_code_gen_t type.
///
struct LTOCodeGenerator {
  static const char *getVersionString();

  LTOCodeGenerator(LLVMContext &Context);
  ~LTOCodeGenerator();

  /// Merge given module.  Return true on success.
  bool addModule(struct LTOModule *);

  /// Set the destination module.
  void setModule(std::unique_ptr<LTOModule> M);

  void setTargetOptions(TargetOptions Options);
  void setDebugInfo(lto_debug_model);
  void setCodePICModel(Reloc::Model Model) { RelocModel = Model; }
  
  /// Set the file type to be emitted (assembly or object code).
  /// The default is TargetMachine::CGFT_ObjectFile. 
  void setFileType(TargetMachine::CodeGenFileType FT) { FileType = FT; }

  void setCpu(const char *MCpu) { this->MCpu = MCpu; }
  void setAttr(const char *MAttr) { this->MAttr = MAttr; }
  void setOptLevel(unsigned OptLevel);

  void setShouldInternalize(bool Value) { ShouldInternalize = Value; }
  void setShouldEmbedUselists(bool Value) { ShouldEmbedUselists = Value; }

  void addMustPreserveSymbol(StringRef Sym) { MustPreserveSymbols[Sym] = 1; }

  /// Pass options to the driver and optimization passes.
  ///
  /// These options are not necessarily for debugging purpose (the function
  /// name is misleading).  This function should be called before
  /// LTOCodeGenerator::compilexxx(), and
  /// LTOCodeGenerator::writeMergedModules().
  void setCodeGenDebugOptions(const char *Opts);

  /// Parse the options set in setCodeGenDebugOptions.
  ///
  /// Like \a setCodeGenDebugOptions(), this must be called before
  /// LTOCodeGenerator::compilexxx() and
  /// LTOCodeGenerator::writeMergedModules().
  void parseCodeGenDebugOptions();

  /// Write the merged module to the file specified by the given path.  Return
  /// true on success.
  bool writeMergedModules(const char *Path);

  /// Compile the merged module into a *single* output file; the path to output
  /// file is returned to the caller via argument "name". Return true on
  /// success.
  ///
  /// \note It is up to the linker to remove the intermediate output file.  Do
  /// not try to remove the object file in LTOCodeGenerator's destructor as we
  /// don't who (LTOCodeGenerator or the output file) will last longer.
  bool compile_to_file(const char **Name, bool DisableVerify,
                       bool DisableInline, bool DisableGVNLoadPRE,
                       bool DisableVectorization);

  /// As with compile_to_file(), this function compiles the merged module into
  /// single output file. Instead of returning the output file path to the
  /// caller (linker), it brings the output to a buffer, and returns the buffer
  /// to the caller. This function should delete the intermediate file once
  /// its content is brought to memory. Return NULL if the compilation was not
  /// successful.
  std::unique_ptr<MemoryBuffer> compile(bool DisableVerify, bool DisableInline,
                                        bool DisableGVNLoadPRE,
                                        bool DisableVectorization);

  /// Optimizes the merged module.  Returns true on success.
  bool optimize(bool DisableVerify, bool DisableInline, bool DisableGVNLoadPRE,
                bool DisableVectorization);

  /// Compiles the merged optimized module into a single output file. It brings
  /// the output to a buffer, and returns the buffer to the caller. Return NULL
  /// if the compilation was not successful.
  std::unique_ptr<MemoryBuffer> compileOptimized();

  /// Compile the merged optimized module into out.size() output files each
  /// representing a linkable partition of the module. If out contains more
  /// than one element, code generation is done in parallel with out.size()
  /// threads.  Output files will be written to members of out. Returns true on
  /// success.
  bool compileOptimized(ArrayRef<raw_pwrite_stream *> Out);

  void setDiagnosticHandler(lto_diagnostic_handler_t, void *);

  LLVMContext &getContext() { return Context; }

  void resetMergedModule() { MergedModule.reset(); }

private:
  void initializeLTOPasses();

  bool compileOptimizedToFile(const char **Name);
  void applyScopeRestrictions();
  void applyRestriction(GlobalValue &GV, ArrayRef<StringRef> Libcalls,
                        std::vector<const char *> &MustPreserveList,
                        SmallPtrSetImpl<GlobalValue *> &AsmUsed,
                        Mangler &Mangler);
  bool determineTarget();

  static void DiagnosticHandler(const DiagnosticInfo &DI, void *Context);

  void DiagnosticHandler2(const DiagnosticInfo &DI);

  void emitError(const std::string &ErrMsg);

  typedef StringMap<uint8_t> StringSet;

  LLVMContext &Context;
  std::unique_ptr<Module> MergedModule;
  std::unique_ptr<Linker> TheLinker;
  std::unique_ptr<TargetMachine> TargetMach;
  bool EmitDwarfDebugInfo = false;
  bool ScopeRestrictionsDone = false;
  Reloc::Model RelocModel = Reloc::Default;
  StringSet MustPreserveSymbols;
  StringSet AsmUndefinedRefs;
  std::vector<std::string> CodegenOptions;
  std::string FeatureStr;
  std::string MCpu;
  std::string MAttr;
  std::string NativeObjectPath;
  TargetOptions Options;
  CodeGenOpt::Level CGOptLevel = CodeGenOpt::Default;
  unsigned OptLevel = 2;
  lto_diagnostic_handler_t DiagHandler = nullptr;
  void *DiagContext = nullptr;
  bool ShouldInternalize = true;
  bool ShouldEmbedUselists = false;
  TargetMachine::CodeGenFileType FileType = TargetMachine::CGFT_ObjectFile;
};
}
#endif
