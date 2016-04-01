//===-ThinLTOCodeGenerator.h - LLVM Link Time Optimizer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ThinLTOCodeGenerator class, similar to the
// LTOCodeGenerator but for the ThinLTO scheme. It provides an interface for
// linker plugin.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_THINLTOCODEGENERATOR_H
#define LLVM_LTO_THINLTOCODEGENERATOR_H

#include "llvm-c/lto.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetOptions.h"

#include <string>

namespace llvm {
class ModuleSummaryIndex;
class LLVMContext;
class TargetMachine;

/// Helper to gather options relevant to the target machine creation
struct TargetMachineBuilder {
  Triple TheTriple;
  std::string MCpu;
  std::string MAttr;
  TargetOptions Options;
  Reloc::Model RelocModel = Reloc::Default;
  CodeGenOpt::Level CGOptLevel = CodeGenOpt::Default;

  std::unique_ptr<TargetMachine> create() const;
};

/// This class define an interface similar to the LTOCodeGenerator, but adapted
/// for ThinLTO processing.
/// The ThinLTOCodeGenerator is not intended to be reuse for multiple
/// compilation: the model is that the client adds modules to the generator and
/// ask to perform the ThinLTO optimizations / codegen, and finally destroys the
/// codegenerator.
class ThinLTOCodeGenerator {
public:
  /// Add given module to the code generator.
  void addModule(StringRef Identifier, StringRef Data);

  /**
   * Adds to a list of all global symbols that must exist in the final generated
   * code. If a symbol is not listed there, it will be optimized away if it is
   * inlined into every usage.
   */
  void preserveSymbol(StringRef Name);

  /**
   * Adds to a list of all global symbols that are cross-referenced between
   * ThinLTO files. If the ThinLTO CodeGenerator can ensure that every
   * references from a ThinLTO module to this symbol is optimized away, then
   * the symbol can be discarded.
   */
  void crossReferenceSymbol(StringRef Name);

  /**
   * Process all the modules that were added to the code generator in parallel.
   *
   * Client can access the resulting object files using getProducedBinaries()
   */
  void run();

  /**
   * Return the "in memory" binaries produced by the code generator.
   */
  std::vector<std::unique_ptr<MemoryBuffer>> &getProducedBinaries() {
    return ProducedBinaries;
  }

  /**
   * \defgroup Options setters
   * @{
   */

  /**
   * \defgroup Cache controlling options
   *
   * These entry points control the ThinLTO cache. The cache is intended to
   * support incremental build, and thus needs to be persistent accross build.
   * The client enabled the cache by supplying a path to an existing directory.
   * The code generator will use this to store objects files that may be reused
   * during a subsequent build.
   * To avoid filling the disk space, a few knobs are provided:
   *  - The pruning interval limit the frequency at which the garbage collector
   *    will try to scan the cache directory to prune it from expired entries.
   *    Setting to -1 disable the pruning (default).
   *  - The pruning expiration time indicates to the garbage collector how old
   *    an entry needs to be to be removed.
   *  - Finally, the garbage collector can be instructed to prune the cache till
   *    the occupied space goes below a threshold.
   * @{
   */

  struct CachingOptions {
    std::string Path;
    int PruningInterval = -1;               // seconds, -1 to disable pruning
    unsigned int Expiration;                // seconds.
    unsigned MaxPercentageOfAvailableSpace; // percentage.
  };

  /// Provide a path to a directory where to store the cached files for
  /// incremental build.
  void setCacheDir(std::string Path) { CacheOptions.Path = std::move(Path); }

  /// Cache policy: interval (seconds) between two prune of the cache. Set to a
  /// negative value (default) to disable pruning.
  void setCachePruningInterval(int Interval) {
    CacheOptions.PruningInterval = Interval;
  }

  /// Cache policy: expiration (in seconds) for an entry.
  void setCacheEntryExpiration(unsigned Expiration) {
    CacheOptions.Expiration = Expiration;
  }

  /**
   * Sets the maximum cache size that can be persistent across build, in terms
   * of percentage of the available space on the the disk. Set to 100 to
   * indicate no limit, 50 to indicate that the cache size will not be left over
   * half the available space. A value over 100 will be reduced to 100.
   *
   * The formula looks like:
   *  AvailableSpace = FreeSpace + ExistingCacheSize
   *  NewCacheSize = AvailableSpace * P/100
   *
   */
  void setMaxCacheSizeRelativeToAvailableSpace(unsigned Percentage) {
    CacheOptions.MaxPercentageOfAvailableSpace = Percentage;
  }

  /**@}*/

  /// Set the path to a directory where to save temporaries at various stages of
  /// the processing.
  void setSaveTempsDir(std::string Path) { SaveTempsDir = std::move(Path); }

  /// CPU to use to initialize the TargetMachine
  void setCpu(std::string Cpu) { TMBuilder.MCpu = std::move(Cpu); }

  /// Subtarget attributes
  void setAttr(std::string MAttr) { TMBuilder.MAttr = std::move(MAttr); }

  /// TargetMachine options
  void setTargetOptions(TargetOptions Options) {
    TMBuilder.Options = std::move(Options);
  }

  /// CodeModel
  void setCodePICModel(Reloc::Model Model) { TMBuilder.RelocModel = Model; }

  /// CodeGen optimization level
  void setCodeGenOptLevel(CodeGenOpt::Level CGOptLevel) {
    TMBuilder.CGOptLevel = CGOptLevel;
  }

  /// Disable CodeGen, only run the stages till codegen and stop. The output
  /// will be bitcode.
  void disableCodeGen(bool Disable) { DisableCodeGen = Disable; }

  /// Perform CodeGen only: disable all other stages.
  void setCodeGenOnly(bool CGOnly) { CodeGenOnly = CGOnly; }

  /**@}*/

  /**
   * \defgroup Set of APIs to run individual stages in isolation.
   * @{
   */

  /**
   * Produce the combined summary index from all the bitcode files:
   * "thin-link".
   */
  std::unique_ptr<ModuleSummaryIndex> linkCombinedIndex();

  /**
   * Perform promotion and renaming of exported internal functions.
   */
  void promote(Module &Module, ModuleSummaryIndex &Index);

  /**
   * Perform cross-module importing for the module identified by
   * ModuleIdentifier.
   */
  void crossModuleImport(Module &Module, ModuleSummaryIndex &Index);

  /**
   * Perform post-importing ThinLTO optimizations.
   */
  void optimize(Module &Module);

  /**
   * Perform ThinLTO CodeGen.
   */
  std::unique_ptr<MemoryBuffer> codegen(Module &Module);

  /**@}*/

private:
  /// Helper factory to build a TargetMachine
  TargetMachineBuilder TMBuilder;

  /// Vector holding the in-memory buffer containing the produced binaries.
  std::vector<std::unique_ptr<MemoryBuffer>> ProducedBinaries;

  /// Vector holding the input buffers containing the bitcode modules to
  /// process.
  std::vector<MemoryBufferRef> Modules;

  /// Set of symbols that need to be preserved outside of the set of bitcode
  /// files.
  StringSet<> PreservedSymbols;

  /// Set of symbols that are cross-referenced between bitcode files.
  StringSet<> CrossReferencedSymbols;

  /// Control the caching behavior.
  CachingOptions CacheOptions;

  /// Path to a directory to save the temporary bitcode files.
  std::string SaveTempsDir;

  /// Flag to enable/disable CodeGen. When set to true, the process stops after
  /// optimizations and a bitcode is produced.
  bool DisableCodeGen = false;

  /// Flag to indicate that only the CodeGen will be performed, no cross-module
  /// importing or optimization.
  bool CodeGenOnly = false;
};
}
#endif
