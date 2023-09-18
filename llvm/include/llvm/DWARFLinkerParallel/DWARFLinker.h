//===- DWARFLinker.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKERPARALLEL_DWARFLINKER_H
#define LLVM_DWARFLINKERPARALLEL_DWARFLINKER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DWARFLinkerParallel/DWARFFile.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/TargetParser/Triple.h"

/// ------------------------------------------------------------------
/// The core of the Dwarf linking logic.
///
/// The generation of the dwarf information from the object files will be
/// driven by the selection of 'root DIEs', which are DIEs that
/// describe variables or functions that resolves to the corresponding
/// code section(and thus have entries in the Addresses map). All the debug
/// information that will be generated(the DIEs, but also the line
/// tables, ranges, ...) is derived from that set of root DIEs.
///
/// The root DIEs are identified because they contain relocations that
/// points to code section(the low_pc for a function, the location for
/// a variable). These relocations are gathered as a very first step
/// when we start processing a object file by AddressesMap.
///
/// The overall linking process looks like this:
///
/// parrallel_for_each(ObjectFile) {
///   for_each (Compile Unit) {
///     1. Load Clang modules.
///   }
///
///   parrallel_for_each(Compile Unit) {
///     1. Load input DWARF for Compile Unit.
///     2. Report warnings for Clang modules.
///     3. Analyze live DIEs and type names(if ODR deduplication is requested).
///     4. Clone DIEs(Generate output DIEs and resulting DWARF tables).
///        The result is in an OutDebugInfoBytes, which is an ELF file
///        containing DWARF tables corresponding to the current compile unit.
///     5. Cleanup Input and Output DIEs.
///   }
///
///   Deallocate loaded Object file.
/// }
///
/// if (ODR deduplication is requested)
///   Generate an artificial compilation unit ("Type Table": used to partially
///   generate DIEs at the clone stage).
///
/// for_each (ObjectFile) {
///   for_each (Compile Unit) {
///     1. Set offsets to Compile Units DWARF tables.
///     2. Sort offsets/attributes/patches to have a predictable result.
///     3. Patch size/offsets fields.
///     4. Generate index tables.
///     5. Move DWARF tables of compile units into the resulting file.
///   }
/// }
///
/// Every compile unit is processed separately, visited only once
/// (except case inter-CU references exist), and used data is freed
/// after the compile unit is processed. The resulting file is glued together
/// from the generated debug tables which correspond to separate compile units.
///
/// Handling inter-CU references: inter-CU references are hard to process
/// using only one pass. f.e. if CU1 references CU100 and CU100 references
/// CU1, we could not finish handling of CU1 until we finished CU100.
/// Thus we either need to load all CUs into the memory, either load CUs several
/// times. This implementation loads inter-connected CU into memory at the first
/// pass and processes them at the second pass.
///
/// ODR deduplication: Artificial compilation unit will be constructed to keep
/// type dies. All types are moved into that compilation unit. Type's references
/// are patched so that they point to the corresponding types from artificial
/// compilation unit. All partial type definitions would be merged into single
/// type definition.
///

namespace llvm {
namespace dwarflinker_parallel {

/// ExtraDwarfEmitter allows adding extra data to the DWARFLinker output.
/// The finish() method should be called after all extra data are emitted.
class ExtraDwarfEmitter {
public:
  virtual ~ExtraDwarfEmitter() = default;

  /// Dump the file to the disk.
  virtual void finish() = 0;

  /// Emit section named SecName with data SecData.
  virtual void emitSectionContents(StringRef SecData, StringRef SecName) = 0;

  /// Emit temporarily symbol named \p SymName inside section \p SecName.
  virtual MCSymbol *emitTempSym(StringRef SecName, StringRef SymName) = 0;

  /// Emit the swift_ast section stored in \p Buffer.
  virtual void emitSwiftAST(StringRef Buffer) = 0;

  /// Emit the swift reflection section stored in \p Buffer.
  virtual void emitSwiftReflectionSection(
      llvm::binaryformat::Swift5ReflectionSectionKind ReflSectionKind,
      StringRef Buffer, uint32_t Alignment, uint32_t Size) = 0;

  /// Returns underlying AsmPrinter.
  virtual AsmPrinter &getAsmPrinter() const = 0;
};

class DWARFLinker {
public:
  /// Type of output file.
  enum class OutputFileType {
    Object,
    Assembly,
  };

  /// The kind of accelerator tables we should emit.
  enum class AccelTableKind : uint8_t {
    Apple,     ///< .apple_names, .apple_namespaces, .apple_types, .apple_objc.
    Pub,       ///< .debug_pubnames, .debug_pubtypes
    DebugNames ///< .debug_names.
  };

  using MessageHandlerTy = std::function<void(
      const Twine &Warning, StringRef Context, const DWARFDie *DIE)>;
  using ObjFileLoaderTy = std::function<ErrorOr<DWARFFile &>(
      StringRef ContainerName, StringRef Path)>;
  using InputVerificationHandlerTy = std::function<void(const DWARFFile &File, llvm::StringRef Output)>;
  using ObjectPrefixMapTy = std::map<std::string, std::string>;
  using CompileUnitHandlerTy = function_ref<void(const DWARFUnit &Unit)>;
  using TranslatorFuncTy = std::function<StringRef(StringRef)>;
  using SwiftInterfacesMapTy = std::map<std::string, std::string>;

  virtual ~DWARFLinker() = default;

  /// Creates dwarf linker instance.
  static std::unique_ptr<DWARFLinker>
  createLinker(MessageHandlerTy ErrorHandler, MessageHandlerTy WarningHandler,
               TranslatorFuncTy StringsTranslator = nullptr);

  /// Creates emitter for output dwarf.
  virtual Error createEmitter(const Triple &TheTriple, OutputFileType FileType,
                              raw_pwrite_stream &OutFile) = 0;

  /// Returns previously created dwarf emitter. May be nullptr.
  virtual ExtraDwarfEmitter *getEmitter() = 0;

  /// Add object file to be linked. Pre-load compile unit die. Call
  /// \p OnCUDieLoaded for each compile unit die. If specified \p File
  /// has reference to the Clang module then such module would be
  /// pre-loaded by \p Loader for !Update case.
  ///
  /// \pre NoODR, Update options should be set before call to addObjectFile.
  virtual void addObjectFile(
      DWARFFile &File, ObjFileLoaderTy Loader = nullptr,
      CompileUnitHandlerTy OnCUDieLoaded = [](const DWARFUnit &) {}) = 0;

  /// Link debug info for added files.
  virtual Error link() = 0;

  /// \defgroup Methods setting various linking options:
  ///
  /// @{

  /// Allows to generate log of linking process to the standard output.
  virtual void setVerbosity(bool Verbose) = 0;

  /// Print statistics to standard output.
  virtual void setStatistics(bool Statistics) = 0;

  /// Verify the input DWARF.
  virtual void setVerifyInputDWARF(bool Verify) = 0;

  /// Do not unique types according to ODR.
  virtual void setNoODR(bool NoODR) = 0;

  /// Update index tables only(do not modify rest of DWARF).
  virtual void setUpdateIndexTablesOnly(bool UpdateIndexTablesOnly) = 0;

  /// Allow generating valid, but non-deterministic output.
  virtual void
  setAllowNonDeterministicOutput(bool AllowNonDeterministicOutput) = 0;

  /// Set to keep the enclosing function for a static variable.
  virtual void setKeepFunctionForStatic(bool KeepFunctionForStatic) = 0;

  /// Use specified number of threads for parallel files linking.
  virtual void setNumThreads(unsigned NumThreads) = 0;

  /// Add kind of accelerator tables to be generated.
  virtual void addAccelTableKind(AccelTableKind Kind) = 0;

  /// Set prepend path for clang modules.
  virtual void setPrependPath(const std::string &Ppath) = 0;

  /// Set estimated objects files amount, for preliminary data allocation.
  virtual void setEstimatedObjfilesAmount(unsigned ObjFilesNum) = 0;

  /// Set verification handler which would be used to report verification
  /// errors.
  virtual void
  setInputVerificationHandler(InputVerificationHandlerTy Handler) = 0;

  /// Set map for Swift interfaces.
  virtual void setSwiftInterfacesMap(SwiftInterfacesMapTy *Map) = 0;

  /// Set prefix map for objects.
  virtual void setObjectPrefixMap(ObjectPrefixMapTy *Map) = 0;

  /// Set target DWARF version.
  virtual Error setTargetDWARFVersion(uint16_t TargetDWARFVersion) = 0;
  /// @}
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_DWARFLINKERPARALLEL_DWARFLINKER_H
