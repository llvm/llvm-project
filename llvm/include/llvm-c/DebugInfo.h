//===------------ DebugInfo.h - LLVM C API Debug Info API -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the C API endpoints for generating DWARF Debug Info
///
/// Note: This interface is experimental. It is *NOT* stable, and may be
///       changed without warning.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_C_DEBUGINFO_H
#define LLVM_C_DEBUGINFO_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"
#include "llvm-c/Visibility.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCCoreDebugInfo Debug Information
 * @ingroup LLVMCCore
 *
 * @{
 */

/**
 * Debug info flags.
 */
typedef enum {
  LLVMDIFlagZero = 0,
  LLVMDIFlagPrivate = 1,
  LLVMDIFlagProtected = 2,
  LLVMDIFlagPublic = 3,
  LLVMDIFlagFwdDecl = 1 << 2,
  LLVMDIFlagAppleBlock = 1 << 3,
  LLVMDIFlagReservedBit4 = 1 << 4,
  LLVMDIFlagVirtual = 1 << 5,
  LLVMDIFlagArtificial = 1 << 6,
  LLVMDIFlagExplicit = 1 << 7,
  LLVMDIFlagPrototyped = 1 << 8,
  LLVMDIFlagObjcClassComplete = 1 << 9,
  LLVMDIFlagObjectPointer = 1 << 10,
  LLVMDIFlagVector = 1 << 11,
  LLVMDIFlagStaticMember = 1 << 12,
  LLVMDIFlagLValueReference = 1 << 13,
  LLVMDIFlagRValueReference = 1 << 14,
  LLVMDIFlagReserved = 1 << 15,
  LLVMDIFlagSingleInheritance = 1 << 16,
  LLVMDIFlagMultipleInheritance = 2 << 16,
  LLVMDIFlagVirtualInheritance = 3 << 16,
  LLVMDIFlagIntroducedVirtual = 1 << 18,
  LLVMDIFlagBitField = 1 << 19,
  LLVMDIFlagNoReturn = 1 << 20,
  LLVMDIFlagTypePassByValue = 1 << 22,
  LLVMDIFlagTypePassByReference = 1 << 23,
  LLVMDIFlagEnumClass = 1 << 24,
  LLVMDIFlagFixedEnum = LLVMDIFlagEnumClass, // Deprecated.
  LLVMDIFlagThunk = 1 << 25,
  LLVMDIFlagNonTrivial = 1 << 26,
  LLVMDIFlagBigEndian = 1 << 27,
  LLVMDIFlagLittleEndian = 1 << 28,
  LLVMDIFlagIndirectVirtualBase = (1 << 2) | (1 << 5),
  LLVMDIFlagAccessibility = LLVMDIFlagPrivate | LLVMDIFlagProtected |
                            LLVMDIFlagPublic,
  LLVMDIFlagPtrToMemberRep = LLVMDIFlagSingleInheritance |
                             LLVMDIFlagMultipleInheritance |
                             LLVMDIFlagVirtualInheritance
} LLVMDIFlags;

/**
 * Source languages known by DWARF.
 */
typedef enum {
  LLVMDWARFSourceLanguageC89,
  LLVMDWARFSourceLanguageC,
  LLVMDWARFSourceLanguageAda83,
  LLVMDWARFSourceLanguageC_plus_plus,
  LLVMDWARFSourceLanguageCobol74,
  LLVMDWARFSourceLanguageCobol85,
  LLVMDWARFSourceLanguageFortran77,
  LLVMDWARFSourceLanguageFortran90,
  LLVMDWARFSourceLanguagePascal83,
  LLVMDWARFSourceLanguageModula2,
  // New in DWARF v3:
  LLVMDWARFSourceLanguageJava,
  LLVMDWARFSourceLanguageC99,
  LLVMDWARFSourceLanguageAda95,
  LLVMDWARFSourceLanguageFortran95,
  LLVMDWARFSourceLanguagePLI,
  LLVMDWARFSourceLanguageObjC,
  LLVMDWARFSourceLanguageObjC_plus_plus,
  LLVMDWARFSourceLanguageUPC,
  LLVMDWARFSourceLanguageD,
  // New in DWARF v4:
  LLVMDWARFSourceLanguagePython,
  // New in DWARF v5:
  LLVMDWARFSourceLanguageOpenCL,
  LLVMDWARFSourceLanguageGo,
  LLVMDWARFSourceLanguageModula3,
  LLVMDWARFSourceLanguageHaskell,
  LLVMDWARFSourceLanguageC_plus_plus_03,
  LLVMDWARFSourceLanguageC_plus_plus_11,
  LLVMDWARFSourceLanguageOCaml,
  LLVMDWARFSourceLanguageRust,
  LLVMDWARFSourceLanguageC11,
  LLVMDWARFSourceLanguageSwift,
  LLVMDWARFSourceLanguageJulia,
  LLVMDWARFSourceLanguageDylan,
  LLVMDWARFSourceLanguageC_plus_plus_14,
  LLVMDWARFSourceLanguageFortran03,
  LLVMDWARFSourceLanguageFortran08,
  LLVMDWARFSourceLanguageRenderScript,
  LLVMDWARFSourceLanguageBLISS,
  LLVMDWARFSourceLanguageKotlin,
  LLVMDWARFSourceLanguageZig,
  LLVMDWARFSourceLanguageCrystal,
  LLVMDWARFSourceLanguageC_plus_plus_17,
  LLVMDWARFSourceLanguageC_plus_plus_20,
  LLVMDWARFSourceLanguageC17,
  LLVMDWARFSourceLanguageFortran18,
  LLVMDWARFSourceLanguageAda2005,
  LLVMDWARFSourceLanguageAda2012,
  LLVMDWARFSourceLanguageHIP,
  LLVMDWARFSourceLanguageAssembly,
  LLVMDWARFSourceLanguageC_sharp,
  LLVMDWARFSourceLanguageMojo,
  LLVMDWARFSourceLanguageGLSL,
  LLVMDWARFSourceLanguageGLSL_ES,
  LLVMDWARFSourceLanguageHLSL,
  LLVMDWARFSourceLanguageOpenCL_CPP,
  LLVMDWARFSourceLanguageCPP_for_OpenCL,
  LLVMDWARFSourceLanguageSYCL,
  LLVMDWARFSourceLanguageRuby,
  LLVMDWARFSourceLanguageMove,
  LLVMDWARFSourceLanguageHylo,
  LLVMDWARFSourceLanguageMetal,

  // Vendor extensions:
  LLVMDWARFSourceLanguageMips_Assembler,
  LLVMDWARFSourceLanguageGOOGLE_RenderScript,
  LLVMDWARFSourceLanguageBORLAND_Delphi
} LLVMDWARFSourceLanguage;

/**
 * The amount of debug information to emit.
 */
typedef enum {
    LLVMDWARFEmissionNone = 0,
    LLVMDWARFEmissionFull,
    LLVMDWARFEmissionLineTablesOnly
} LLVMDWARFEmissionKind;

/**
 * The kind of metadata nodes.
 */
// NOTE: New entries should always be appended instead of matching the order
// in Metadata.def.
enum {
  LLVMMDStringMetadataKind,
  LLVMConstantAsMetadataMetadataKind,
  LLVMLocalAsMetadataMetadataKind,
  LLVMDistinctMDOperandPlaceholderMetadataKind,
  LLVMMDTupleMetadataKind,
  LLVMDILocationMetadataKind,
  LLVMDIExpressionMetadataKind,
  LLVMDIGlobalVariableExpressionMetadataKind,
  LLVMGenericDINodeMetadataKind,
  LLVMDISubrangeMetadataKind,
  LLVMDIEnumeratorMetadataKind,
  LLVMDIBasicTypeMetadataKind,
  LLVMDIDerivedTypeMetadataKind,
  LLVMDICompositeTypeMetadataKind,
  LLVMDISubroutineTypeMetadataKind,
  LLVMDIFileMetadataKind,
  LLVMDICompileUnitMetadataKind,
  LLVMDISubprogramMetadataKind,
  LLVMDILexicalBlockMetadataKind,
  LLVMDILexicalBlockFileMetadataKind,
  LLVMDINamespaceMetadataKind,
  LLVMDIModuleMetadataKind,
  LLVMDITemplateTypeParameterMetadataKind,
  LLVMDITemplateValueParameterMetadataKind,
  LLVMDIGlobalVariableMetadataKind,
  LLVMDILocalVariableMetadataKind,
  LLVMDILabelMetadataKind,
  LLVMDIObjCPropertyMetadataKind,
  LLVMDIImportedEntityMetadataKind,
  LLVMDIMacroMetadataKind,
  LLVMDIMacroFileMetadataKind,
  LLVMDICommonBlockMetadataKind,
  LLVMDIStringTypeMetadataKind,
  LLVMDIGenericSubrangeMetadataKind,
  LLVMDIArgListMetadataKind,
  LLVMDIAssignIDMetadataKind,
  LLVMDISubrangeTypeMetadataKind,
  LLVMDIFixedPointTypeMetadataKind,
};
typedef unsigned LLVMMetadataKind;

/**
 * The kind of checksum to emit.
 */
typedef enum { CSK_MD5, CSK_SHA1, CSK_SHA256 } LLVMChecksumKind;

/**
 * An LLVM DWARF type encoding.
 */
typedef unsigned LLVMDWARFTypeEncoding;

/**
 * Describes the kind of macro declaration used for LLVMDIBuilderCreateMacro.
 * @see llvm::dwarf::MacinfoRecordType
 * @note Values are from DW_MACINFO_* constants in the DWARF specification.
 */
typedef enum {
  LLVMDWARFMacinfoRecordTypeDefine = 0x01,
  LLVMDWARFMacinfoRecordTypeMacro = 0x02,
  LLVMDWARFMacinfoRecordTypeStartFile = 0x03,
  LLVMDWARFMacinfoRecordTypeEndFile = 0x04,
  LLVMDWARFMacinfoRecordTypeVendorExt = 0xff
} LLVMDWARFMacinfoRecordType;

/**
 * The current debug metadata version number.
 */
LLVM_C_ABI unsigned LLVMDebugMetadataVersion(void);

/**
 * The version of debug metadata that's present in the provided \c Module.
 */
LLVM_C_ABI unsigned LLVMGetModuleDebugMetadataVersion(LLVMModuleRef Module);

/**
 * Strip debug info in the module if it exists.
 * To do this, we remove all calls to the debugger intrinsics and any named
 * metadata for debugging. We also remove debug locations for instructions.
 * Return true if module is modified.
 */
LLVM_C_ABI LLVMBool LLVMStripModuleDebugInfo(LLVMModuleRef Module);

/**
 * Construct a builder for a module, and do not allow for unresolved nodes
 * attached to the module.
 */
LLVM_C_ABI LLVMDIBuilderRef
LLVMCreateDIBuilderDisallowUnresolved(LLVMModuleRef M);

/**
 * Construct a builder for a module and collect unresolved nodes attached
 * to the module in order to resolve cycles during a call to
 * \c LLVMDIBuilderFinalize.
 */
LLVM_C_ABI LLVMDIBuilderRef LLVMCreateDIBuilder(LLVMModuleRef M);

/**
 * Deallocates the \c DIBuilder and everything it owns.
 * @note You must call \c LLVMDIBuilderFinalize before this
 */
LLVM_C_ABI void LLVMDisposeDIBuilder(LLVMDIBuilderRef Builder);

/**
 * Construct any deferred debug info descriptors.
 */
LLVM_C_ABI void LLVMDIBuilderFinalize(LLVMDIBuilderRef Builder);

/**
 * Finalize a specific subprogram.
 * No new variables may be added to this subprogram afterwards.
 */
LLVM_C_ABI void LLVMDIBuilderFinalizeSubprogram(LLVMDIBuilderRef Builder,
                                                LLVMMetadataRef Subprogram);

/**
 * A CompileUnit provides an anchor for all debugging
 * information generated during this instance of compilation.
 * \param Lang          Source programming language, eg.
 *                      \c LLVMDWARFSourceLanguageC99
 * \param FileRef       File info.
 * \param Producer      Identify the producer of debugging information
 *                      and code.  Usually this is a compiler
 *                      version string.
 * \param ProducerLen   The length of the C string passed to \c Producer.
 * \param isOptimized   A boolean flag which indicates whether optimization
 *                      is enabled or not.
 * \param Flags         This string lists command line options. This
 *                      string is directly embedded in debug info
 *                      output which may be used by a tool
 *                      analyzing generated debugging information.
 * \param FlagsLen      The length of the C string passed to \c Flags.
 * \param RuntimeVer    This indicates runtime version for languages like
 *                      Objective-C.
 * \param SplitName     The name of the file that we'll split debug info
 *                      out into.
 * \param SplitNameLen  The length of the C string passed to \c SplitName.
 * \param Kind          The kind of debug information to generate.
 * \param DWOId         The DWOId if this is a split skeleton compile unit.
 * \param SplitDebugInlining    Whether to emit inline debug info.
 * \param DebugInfoForProfiling Whether to emit extra debug info for
 *                              profile collection.
 * \param SysRoot         The Clang system root (value of -isysroot).
 * \param SysRootLen      The length of the C string passed to \c SysRoot.
 * \param SDK           The SDK. On Darwin, the last component of the sysroot.
 * \param SDKLen        The length of the C string passed to \c SDK.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateCompileUnit(
    LLVMDIBuilderRef Builder, LLVMDWARFSourceLanguage Lang,
    LLVMMetadataRef FileRef, const char *Producer, size_t ProducerLen,
    LLVMBool isOptimized, const char *Flags, size_t FlagsLen,
    unsigned RuntimeVer, const char *SplitName, size_t SplitNameLen,
    LLVMDWARFEmissionKind Kind, unsigned DWOId, LLVMBool SplitDebugInlining,
    LLVMBool DebugInfoForProfiling, const char *SysRoot, size_t SysRootLen,
    const char *SDK, size_t SDKLen);

/**
 * Create a file descriptor to hold debugging information for a file.
 * \param Builder      The \c DIBuilder.
 * \param Filename     File name.
 * \param FilenameLen  The length of the C string passed to \c Filename.
 * \param Directory    Directory.
 * \param DirectoryLen The length of the C string passed to \c Directory.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateFile(LLVMDIBuilderRef Builder,
                                                   const char *Filename,
                                                   size_t FilenameLen,
                                                   const char *Directory,
                                                   size_t DirectoryLen);

/**
 * Create a file descriptor to hold debugging information for a file.
 * \param Builder      The \c DIBuilder.
 * \param Filename     File name.
 * \param FilenameLen  The length of the C string passed to \c Filename.
 * \param Directory    Directory.
 * \param DirectoryLen The length of the C string passed to \c Directory.
 * \param ChecksumKind The kind of checksum. eg MD5, SHA256
 * \param Checksum     The checksum.
 * \param ChecksumLen  The length of the checksum.
 * \param Souce        The embedded source.
 * \param SourceLen    The length of the source.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateFileWithChecksum(
    LLVMDIBuilderRef Builder, const char *Filename, size_t FilenameLen,
    const char *Directory, size_t DirectoryLen, LLVMChecksumKind ChecksumKind,
    const char *Checksum, size_t ChecksumLen, const char *Source,
    size_t SourceLen);

/**
 * Creates a new descriptor for a module with the specified parent scope.
 * \param Builder         The \c DIBuilder.
 * \param ParentScope     The parent scope containing this module declaration.
 * \param Name            Module name.
 * \param NameLen         The length of the C string passed to \c Name.
 * \param ConfigMacros    A space-separated shell-quoted list of -D macro
                          definitions as they would appear on a command line.
 * \param ConfigMacrosLen The length of the C string passed to \c ConfigMacros.
 * \param IncludePath     The path to the module map file.
 * \param IncludePathLen  The length of the C string passed to \c IncludePath.
 * \param APINotesFile    The path to an API notes file for the module.
 * \param APINotesFileLen The length of the C string passed to \c APINotestFile.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateModule(
    LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope, const char *Name,
    size_t NameLen, const char *ConfigMacros, size_t ConfigMacrosLen,
    const char *IncludePath, size_t IncludePathLen, const char *APINotesFile,
    size_t APINotesFileLen);

/**
 * Creates a new descriptor for a namespace with the specified parent scope.
 * \param Builder          The \c DIBuilder.
 * \param ParentScope      The parent scope containing this module declaration.
 * \param Name             NameSpace name.
 * \param NameLen          The length of the C string passed to \c Name.
 * \param ExportSymbols    Whether or not the namespace exports symbols, e.g.
 *                         this is true of C++ inline namespaces.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateNameSpace(
    LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope, const char *Name,
    size_t NameLen, LLVMBool ExportSymbols);

/**
 * Create a new descriptor for the specified subprogram.
 * \param Builder         The \c DIBuilder.
 * \param Scope           Function scope.
 * \param Name            Function name.
 * \param NameLen         Length of enumeration name.
 * \param LinkageName     Mangled function name.
 * \param LinkageNameLen  Length of linkage name.
 * \param File            File where this variable is defined.
 * \param LineNo          Line number.
 * \param Ty              Function type.
 * \param IsLocalToUnit   True if this function is not externally visible.
 * \param IsDefinition    True if this is a function definition.
 * \param ScopeLine       Set to the beginning of the scope this starts
 * \param Flags           E.g.: \c LLVMDIFlagLValueReference. These flags are
 *                        used to emit dwarf attributes.
 * \param IsOptimized     True if optimization is ON.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateFunction(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, const char *LinkageName, size_t LinkageNameLen,
    LLVMMetadataRef File, unsigned LineNo, LLVMMetadataRef Ty,
    LLVMBool IsLocalToUnit, LLVMBool IsDefinition, unsigned ScopeLine,
    LLVMDIFlags Flags, LLVMBool IsOptimized);

/**
 * Create a descriptor for a lexical block with the specified parent context.
 * \param Builder      The \c DIBuilder.
 * \param Scope        Parent lexical block.
 * \param File         Source file.
 * \param Line         The line in the source file.
 * \param Column       The column in the source file.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef File,
    unsigned Line, unsigned Column);

/**
 * Create a descriptor for a lexical block with a new file attached.
 * \param Builder        The \c DIBuilder.
 * \param Scope          Lexical block.
 * \param File           Source file.
 * \param Discriminator  DWARF path discriminator value.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateLexicalBlockFile(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef File,
    unsigned Discriminator);

/**
 * Create a descriptor for an imported namespace. Suitable for e.g. C++
 * using declarations.
 * \param Builder    The \c DIBuilder.
 * \param Scope      The scope this module is imported into
 * \param File       File where the declaration is located.
 * \param Line       Line number of the declaration.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromNamespace(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef NS,
    LLVMMetadataRef File, unsigned Line);

/**
 * Create a descriptor for an imported module that aliases another
 * imported entity descriptor.
 * \param Builder        The \c DIBuilder.
 * \param Scope          The scope this module is imported into
 * \param ImportedEntity Previous imported entity to alias.
 * \param File           File where the declaration is located.
 * \param Line           Line number of the declaration.
 * \param Elements       Renamed elements.
 * \param NumElements    Number of renamed elements.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromAlias(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope,
    LLVMMetadataRef ImportedEntity, LLVMMetadataRef File, unsigned Line,
    LLVMMetadataRef *Elements, unsigned NumElements);

/**
 * Create a descriptor for an imported module.
 * \param Builder        The \c DIBuilder.
 * \param Scope          The scope this module is imported into
 * \param M              The module being imported here
 * \param File           File where the declaration is located.
 * \param Line           Line number of the declaration.
 * \param Elements       Renamed elements.
 * \param NumElements    Number of renamed elements.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromModule(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef M,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef *Elements,
    unsigned NumElements);

/**
 * Create a descriptor for an imported function, type, or variable.  Suitable
 * for e.g. FORTRAN-style USE declarations.
 * \param Builder        The DIBuilder.
 * \param Scope          The scope this module is imported into.
 * \param Decl           The declaration (or definition) of a function, type,
                         or variable.
 * \param File           File where the declaration is located.
 * \param Line           Line number of the declaration.
 * \param Name           A name that uniquely identifies this imported
 declaration.
 * \param NameLen        The length of the C string passed to \c Name.
 * \param Elements       Renamed elements.
 * \param NumElements    Number of renamed elements.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateImportedDeclaration(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef Decl,
    LLVMMetadataRef File, unsigned Line, const char *Name, size_t NameLen,
    LLVMMetadataRef *Elements, unsigned NumElements);

/**
 * Creates a new DebugLocation that describes a source location.
 * \param Line The line in the source file.
 * \param Column The column in the source file.
 * \param Scope The scope in which the location resides.
 * \param InlinedAt The scope where this location was inlined, if at all.
 *                  (optional).
 * \note If the item to which this location is attached cannot be
 *       attributed to a source line, pass 0 for the line and column.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateDebugLocation(
    LLVMContextRef Ctx, unsigned Line, unsigned Column, LLVMMetadataRef Scope,
    LLVMMetadataRef InlinedAt);

/**
 * Get the line number of this debug location.
 * \param Location     The debug location.
 *
 * @see DILocation::getLine()
 */
LLVM_C_ABI unsigned LLVMDILocationGetLine(LLVMMetadataRef Location);

/**
 * Get the column number of this debug location.
 * \param Location     The debug location.
 *
 * @see DILocation::getColumn()
 */
LLVM_C_ABI unsigned LLVMDILocationGetColumn(LLVMMetadataRef Location);

/**
 * Get the local scope associated with this debug location.
 * \param Location     The debug location.
 *
 * @see DILocation::getScope()
 */
LLVM_C_ABI LLVMMetadataRef LLVMDILocationGetScope(LLVMMetadataRef Location);

/**
 * Get the "inline at" location associated with this debug location.
 * \param Location     The debug location.
 *
 * @see DILocation::getInlinedAt()
 */
LLVM_C_ABI LLVMMetadataRef LLVMDILocationGetInlinedAt(LLVMMetadataRef Location);

/**
 * Get the metadata of the file associated with a given scope.
 * \param Scope     The scope object.
 *
 * @see DIScope::getFile()
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIScopeGetFile(LLVMMetadataRef Scope);

/**
 * Get the directory of a given file.
 * \param File     The file object.
 * \param Len      The length of the returned string.
 *
 * @see DIFile::getDirectory()
 */
LLVM_C_ABI const char *LLVMDIFileGetDirectory(LLVMMetadataRef File,
                                              unsigned *Len);

/**
 * Get the name of a given file.
 * \param File     The file object.
 * \param Len      The length of the returned string.
 *
 * @see DIFile::getFilename()
 */
LLVM_C_ABI const char *LLVMDIFileGetFilename(LLVMMetadataRef File,
                                             unsigned *Len);

/**
 * Get the source of a given file.
 * \param File     The file object.
 * \param Len      The length of the returned string.
 *
 * @see DIFile::getSource()
 */
LLVM_C_ABI const char *LLVMDIFileGetSource(LLVMMetadataRef File, unsigned *Len);

/**
 * Create a type array.
 * \param Builder        The DIBuilder.
 * \param Data           The type elements.
 * \param NumElements    Number of type elements.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(
    LLVMDIBuilderRef Builder, LLVMMetadataRef *Data, size_t NumElements);

/**
 * Create subroutine type.
 * \param Builder        The DIBuilder.
 * \param File            The file in which the subroutine resides.
 * \param ParameterTypes  An array of subroutine parameter types. This
 *                        includes return type at 0th index.
 * \param NumParameterTypes The number of parameter types in \c ParameterTypes
 * \param Flags           E.g.: \c LLVMDIFlagLValueReference.
 *                        These flags are used to emit dwarf attributes.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateSubroutineType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef File,
    LLVMMetadataRef *ParameterTypes, unsigned NumParameterTypes,
    LLVMDIFlags Flags);

/**
 * Create debugging information entry for a macro.
 * @param Builder         The DIBuilder.
 * @param ParentMacroFile Macro parent (could be NULL).
 * @param Line            Source line number where the macro is defined.
 * @param RecordType      DW_MACINFO_define or DW_MACINFO_undef.
 * @param Name            Macro name.
 * @param NameLen         Macro name length.
 * @param Value           Macro value.
 * @param ValueLen        Macro value length.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateMacro(
    LLVMDIBuilderRef Builder, LLVMMetadataRef ParentMacroFile, unsigned Line,
    LLVMDWARFMacinfoRecordType RecordType, const char *Name, size_t NameLen,
    const char *Value, size_t ValueLen);

/**
 * Create debugging information temporary entry for a macro file.
 * List of macro node direct children will be calculated by DIBuilder,
 * using the \p ParentMacroFile relationship.
 * @param Builder         The DIBuilder.
 * @param ParentMacroFile Macro parent (could be NULL).
 * @param Line            Source line number where the macro file is included.
 * @param File            File descriptor containing the name of the macro file.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateTempMacroFile(
    LLVMDIBuilderRef Builder, LLVMMetadataRef ParentMacroFile, unsigned Line,
    LLVMMetadataRef File);

/**
 * Create debugging information entry for an enumerator.
 * @param Builder        The DIBuilder.
 * @param Name           Enumerator name.
 * @param NameLen        Length of enumerator name.
 * @param Value          Enumerator value.
 * @param IsUnsigned     True if the value is unsigned.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateEnumerator(
    LLVMDIBuilderRef Builder, const char *Name, size_t NameLen, int64_t Value,
    LLVMBool IsUnsigned);

/**
 * Create debugging information entry for an enumerator of arbitrary precision.
 * @param Builder        The DIBuilder.
 * @param Name           Enumerator name.
 * @param NameLen        Length of enumerator name.
 * @param SizeInBits     Number of bits of the value.
 * @param Words          The words that make up the value.
 * @param IsUnsigned     True if the value is unsigned.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision(
    LLVMDIBuilderRef Builder, const char *Name, size_t NameLen,
    uint64_t SizeInBits, const uint64_t Words[], LLVMBool IsUnsigned);

/**
 * Create debugging information entry for an enumeration.
 * \param Builder        The DIBuilder.
 * \param Scope          Scope in which this enumeration is defined.
 * \param Name           Enumeration name.
 * \param NameLen        Length of enumeration name.
 * \param File           File where this member is defined.
 * \param LineNumber     Line number.
 * \param SizeInBits     Member size.
 * \param AlignInBits    Member alignment.
 * \param Elements       Enumeration elements.
 * \param NumElements    Number of enumeration elements.
 * \param ClassTy        Underlying type of a C++11/ObjC fixed enum.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateEnumerationType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint32_t AlignInBits, LLVMMetadataRef *Elements,
    unsigned NumElements, LLVMMetadataRef ClassTy);

/**
 * Create debugging information entry for a union.
 * \param Builder      The DIBuilder.
 * \param Scope        Scope in which this union is defined.
 * \param Name         Union name.
 * \param NameLen      Length of union name.
 * \param File         File where this member is defined.
 * \param LineNumber   Line number.
 * \param SizeInBits   Member size.
 * \param AlignInBits  Member alignment.
 * \param Flags        Flags to encode member attribute, e.g. private
 * \param Elements     Union elements.
 * \param NumElements  Number of union elements.
 * \param RunTimeLang  Optional parameter, Objective-C runtime version.
 * \param UniqueId     A unique identifier for the union.
 * \param UniqueIdLen  Length of unique identifier.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateUnionType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags,
    LLVMMetadataRef *Elements, unsigned NumElements, unsigned RunTimeLang,
    const char *UniqueId, size_t UniqueIdLen);

/**
 * Create debugging information entry for an array.
 * \param Builder      The DIBuilder.
 * \param Size         Array size.
 * \param AlignInBits  Alignment.
 * \param Ty           Element type.
 * \param Subscripts   Subscripts.
 * \param NumSubscripts Number of subscripts.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateArrayType(
    LLVMDIBuilderRef Builder, uint64_t Size, uint32_t AlignInBits,
    LLVMMetadataRef Ty, LLVMMetadataRef *Subscripts, unsigned NumSubscripts);

/**
 * Create debugging information entry for a set.
 * \param Builder        The DIBuilder.
 * \param Scope          The scope in which the set is defined.
 * \param Name           A name that uniquely identifies this set.
 * \param NameLen        The length of the C string passed to \c Name.
 * \param File           File where the set is located.
 * \param Line           Line number of the declaration.
 * \param SizeInBits     Set size.
 * \param AlignInBits    Set alignment.
 * \param BaseTy         The base type of the set.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateSetType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint32_t AlignInBits, LLVMMetadataRef BaseTy);

/**
 * Create a descriptor for a subrange with dynamic bounds.
 * \param Builder    The DIBuilder.
 * \param Scope      The scope in which the subrange is defined.
 * \param Name       A name that uniquely identifies this subrange.
 * \param NameLen    The length of the C string passed to \c Name.
 * \param LineNo     Line number.
 * \param File       File where the subrange is located.
 * \param SizeInBits Member size.
 * \param AlignInBits Member alignment.
 * \param Flags      Flags.
 * \param BaseTy     The base type of the subrange. eg integer or enumeration
 * \param LowerBound Lower bound of the subrange.
 * \param UpperBound Upper bound of the subrange.
 * \param Stride     Stride of the subrange.
 * \param Bias       Bias of the subrange.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateSubrangeType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, unsigned LineNo, LLVMMetadataRef File, uint64_t SizeInBits,
    uint32_t AlignInBits, LLVMDIFlags Flags, LLVMMetadataRef BaseTy,
    LLVMMetadataRef LowerBound, LLVMMetadataRef UpperBound,
    LLVMMetadataRef Stride, LLVMMetadataRef Bias);

/**
 * Create debugging information entry for a dynamic array.
 * \param Builder      The DIBuilder.
 * \param Size         Array size.
 * \param AlignInBits  Alignment.
 * \param Ty           Element type.
 * \param Subscripts   Subscripts.
 * \param NumSubscripts Number of subscripts.
 * \param DataLocation DataLocation. (DIVariable, DIExpression or NULL)
 * \param Associated   Associated. (DIVariable, DIExpression or NULL)
 * \param Allocated    Allocated. (DIVariable, DIExpression or NULL)
 * \param Rank         Rank. (DIVariable, DIExpression or NULL)
 * \param BitStride    BitStride.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateDynamicArrayType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, unsigned LineNo, LLVMMetadataRef File, uint64_t Size,
    uint32_t AlignInBits, LLVMMetadataRef Ty, LLVMMetadataRef *Subscripts,
    unsigned NumSubscripts, LLVMMetadataRef DataLocation,
    LLVMMetadataRef Associated, LLVMMetadataRef Allocated, LLVMMetadataRef Rank,
    LLVMMetadataRef BitStride);

/**
 * Replace arrays.
 *
 * @see DIBuilder::replaceArrays()
 */
LLVM_C_ABI void LLVMReplaceArrays(LLVMDIBuilderRef Builder, LLVMMetadataRef *T,
                                  LLVMMetadataRef *Elements,
                                  unsigned NumElements);

/**
 * Create debugging information entry for a vector type.
 * \param Builder      The DIBuilder.
 * \param Size         Vector size.
 * \param AlignInBits  Alignment.
 * \param Ty           Element type.
 * \param Subscripts   Subscripts.
 * \param NumSubscripts Number of subscripts.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateVectorType(
    LLVMDIBuilderRef Builder, uint64_t Size, uint32_t AlignInBits,
    LLVMMetadataRef Ty, LLVMMetadataRef *Subscripts, unsigned NumSubscripts);

/**
 * Create a DWARF unspecified type.
 * \param Builder   The DIBuilder.
 * \param Name      The unspecified type's name.
 * \param NameLen   Length of type name.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateUnspecifiedType(
    LLVMDIBuilderRef Builder, const char *Name, size_t NameLen);

/**
 * Create debugging information entry for a basic
 * type.
 * \param Builder     The DIBuilder.
 * \param Name        Type name.
 * \param NameLen     Length of type name.
 * \param SizeInBits  Size of the type.
 * \param Encoding    DWARF encoding code, e.g. \c LLVMDWARFTypeEncoding_float.
 * \param Flags       Flags to encode optional attribute like endianity
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateBasicType(
    LLVMDIBuilderRef Builder, const char *Name, size_t NameLen,
    uint64_t SizeInBits, LLVMDWARFTypeEncoding Encoding, LLVMDIFlags Flags);

/**
 * Create debugging information entry for a pointer.
 * \param Builder     The DIBuilder.
 * \param PointeeTy         Type pointed by this pointer.
 * \param SizeInBits        Size.
 * \param AlignInBits       Alignment. (optional, pass 0 to ignore)
 * \param AddressSpace      DWARF address space. (optional, pass 0 to ignore)
 * \param Name              Pointer type name. (optional)
 * \param NameLen           Length of pointer type name. (optional)
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreatePointerType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeTy, uint64_t SizeInBits,
    uint32_t AlignInBits, unsigned AddressSpace, const char *Name,
    size_t NameLen);

/**
 * Create debugging information entry for a struct.
 * \param Builder     The DIBuilder.
 * \param Scope        Scope in which this struct is defined.
 * \param Name         Struct name.
 * \param NameLen      Struct name length.
 * \param File         File where this member is defined.
 * \param LineNumber   Line number.
 * \param SizeInBits   Member size.
 * \param AlignInBits  Member alignment.
 * \param Flags        Flags to encode member attribute, e.g. private
 * \param Elements     Struct elements.
 * \param NumElements  Number of struct elements.
 * \param RunTimeLang  Optional parameter, Objective-C runtime version.
 * \param VTableHolder The object containing the vtable for the struct.
 * \param UniqueId     A unique identifier for the struct.
 * \param UniqueIdLen  Length of the unique identifier for the struct.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateStructType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags,
    LLVMMetadataRef DerivedFrom, LLVMMetadataRef *Elements,
    unsigned NumElements, unsigned RunTimeLang, LLVMMetadataRef VTableHolder,
    const char *UniqueId, size_t UniqueIdLen);

/**
 * Create debugging information entry for a member.
 * \param Builder      The DIBuilder.
 * \param Scope        Member scope.
 * \param Name         Member name.
 * \param NameLen      Length of member name.
 * \param File         File where this member is defined.
 * \param LineNo       Line number.
 * \param SizeInBits   Member size.
 * \param AlignInBits  Member alignment.
 * \param OffsetInBits Member offset.
 * \param Flags        Flags to encode member attribute, e.g. private
 * \param Ty           Parent type.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateMemberType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNo, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags,
    LLVMMetadataRef Ty);

/**
 * Create debugging information entry for a
 * C++ static data member.
 * \param Builder      The DIBuilder.
 * \param Scope        Member scope.
 * \param Name         Member name.
 * \param NameLen      Length of member name.
 * \param File         File where this member is declared.
 * \param LineNumber   Line number.
 * \param Type         Type of the static member.
 * \param Flags        Flags to encode member attribute, e.g. private.
 * \param ConstantVal  Const initializer of the member.
 * \param AlignInBits  Member alignment.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateStaticMemberType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    LLVMMetadataRef Type, LLVMDIFlags Flags, LLVMValueRef ConstantVal,
    uint32_t AlignInBits);

/**
 * Create debugging information entry for a pointer to member.
 * \param Builder      The DIBuilder.
 * \param PointeeType  Type pointed to by this pointer.
 * \param ClassType    Type for which this pointer points to members of.
 * \param SizeInBits   Size.
 * \param AlignInBits  Alignment.
 * \param Flags        Flags.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateMemberPointerType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeType,
    LLVMMetadataRef ClassType, uint64_t SizeInBits, uint32_t AlignInBits,
    LLVMDIFlags Flags);
/**
 * Create debugging information entry for Objective-C instance variable.
 * \param Builder      The DIBuilder.
 * \param Name         Member name.
 * \param NameLen      The length of the C string passed to \c Name.
 * \param File         File where this member is defined.
 * \param LineNo       Line number.
 * \param SizeInBits   Member size.
 * \param AlignInBits  Member alignment.
 * \param OffsetInBits Member offset.
 * \param Flags        Flags to encode member attribute, e.g. private
 * \param Ty           Parent type.
 * \param PropertyNode Property associated with this ivar.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateObjCIVar(
    LLVMDIBuilderRef Builder, const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNo, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags,
    LLVMMetadataRef Ty, LLVMMetadataRef PropertyNode);

/**
 * Create debugging information entry for Objective-C property.
 * \param Builder            The DIBuilder.
 * \param Name               Property name.
 * \param NameLen            The length of the C string passed to \c Name.
 * \param File               File where this property is defined.
 * \param LineNo             Line number.
 * \param GetterName         Name of the Objective C property getter selector.
 * \param GetterNameLen      The length of the C string passed to \c GetterName.
 * \param SetterName         Name of the Objective C property setter selector.
 * \param SetterNameLen      The length of the C string passed to \c SetterName.
 * \param PropertyAttributes Objective C property attributes.
 * \param Ty                 Type.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateObjCProperty(
    LLVMDIBuilderRef Builder, const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNo, const char *GetterName,
    size_t GetterNameLen, const char *SetterName, size_t SetterNameLen,
    unsigned PropertyAttributes, LLVMMetadataRef Ty);

/**
 * Create a uniqued DIType* clone with FlagObjectPointer. If \c Implicit
 * is true, then also set FlagArtificial.
 * \param Builder   The DIBuilder.
 * \param Type      The underlying type to which this pointer points.
 * \param Implicit  Indicates whether this pointer was implicitly generated
 *                  (i.e., not spelled out in source).
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateObjectPointerType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Type, LLVMBool Implicit);

/**
 * Create debugging information entry for a qualified
 * type, e.g. 'const int'.
 * \param Builder     The DIBuilder.
 * \param Tag         Tag identifying type,
 *                    e.g. LLVMDWARFTypeQualifier_volatile_type
 * \param Type        Base Type.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateQualifiedType(
    LLVMDIBuilderRef Builder, unsigned Tag, LLVMMetadataRef Type);

/**
 * Create debugging information entry for a c++
 * style reference or rvalue reference type.
 * \param Builder   The DIBuilder.
 * \param Tag       Tag identifying type,
 * \param Type      Base Type.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateReferenceType(
    LLVMDIBuilderRef Builder, unsigned Tag, LLVMMetadataRef Type);

/**
 * Create C++11 nullptr type.
 * \param Builder   The DIBuilder.
 */
LLVM_C_ABI LLVMMetadataRef
LLVMDIBuilderCreateNullPtrType(LLVMDIBuilderRef Builder);

/**
 * Create debugging information entry for a typedef.
 * \param Builder    The DIBuilder.
 * \param Type       Original type.
 * \param Name       Typedef name.
 * \param File       File where this type is defined.
 * \param LineNo     Line number.
 * \param Scope      The surrounding context for the typedef.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateTypedef(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Type, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Scope, uint32_t AlignInBits);

/**
 * Create debugging information entry to establish inheritance relationship
 * between two types.
 * \param Builder       The DIBuilder.
 * \param Ty            Original type.
 * \param BaseTy        Base type. Ty is inherits from base.
 * \param BaseOffset    Base offset.
 * \param VBPtrOffset  Virtual base pointer offset.
 * \param Flags         Flags to describe inheritance attribute, e.g. private
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateInheritance(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Ty, LLVMMetadataRef BaseTy,
    uint64_t BaseOffset, uint32_t VBPtrOffset, LLVMDIFlags Flags);

/**
 * Create a permanent forward-declared type.
 * \param Builder             The DIBuilder.
 * \param Tag                 A unique tag for this type.
 * \param Name                Type name.
 * \param NameLen             Length of type name.
 * \param Scope               Type scope.
 * \param File                File where this type is defined.
 * \param Line                Line number where this type is defined.
 * \param RuntimeLang         Indicates runtime version for languages like
 *                            Objective-C.
 * \param SizeInBits          Member size.
 * \param AlignInBits         Member alignment.
 * \param UniqueIdentifier    A unique identifier for the type.
 * \param UniqueIdentifierLen Length of the unique identifier.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateForwardDecl(
    LLVMDIBuilderRef Builder, unsigned Tag, const char *Name, size_t NameLen,
    LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits,
    const char *UniqueIdentifier, size_t UniqueIdentifierLen);

/**
 * Create a temporary forward-declared type.
 * \param Builder             The DIBuilder.
 * \param Tag                 A unique tag for this type.
 * \param Name                Type name.
 * \param NameLen             Length of type name.
 * \param Scope               Type scope.
 * \param File                File where this type is defined.
 * \param Line                Line number where this type is defined.
 * \param RuntimeLang         Indicates runtime version for languages like
 *                            Objective-C.
 * \param SizeInBits          Member size.
 * \param AlignInBits         Member alignment.
 * \param Flags               Flags.
 * \param UniqueIdentifier    A unique identifier for the type.
 * \param UniqueIdentifierLen Length of the unique identifier.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateReplaceableCompositeType(
    LLVMDIBuilderRef Builder, unsigned Tag, const char *Name, size_t NameLen,
    LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits,
    LLVMDIFlags Flags, const char *UniqueIdentifier,
    size_t UniqueIdentifierLen);

/**
 * Create debugging information entry for a bit field member.
 * \param Builder             The DIBuilder.
 * \param Scope               Member scope.
 * \param Name                Member name.
 * \param NameLen             Length of member name.
 * \param File                File where this member is defined.
 * \param LineNumber          Line number.
 * \param SizeInBits          Member size.
 * \param OffsetInBits        Member offset.
 * \param StorageOffsetInBits Member storage offset.
 * \param Flags               Flags to encode member attribute.
 * \param Type                Parent type.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateBitFieldMemberType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t OffsetInBits, uint64_t StorageOffsetInBits,
    LLVMDIFlags Flags, LLVMMetadataRef Type);

/**
 * Create debugging information entry for a class.
 * \param Scope               Scope in which this class is defined.
 * \param Name                Class name.
 * \param NameLen             The length of the C string passed to \c Name.
 * \param File                File where this member is defined.
 * \param LineNumber          Line number.
 * \param SizeInBits          Member size.
 * \param AlignInBits         Member alignment.
 * \param OffsetInBits        Member offset.
 * \param Flags               Flags to encode member attribute, e.g. private.
 * \param DerivedFrom         Debug info of the base class of this type.
 * \param Elements            Class members.
 * \param NumElements         Number of class elements.
 * \param VTableHolder        Debug info of the base class that contains vtable
 *                            for this type. This is used in
 *                            DW_AT_containing_type. See DWARF documentation
 *                            for more info.
 * \param TemplateParamsNode  Template type parameters.
 * \param UniqueIdentifier    A unique identifier for the type.
 * \param UniqueIdentifierLen Length of the unique identifier.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateClassType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits,
    LLVMDIFlags Flags, LLVMMetadataRef DerivedFrom, LLVMMetadataRef *Elements,
    unsigned NumElements, LLVMMetadataRef VTableHolder,
    LLVMMetadataRef TemplateParamsNode, const char *UniqueIdentifier,
    size_t UniqueIdentifierLen);

/**
 * Create a uniqued DIType* clone with FlagArtificial set.
 * \param Builder     The DIBuilder.
 * \param Type        The underlying type.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateArtificialType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Type);

/**
 * Get the name of this DIType.
 * \param DType     The DIType.
 * \param Length    The length of the returned string.
 *
 * @see DIType::getName()
 */
LLVM_C_ABI const char *LLVMDITypeGetName(LLVMMetadataRef DType, size_t *Length);

/**
 * Get the size of this DIType in bits.
 * \param DType     The DIType.
 *
 * @see DIType::getSizeInBits()
 */
LLVM_C_ABI uint64_t LLVMDITypeGetSizeInBits(LLVMMetadataRef DType);

/**
 * Get the offset of this DIType in bits.
 * \param DType     The DIType.
 *
 * @see DIType::getOffsetInBits()
 */
LLVM_C_ABI uint64_t LLVMDITypeGetOffsetInBits(LLVMMetadataRef DType);

/**
 * Get the alignment of this DIType in bits.
 * \param DType     The DIType.
 *
 * @see DIType::getAlignInBits()
 */
LLVM_C_ABI uint32_t LLVMDITypeGetAlignInBits(LLVMMetadataRef DType);

/**
 * Get the source line where this DIType is declared.
 * \param DType     The DIType.
 *
 * @see DIType::getLine()
 */
LLVM_C_ABI unsigned LLVMDITypeGetLine(LLVMMetadataRef DType);

/**
 * Get the flags associated with this DIType.
 * \param DType     The DIType.
 *
 * @see DIType::getFlags()
 */
LLVM_C_ABI LLVMDIFlags LLVMDITypeGetFlags(LLVMMetadataRef DType);

/**
 * Create a descriptor for a value range.
 * \param Builder    The DIBuilder.
 * \param LowerBound Lower bound of the subrange, e.g. 0 for C, 1 for Fortran.
 * \param Count      Count of elements in the subrange.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(
    LLVMDIBuilderRef Builder, int64_t LowerBound, int64_t Count);

/**
 * Create an array of DI Nodes.
 * \param Builder        The DIBuilder.
 * \param Data           The DI Node elements.
 * \param NumElements    Number of DI Node elements.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(
    LLVMDIBuilderRef Builder, LLVMMetadataRef *Data, size_t NumElements);

/**
 * Create a new descriptor for the specified variable which has a complex
 * address expression for its address.
 * \param Builder     The DIBuilder.
 * \param Addr        An array of complex address operations.
 * \param Length      Length of the address operation array.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateExpression(
    LLVMDIBuilderRef Builder, uint64_t *Addr, size_t Length);

/**
 * Create a new descriptor for the specified variable that does not have an
 * address, but does have a constant value.
 * \param Builder     The DIBuilder.
 * \param Value       The constant value.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateConstantValueExpression(
    LLVMDIBuilderRef Builder, uint64_t Value);

/**
 * Create a new descriptor for the specified variable.
 * \param Scope       Variable scope.
 * \param Name        Name of the variable.
 * \param NameLen     The length of the C string passed to \c Name.
 * \param Linkage     Mangled  name of the variable.
 * \param LinkLen     The length of the C string passed to \c Linkage.
 * \param File        File where this variable is defined.
 * \param LineNo      Line number.
 * \param Ty          Variable Type.
 * \param LocalToUnit Boolean flag indicate whether this variable is
 *                    externally visible or not.
 * \param Expr        The location of the global relative to the attached
 *                    GlobalVariable.
 * \param Decl        Reference to the corresponding declaration.
 *                    variables.
 * \param AlignInBits Variable alignment(or 0 if no alignment attr was
 *                    specified)
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateGlobalVariableExpression(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, const char *Linkage, size_t LinkLen, LLVMMetadataRef File,
    unsigned LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit,
    LLVMMetadataRef Expr, LLVMMetadataRef Decl, uint32_t AlignInBits);

/**
 * Get the dwarf::Tag of a DINode
 */
LLVM_C_ABI uint16_t LLVMGetDINodeTag(LLVMMetadataRef MD);

/**
 * Retrieves the \c DIVariable associated with this global variable expression.
 * \param GVE    The global variable expression.
 *
 * @see llvm::DIGlobalVariableExpression::getVariable()
 */
LLVM_C_ABI LLVMMetadataRef
LLVMDIGlobalVariableExpressionGetVariable(LLVMMetadataRef GVE);

/**
 * Retrieves the \c DIExpression associated with this global variable expression.
 * \param GVE    The global variable expression.
 *
 * @see llvm::DIGlobalVariableExpression::getExpression()
 */
LLVM_C_ABI LLVMMetadataRef
LLVMDIGlobalVariableExpressionGetExpression(LLVMMetadataRef GVE);

/**
 * Get the metadata of the file associated with a given variable.
 * \param Var     The variable object.
 *
 * @see DIVariable::getFile()
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIVariableGetFile(LLVMMetadataRef Var);

/**
 * Get the metadata of the scope associated with a given variable.
 * \param Var     The variable object.
 *
 * @see DIVariable::getScope()
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIVariableGetScope(LLVMMetadataRef Var);

/**
 * Get the source line where this \c DIVariable is declared.
 * \param Var     The DIVariable.
 *
 * @see DIVariable::getLine()
 */
LLVM_C_ABI unsigned LLVMDIVariableGetLine(LLVMMetadataRef Var);

/**
 * Create a new temporary \c MDNode.  Suitable for use in constructing cyclic
 * \c MDNode structures. A temporary \c MDNode is not uniqued, may be RAUW'd,
 * and must be manually deleted with \c LLVMDisposeTemporaryMDNode.
 * \param Ctx            The context in which to construct the temporary node.
 * \param Data           The metadata elements.
 * \param NumElements    Number of metadata elements.
 */
LLVM_C_ABI LLVMMetadataRef LLVMTemporaryMDNode(LLVMContextRef Ctx,
                                               LLVMMetadataRef *Data,
                                               size_t NumElements);

/**
 * Deallocate a temporary node.
 *
 * Calls \c replaceAllUsesWith(nullptr) before deleting, so any remaining
 * references will be reset.
 * \param TempNode    The temporary metadata node.
 */
LLVM_C_ABI void LLVMDisposeTemporaryMDNode(LLVMMetadataRef TempNode);

/**
 * Replace all uses of temporary metadata.
 * \param TempTargetMetadata    The temporary metadata node.
 * \param Replacement           The replacement metadata node.
 */
LLVM_C_ABI void
LLVMMetadataReplaceAllUsesWith(LLVMMetadataRef TempTargetMetadata,
                               LLVMMetadataRef Replacement);

/**
 * Create a new descriptor for the specified global variable that is temporary
 * and meant to be RAUWed.
 * \param Scope       Variable scope.
 * \param Name        Name of the variable.
 * \param NameLen     The length of the C string passed to \c Name.
 * \param Linkage     Mangled  name of the variable.
 * \param LnkLen      The length of the C string passed to \c Linkage.
 * \param File        File where this variable is defined.
 * \param LineNo      Line number.
 * \param Ty          Variable Type.
 * \param LocalToUnit Boolean flag indicate whether this variable is
 *                    externally visible or not.
 * \param Decl        Reference to the corresponding declaration.
 * \param AlignInBits Variable alignment(or 0 if no alignment attr was
 *                    specified)
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateTempGlobalVariableFwdDecl(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, const char *Linkage, size_t LnkLen, LLVMMetadataRef File,
    unsigned LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit,
    LLVMMetadataRef Decl, uint32_t AlignInBits);

/**
 * Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).
 * See https://llvm.org/docs/RemoveDIsDebugInfo.html#c-api-changes
 *
 * The debug format can be switched later after inserting the records using
 * LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.
 *
 * Insert a Declare DbgRecord before the given instruction.
 * \param Builder     The DIBuilder.
 * \param Storage     The storage of the variable to declare.
 * \param VarInfo     The variable's debug info descriptor.
 * \param Expr        A complex location expression for the variable.
 * \param DebugLoc    Debug info location.
 * \param Instr       Instruction acting as a location for the new record.
 */
LLVM_C_ABI LLVMDbgRecordRef LLVMDIBuilderInsertDeclareRecordBefore(
    LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr);

/**
 * Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).
 * See https://llvm.org/docs/RemoveDIsDebugInfo.html#c-api-changes
 *
 * The debug format can be switched later after inserting the records using
 * LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.
 *
 * Insert a Declare DbgRecord at the end of the given basic block. If the basic
 * block has a terminator instruction, the record is inserted before that
 * terminator instruction.
 * \param Builder     The DIBuilder.
 * \param Storage     The storage of the variable to declare.
 * \param VarInfo     The variable's debug info descriptor.
 * \param Expr        A complex location expression for the variable.
 * \param DebugLoc    Debug info location.
 * \param Block       Basic block acting as a location for the new record.
 */
LLVM_C_ABI LLVMDbgRecordRef LLVMDIBuilderInsertDeclareRecordAtEnd(
    LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMBasicBlockRef Block);

/**
 * Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).
 * See https://llvm.org/docs/RemoveDIsDebugInfo.html#c-api-changes
 *
 * The debug format can be switched later after inserting the records using
 * LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.
 *
 * Insert a new debug record before the given instruction.
 * \param Builder     The DIBuilder.
 * \param Val         The value of the variable.
 * \param VarInfo     The variable's debug info descriptor.
 * \param Expr        A complex location expression for the variable.
 * \param DebugLoc    Debug info location.
 * \param Instr       Instruction acting as a location for the new record.
 */
LLVM_C_ABI LLVMDbgRecordRef LLVMDIBuilderInsertDbgValueRecordBefore(
    LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr);

/**
 * Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).
 * See https://llvm.org/docs/RemoveDIsDebugInfo.html#c-api-changes
 *
 * The debug format can be switched later after inserting the records using
 * LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.
 *
 * Insert a new debug record at the end of the given basic block. If the
 * basic block has a terminator instruction, the record is inserted before
 * that terminator instruction.
 * \param Builder     The DIBuilder.
 * \param Val         The value of the variable.
 * \param VarInfo     The variable's debug info descriptor.
 * \param Expr        A complex location expression for the variable.
 * \param DebugLoc    Debug info location.
 * \param Block       Basic block acting as a location for the new record.
 */
LLVM_C_ABI LLVMDbgRecordRef LLVMDIBuilderInsertDbgValueRecordAtEnd(
    LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMBasicBlockRef Block);

/**
 * Create a new descriptor for a local auto variable.
 * \param Builder         The DIBuilder.
 * \param Scope           The local scope the variable is declared in.
 * \param Name            Variable name.
 * \param NameLen         Length of variable name.
 * \param File            File where this variable is defined.
 * \param LineNo          Line number.
 * \param Ty              Metadata describing the type of the variable.
 * \param AlwaysPreserve  If true, this descriptor will survive optimizations.
 * \param Flags           Flags.
 * \param AlignInBits     Variable alignment.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNo, LLVMMetadataRef Ty,
    LLVMBool AlwaysPreserve, LLVMDIFlags Flags, uint32_t AlignInBits);

/**
 * Create a new descriptor for a function parameter variable.
 * \param Builder         The DIBuilder.
 * \param Scope           The local scope the variable is declared in.
 * \param Name            Variable name.
 * \param NameLen         Length of variable name.
 * \param ArgNo           Unique argument number for this variable; starts at 1.
 * \param File            File where this variable is defined.
 * \param LineNo          Line number.
 * \param Ty              Metadata describing the type of the variable.
 * \param AlwaysPreserve  If true, this descriptor will survive optimizations.
 * \param Flags           Flags.
 */
LLVM_C_ABI LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, unsigned ArgNo, LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, LLVMBool AlwaysPreserve, LLVMDIFlags Flags);

/**
 * Get the metadata of the subprogram attached to a function.
 *
 * @see llvm::Function::getSubprogram()
 */
LLVM_C_ABI LLVMMetadataRef LLVMGetSubprogram(LLVMValueRef Func);

/**
 * Set the subprogram attached to a function.
 *
 * @see llvm::Function::setSubprogram()
 */
LLVM_C_ABI void LLVMSetSubprogram(LLVMValueRef Func, LLVMMetadataRef SP);

/**
 * Get the line associated with a given subprogram.
 * \param Subprogram     The subprogram object.
 *
 * @see DISubprogram::getLine()
 */
LLVM_C_ABI unsigned LLVMDISubprogramGetLine(LLVMMetadataRef Subprogram);

/**
 * Replace the subprogram subroutine type.
 * \param Subprogram        The subprogram object.
 * \param SubroutineType    The new subroutine type.
 *
 * @see DISubprogram::replaceType()
 */
LLVM_C_ABI void LLVMDISubprogramReplaceType(LLVMMetadataRef Subprogram,
                                            LLVMMetadataRef SubroutineType);

/**
 * Get the debug location for the given instruction.
 *
 * @see llvm::Instruction::getDebugLoc()
 */
LLVM_C_ABI LLVMMetadataRef LLVMInstructionGetDebugLoc(LLVMValueRef Inst);

/**
 * Set the debug location for the given instruction.
 *
 * To clear the location metadata of the given instruction, pass NULL to \p Loc.
 *
 * @see llvm::Instruction::setDebugLoc()
 */
LLVM_C_ABI void LLVMInstructionSetDebugLoc(LLVMValueRef Inst,
                                           LLVMMetadataRef Loc);

/**
 * Create a new descriptor for a label
 *
 * \param Builder         The DIBuilder.
 * \param Scope           The scope to create the label in.
 * \param Name            Variable name.
 * \param NameLen         Length of variable name.
 * \param File            The file to create the label in.
 * \param LineNo          Line Number.
 * \param AlwaysPreserve  Preserve the label regardless of optimization.
 *
 * @see llvm::DIBuilder::createLabel()
 */
LLVM_C_ABI LLVMMetadataRef
LLVMDIBuilderCreateLabel(LLVMDIBuilderRef Builder, LLVMMetadataRef Context,
                         const char *Name, size_t NameLen, LLVMMetadataRef File,
                         unsigned LineNo, LLVMBool AlwaysPreserve);

/**
 * Insert a new llvm.dbg.label intrinsic call
 *
 * \param Builder         The DIBuilder.
 * \param LabelInfo       The Label's debug info descriptor
 * \param Location        The debug info location
 * \param InsertBefore    Location for the new intrinsic.
 *
 * @see llvm::DIBuilder::insertLabel()
 */
LLVM_C_ABI LLVMDbgRecordRef LLVMDIBuilderInsertLabelBefore(
    LLVMDIBuilderRef Builder, LLVMMetadataRef LabelInfo,
    LLVMMetadataRef Location, LLVMValueRef InsertBefore);

/**
 * Insert a new llvm.dbg.label intrinsic call
 *
 * \param Builder         The DIBuilder.
 * \param LabelInfo       The Label's debug info descriptor
 * \param Location        The debug info location
 * \param InsertAtEnd     Location for the new intrinsic.
 *
 * @see llvm::DIBuilder::insertLabel()
 */
LLVM_C_ABI LLVMDbgRecordRef LLVMDIBuilderInsertLabelAtEnd(
    LLVMDIBuilderRef Builder, LLVMMetadataRef LabelInfo,
    LLVMMetadataRef Location, LLVMBasicBlockRef InsertAtEnd);

/**
 * Obtain the enumerated type of a Metadata instance.
 *
 * @see llvm::Metadata::getMetadataID()
 */
LLVM_C_ABI LLVMMetadataKind LLVMGetMetadataKind(LLVMMetadataRef Metadata);

/**
 * @}
 */

LLVM_C_EXTERN_C_END

#endif
