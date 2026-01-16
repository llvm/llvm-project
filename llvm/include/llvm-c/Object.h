/*===-- llvm-c/Object.h - Object Lib C Iface --------------------*- C++ -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/
/*                                                                            */
/* This header declares the C interface to libLLVMObject.a, which             */
/* implements object file reading and writing.                                */
/*                                                                            */
/* Many exotic languages can interoperate with C code but have a harder time  */
/* with C++ due to name mangling. So in addition to C, this interface enables */
/* tools written in such languages.                                           */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_OBJECT_H
#define LLVM_C_OBJECT_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"
#include "llvm-c/Visibility.h"
#include "llvm/Config/llvm-config.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCObject Object file reading and writing
 * @ingroup LLVMC
 *
 * @{
 */

// Opaque type wrappers
typedef struct LLVMOpaqueSectionIterator *LLVMSectionIteratorRef;
typedef struct LLVMOpaqueSymbolIterator *LLVMSymbolIteratorRef;
typedef struct LLVMOpaqueRelocationIterator *LLVMRelocationIteratorRef;

typedef enum {
  LLVMBinaryTypeArchive,              /**< Archive file. */
  LLVMBinaryTypeMachOUniversalBinary, /**< Mach-O Universal Binary file. */
  LLVMBinaryTypeCOFFImportFile,       /**< COFF Import file. */
  LLVMBinaryTypeIR,                   /**< LLVM IR. */
  LLVMBinaryTypeWinRes,               /**< Windows resource (.res) file. */
  LLVMBinaryTypeCOFF,                 /**< COFF Object file. */
  LLVMBinaryTypeELF32L,               /**< ELF 32-bit, little endian. */
  LLVMBinaryTypeELF32B,               /**< ELF 32-bit, big endian. */
  LLVMBinaryTypeELF64L,               /**< ELF 64-bit, little endian. */
  LLVMBinaryTypeELF64B,               /**< ELF 64-bit, big endian. */
  LLVMBinaryTypeMachO32L,             /**< MachO 32-bit, little endian. */
  LLVMBinaryTypeMachO32B,             /**< MachO 32-bit, big endian. */
  LLVMBinaryTypeMachO64L,             /**< MachO 64-bit, little endian. */
  LLVMBinaryTypeMachO64B,             /**< MachO 64-bit, big endian. */
  LLVMBinaryTypeWasm,                 /**< Web Assembly. */
  LLVMBinaryTypeOffload,              /**< Offloading fatbinary. */
  LLVMBinaryTypeDXcontainer,          /**< DirectX Binary Container. */

} LLVMBinaryType;

/**
 * Create a binary file from the given memory buffer.
 *
 * The exact type of the binary file will be inferred automatically, and the
 * appropriate implementation selected.  The context may be NULL except if
 * the resulting file is an LLVM IR file.
 *
 * The memory buffer is not consumed by this function. It is the responsibility
 * of the caller to free it with \c LLVMDisposeMemoryBuffer.
 *
 * If NULL is returned, the \p ErrorMessage parameter is populated with the
 * error's description.  It is then the caller's responsibility to free this
 * message by calling \c LLVMDisposeMessage.
 *
 * @see llvm::object::createBinary
 */
LLVM_C_ABI LLVMBinaryRef LLVMCreateBinary(LLVMMemoryBufferRef MemBuf,
                                          LLVMContextRef Context,
                                          char **ErrorMessage);

/**
 * Dispose of a binary file.
 *
 * The binary file does not own its backing buffer. It is the responsibility
 * of the caller to free it with \c LLVMDisposeMemoryBuffer.
 */
LLVM_C_ABI void LLVMDisposeBinary(LLVMBinaryRef BR);

/**
 * Retrieves a copy of the memory buffer associated with this object file.
 *
 * The returned buffer is merely a shallow copy and does not own the actual
 * backing buffer of the binary. Nevertheless, it is the responsibility of the
 * caller to free it with \c LLVMDisposeMemoryBuffer.
 *
 * @see llvm::object::getMemoryBufferRef
 */
LLVM_C_ABI LLVMMemoryBufferRef LLVMBinaryCopyMemoryBuffer(LLVMBinaryRef BR);

/**
 * Retrieve the specific type of a binary.
 *
 * @see llvm::object::Binary::getType
 */
LLVM_C_ABI LLVMBinaryType LLVMBinaryGetType(LLVMBinaryRef BR);

/*
 * For a Mach-O universal binary file, retrieves the object file corresponding
 * to the given architecture if it is present as a slice.
 *
 * If NULL is returned, the \p ErrorMessage parameter is populated with the
 * error's description.  It is then the caller's responsibility to free this
 * message by calling \c LLVMDisposeMessage.
 *
 * It is the responsiblity of the caller to free the returned object file by
 * calling \c LLVMDisposeBinary.
 */
LLVM_C_ABI LLVMBinaryRef LLVMMachOUniversalBinaryCopyObjectForArch(
    LLVMBinaryRef BR, const char *Arch, size_t ArchLen, char **ErrorMessage);

/**
 * Retrieve a copy of the section iterator for this object file.
 *
 * If there are no sections, the result is NULL.
 *
 * The returned iterator is merely a shallow copy. Nevertheless, it is
 * the responsibility of the caller to free it with
 * \c LLVMDisposeSectionIterator.
 *
 * @see llvm::object::sections()
 */
LLVM_C_ABI LLVMSectionIteratorRef
LLVMObjectFileCopySectionIterator(LLVMBinaryRef BR);

/**
 * Returns whether the given section iterator is at the end.
 *
 * @see llvm::object::section_end
 */
LLVM_C_ABI LLVMBool LLVMObjectFileIsSectionIteratorAtEnd(
    LLVMBinaryRef BR, LLVMSectionIteratorRef SI);

/**
 * Retrieve a copy of the symbol iterator for this object file.
 *
 * If there are no symbols, the result is NULL.
 *
 * The returned iterator is merely a shallow copy. Nevertheless, it is
 * the responsibility of the caller to free it with
 * \c LLVMDisposeSymbolIterator.
 *
 * @see llvm::object::symbols()
 */
LLVM_C_ABI LLVMSymbolIteratorRef
LLVMObjectFileCopySymbolIterator(LLVMBinaryRef BR);

/**
 * Returns whether the given symbol iterator is at the end.
 *
 * @see llvm::object::symbol_end
 */
LLVM_C_ABI LLVMBool
LLVMObjectFileIsSymbolIteratorAtEnd(LLVMBinaryRef BR, LLVMSymbolIteratorRef SI);

LLVM_C_ABI void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI);

LLVM_C_ABI void LLVMMoveToNextSection(LLVMSectionIteratorRef SI);
LLVM_C_ABI void LLVMMoveToContainingSection(LLVMSectionIteratorRef Sect,
                                            LLVMSymbolIteratorRef Sym);

// ObjectFile Symbol iterators
LLVM_C_ABI void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI);
LLVM_C_ABI void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI);

// SectionRef accessors
LLVM_C_ABI const char *LLVMGetSectionName(LLVMSectionIteratorRef SI);
LLVM_C_ABI uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI);
LLVM_C_ABI const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI);
LLVM_C_ABI uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI);
LLVM_C_ABI LLVMBool LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI,
                                                 LLVMSymbolIteratorRef Sym);

// Section Relocation iterators
LLVM_C_ABI LLVMRelocationIteratorRef
LLVMGetRelocations(LLVMSectionIteratorRef Section);
LLVM_C_ABI void LLVMDisposeRelocationIterator(LLVMRelocationIteratorRef RI);
LLVM_C_ABI LLVMBool LLVMIsRelocationIteratorAtEnd(
    LLVMSectionIteratorRef Section, LLVMRelocationIteratorRef RI);
LLVM_C_ABI void LLVMMoveToNextRelocation(LLVMRelocationIteratorRef RI);

// SymbolRef accessors
LLVM_C_ABI const char *LLVMGetSymbolName(LLVMSymbolIteratorRef SI);
LLVM_C_ABI uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI);
LLVM_C_ABI uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI);

// RelocationRef accessors
LLVM_C_ABI uint64_t LLVMGetRelocationOffset(LLVMRelocationIteratorRef RI);
LLVM_C_ABI LLVMSymbolIteratorRef
LLVMGetRelocationSymbol(LLVMRelocationIteratorRef RI);
LLVM_C_ABI uint64_t LLVMGetRelocationType(LLVMRelocationIteratorRef RI);
// NOTE: Caller takes ownership of returned string of the two
// following functions.
LLVM_C_ABI const char *LLVMGetRelocationTypeName(LLVMRelocationIteratorRef RI);
LLVM_C_ABI const char *
LLVMGetRelocationValueString(LLVMRelocationIteratorRef RI);

/** Deprecated: Use LLVMBinaryRef instead. */
typedef struct LLVMOpaqueObjectFile *LLVMObjectFileRef;

/** Deprecated: Use LLVMCreateBinary instead. */
LLVM_C_ABI LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf);

/** Deprecated: Use LLVMDisposeBinary instead. */
LLVM_C_ABI void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile);

/** Deprecated: Use LLVMObjectFileCopySectionIterator instead. */
LLVM_C_ABI LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile);

/** Deprecated: Use LLVMObjectFileIsSectionIteratorAtEnd instead. */
LLVM_C_ABI LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                               LLVMSectionIteratorRef SI);

/** Deprecated: Use LLVMObjectFileCopySymbolIterator instead. */
LLVM_C_ABI LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef ObjectFile);

/** Deprecated: Use LLVMObjectFileIsSymbolIteratorAtEnd instead. */
LLVM_C_ABI LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                              LLVMSymbolIteratorRef SI);
/**
 * @}
 */

LLVM_C_EXTERN_C_END

#endif
