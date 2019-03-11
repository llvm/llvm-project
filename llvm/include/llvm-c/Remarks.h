/*===-- llvm-c/Remarks.h - Remarks Public C Interface -------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a public interface to a remark diagnostics library.   *|
|* LLVM provides an implementation of this interface.                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_REMARKS_H
#define LLVM_C_REMARKS_H

#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif /* !defined(__cplusplus) */

/**
 * @defgroup LLVMCREMARKS Remarks
 * @ingroup LLVMC
 *
 * @{
 */

#define REMARKS_API_VERSION 0

/**
 * String containing a buffer and a length. The buffer is not guaranteed to be
 * zero-terminated.
 *
 * \since REMARKS_API_VERSION=0
 */
typedef struct {
  const char *Str;
  uint32_t Len;
} LLVMRemarkStringRef;

/**
 * DebugLoc containing File, Line and Column.
 *
 * \since REMARKS_API_VERSION=0
 */
typedef struct {
  // File:
  LLVMRemarkStringRef SourceFile;
  // Line:
  uint32_t SourceLineNumber;
  // Column:
  uint32_t SourceColumnNumber;
} LLVMRemarkDebugLoc;

/**
 * Element of the "Args" list. The key might give more information about what
 * are the semantics of the value, e.g. "Callee" will tell you that the value
 * is a symbol that names a function.
 *
 * \since REMARKS_API_VERSION=0
 */
typedef struct {
  // e.g. "Callee"
  LLVMRemarkStringRef Key;
  // e.g. "malloc"
  LLVMRemarkStringRef Value;

  // "DebugLoc": Optional
  LLVMRemarkDebugLoc DebugLoc;
} LLVMRemarkArg;

/**
 * One remark entry.
 *
 * \since REMARKS_API_VERSION=0
 */
typedef struct {
  // e.g. !Missed, !Passed
  LLVMRemarkStringRef RemarkType;
  // "Pass": Required
  LLVMRemarkStringRef PassName;
  // "Name": Required
  LLVMRemarkStringRef RemarkName;
  // "Function": Required
  LLVMRemarkStringRef FunctionName;

  // "DebugLoc": Optional
  LLVMRemarkDebugLoc DebugLoc;
  // "Hotness": Optional
  uint32_t Hotness;
  // "Args": Optional. It is an array of `num_args` elements.
  uint32_t NumArgs;
  LLVMRemarkArg *Args;
} LLVMRemarkEntry;

typedef struct LLVMRemarkOpaqueParser *LLVMRemarkParserRef;

/**
 * Creates a remark parser that can be used to read and parse the buffer located
 * in \p Buf of size \p Size.
 *
 * \p Buf cannot be NULL.
 *
 * This function should be paired with LLVMRemarkParserDispose() to avoid
 * leaking resources.
 *
 * \since REMARKS_API_VERSION=0
 */
extern LLVMRemarkParserRef LLVMRemarkParserCreate(const void *Buf,
                                                  uint64_t Size);

/**
 * Returns the next remark in the file.
 *
 * The value pointed to by the return value is invalidated by the next call to
 * LLVMRemarkParserGetNext().
 *
 * If the parser reaches the end of the buffer, the return value will be NULL.
 *
 * In the case of an error, the return value will be NULL, and:
 *
 * 1) LLVMRemarkParserHasError() will return `1`.
 *
 * 2) LLVMRemarkParserGetErrorMessage() will return a descriptive error
 *    message.
 *
 * An error may occur if:
 *
 * 1) An argument is invalid.
 *
 * 2) There is a YAML parsing error. This type of error aborts parsing
 *    immediately and returns `1`. It can occur on malformed YAML.
 *
 * 3) Remark parsing error. If this type of error occurs, the parser won't call
 *    the handler and will continue to the next one. It can occur on malformed
 *    remarks, like missing or extra fields in the file.
 *
 * Here is a quick example of the usage:
 *
 * ```
 * LLVMRemarkParserRef Parser = LLVMRemarkParserCreate(Buf, Size);
 * LLVMRemarkEntry *Remark = NULL;
 * while ((Remark == LLVMRemarkParserGetNext(Parser))) {
 *    // use Remark
 * }
 * bool HasError = LLVMRemarkParserHasError(Parser);
 * LLVMRemarkParserDispose(Parser);
 * ```
 *
 * \since REMARKS_API_VERSION=0
 */
extern LLVMRemarkEntry *LLVMRemarkParserGetNext(LLVMRemarkParserRef Parser);

/**
 * Returns `1` if the parser encountered an error while parsing the buffer.
 *
 * \since REMARKS_API_VERSION=0
 */
extern LLVMBool LLVMRemarkParserHasError(LLVMRemarkParserRef Parser);

/**
 * Returns a null-terminated string containing an error message.
 *
 * In case of no error, the result is `NULL`.
 *
 * The memory of the string is bound to the lifetime of \p Parser. If
 * LLVMRemarkParserDispose() is called, the memory of the string will be
 * released.
 *
 * \since REMARKS_API_VERSION=0
 */
extern const char *LLVMRemarkParserGetErrorMessage(LLVMRemarkParserRef Parser);

/**
 * Releases all the resources used by \p Parser.
 *
 * \since REMARKS_API_VERSION=0
 */
extern void LLVMRemarkParserDispose(LLVMRemarkParserRef Parser);

/**
 * Returns the version of the remarks dylib.
 *
 * \since REMARKS_API_VERSION=0
 */
extern uint32_t LLVMRemarkVersion(void);

/**
 * @} // endgoup LLVMCREMARKS
 */

#ifdef __cplusplus
}
#endif /* !defined(__cplusplus) */

#endif /* LLVM_C_REMARKS_H */
