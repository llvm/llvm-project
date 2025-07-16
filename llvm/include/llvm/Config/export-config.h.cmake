/*===------- llvm/Config/export-config.h - export configuration ----- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/


/* This file contains configuration definitions controling which LLVM projects
   and components require public ABI export for shared library builds. It also
   contains generic LLVM_INTERFACE_ macro definitions that can be aliased by
   libraries and components and used to decorate their public interface. */

#ifndef LLVM_EXPORT_CONFIG_H
#define LLVM_EXPORT_CONFIG_H

/* Define if exporting LLVM public interface for shared library */
#cmakedefine LLVM_ENABLE_LLVM_EXPORT_ANNOTATIONS

/* Define if exporting LLVM-C public interface for shared library */
#cmakedefine LLVM_ENABLE_LLVM_C_EXPORT_ANNOTATIONS

/// LLVM_INTERFACE_ABI is the main export/visibility macro to mark something as explicitly
/// exported when llvm is built as a shared library with everything else that is
/// unannotated will have internal visibility.
///
/// LLVM_INTERFACE_ABI_EXPORT is for the special case for things like plugin symbol
/// declarations or definitions where we don't want the macro to be switching
/// between dllexport and dllimport on windows based on what codebase is being
/// built, it will only be dllexport. For non windows platforms this macro
/// behaves the same as LLVM_ABI.
///
/// LLVM_INTERFACE_EXPORT_TEMPLATE is used on explicit template instantiations in source
/// files that were declared extern in a header. This macro is only set as a
/// compiler export attribute on windows, on other platforms it does nothing.
///
/// LLVM_INTERFACE_TEMPLATE_ABI is for annotating extern template declarations in headers
/// for both functions and classes. On windows its turned in to dllimport for
/// library consumers, for other platforms its a default visibility attribute.
///
/// LLVM_INTERFACE_ABI_FOR_TEST is for annotating symbols that are only exported because
/// they are imported from a test. These symbols are not technically part of the
/// LLVM public interface and could be conditionally excluded when not building
/// tests in the future.
///
// Marker to add to classes or functions in public headers that should not have
// export macros added to them by the clang tool
#define LLVM_INTERFACE_ABI_NOT_EXPORTED
// TODO(https://github.com/llvm/llvm-project/issues/145406): eliminate need for
// two preprocessor definitions to gate LLVM_ABI macro definitions.
#if !defined(LLVM_BUILD_STATIC)
#if defined(_WIN32) && !defined(__MINGW32__)
#if defined(LLVM_EXPORTS)
#define LLVM_INTERFACE_ABI __declspec(dllexport)
#define LLVM_INTERFACE_TEMPLATE_ABI
#define LLVM_INTERFACE_EXPORT_TEMPLATE __declspec(dllexport)
#else
#define LLVM_INTERFACE_ABI __declspec(dllimport)
#define LLVM_INTERFACE_TEMPLATE_ABI __declspec(dllimport)
#define LLVM_INTERFACE_EXPORT_TEMPLATE
#endif
#define LLVM_INTERFACE_ABI_EXPORT __declspec(dllexport)
#elif defined(__has_attribute) && __has_attribute(visibility)
#if defined(__ELF__) || defined(__MINGW32__) || defined(_AIX) ||               \
    defined(__MVS__) || defined(__CYGWIN__)
#define LLVM_INTERFACE_ABI __attribute__((visibility("default")))
#define LLVM_INTERFACE_TEMPLATE_ABI LLVM_INTERFACE_ABI
#define LLVM_INTERFACE_EXPORT_TEMPLATE
#define LLVM_INTERFACE_ABI_EXPORT LLVM_INTERFACE_ABI
#elif defined(__MACH__) || defined(__WASM__) || defined(__EMSCRIPTEN__)
#define LLVM_INTERFACE_ABI __attribute__((visibility("default")))
#define LLVM_INTERFACE_TEMPLATE_ABI
#define LLVM_INTERFACE_EXPORT_TEMPLATE
#define LLVM_INTERFACE_ABI_EXPORT LLVM_INTERFACE_ABI
#endif
#endif
#endif
#if !defined(LLVM_INTERFACE_ABI)
#define LLVM_INTERFACE_ABI
#define LLVM_INTERFACE_TEMPLATE_ABI
#define LLVM_INTERFACE_EXPORT_TEMPLATE
#define LLVM_INTERFACE_ABI_EXPORT
#endif
#define LLVM_INTERFACE_ABI_FOR_TEST LLVM_INTERFACE_ABI

#endif
