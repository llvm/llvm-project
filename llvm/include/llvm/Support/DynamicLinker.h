//===-- llvm/Support/DynamicLinker.h - Portable Dynamic Linker --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Lightweight interface to dynamic library linking and loading, and dynamic
// symbol lookup functionality, in whatever form the operating system
// provides it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DYNAMICLINKER_H
#define LLVM_SUPPORT_DYNAMICLINKER_H

#include <string>

namespace llvm {

/// LinkDynamicObject - Load the named file as a dynamic library
/// and link it with the currently running process. Returns false
/// on success, true if there is an error (and sets ErrorMessage
/// if it is not NULL). Analogous to dlopen().
///
bool LinkDynamicObject (const char *filename, std::string *ErrorMessage);

/// GetAddressOfSymbol - Returns the address of the named symbol in
/// the currently running process, as reported by the dynamic linker,
/// or NULL if the symbol does not exist or some other error has
/// occurred.
///
void *GetAddressOfSymbol (const char *symbolName);
void *GetAddressOfSymbol (const std::string &symbolName);

} // End llvm namespace

#endif // SUPPORT_DYNAMICLINKER_H
