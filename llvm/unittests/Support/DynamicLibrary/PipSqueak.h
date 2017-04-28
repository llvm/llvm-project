//===- llvm/unittest/Support/DynamicLibrary/PipSqueak.h -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PIPSQUEAK_H
#define LLVM_PIPSQUEAK_H

#ifdef _WIN32
#define PIPSQUEAK_EXPORT __declspec(dllexport)
#else
#define PIPSQUEAK_EXPORT
#endif

#endif
