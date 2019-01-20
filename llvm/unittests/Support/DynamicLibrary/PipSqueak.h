//===- llvm/unittest/Support/DynamicLibrary/PipSqueak.h -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PIPSQUEAK_H
#define LLVM_PIPSQUEAK_H

#if defined(_WIN32) && !defined(__GNUC__)
// Disable warnings from inclusion of xlocale & exception
#pragma warning(push)
#pragma warning(disable: 4530)
#pragma warning(disable: 4577)
#include <string>
#include <vector>
#pragma warning(pop)
#else
#include <string>
#include <vector>
#endif

#ifdef _WIN32
#define PIPSQUEAK_EXPORT __declspec(dllexport)
#else
#define PIPSQUEAK_EXPORT
#endif

extern "C" PIPSQUEAK_EXPORT const char *TestA();

#endif
