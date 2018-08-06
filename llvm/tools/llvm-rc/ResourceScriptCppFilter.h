//===-- ResourceScriptCppFilter.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This filters the input to llvm-rc for preprocessor markers, removing
// preprocessing directives that a preprocessor can output or leave behind.
//
// It also filters out any contribution from files named *.h or *.c, based
// on preprocessor line markers. When preprocessing RC files, the included
// headers can leave behind C declarations, that RC doesn't understand.
// Rc.exe simply discards anything that comes from files named *.h or *.h.
//
// https://msdn.microsoft.com/en-us/library/windows/desktop/aa381033(v=vs.85).aspx
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMRC_RESOURCESCRIPTCPPFILTER_H
#define LLVM_TOOLS_LLVMRC_RESOURCESCRIPTCPPFILTER_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {

std::string filterCppOutput(StringRef Input);

} // namespace llvm

#endif
