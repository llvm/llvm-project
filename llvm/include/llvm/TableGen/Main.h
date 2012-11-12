//===- llvm/TableGen/Main.h - tblgen entry point ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the common entry point for tblgen tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_MAIN_H
#define LLVM_TABLEGEN_MAIN_H

namespace llvm {

class RecordKeeper;
class raw_ostream;
/// \brief Perform the action using Records, and write output to OS.
/// \returns true on error, false otherwise
typedef bool TableGenMainFn(raw_ostream &OS, RecordKeeper &Records);

int TableGenMain(char *argv0, TableGenMainFn *MainFn);

}

#endif
