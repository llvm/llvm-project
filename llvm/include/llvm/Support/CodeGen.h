//===-- llvm/Support/CodeGen.h - CodeGen Concepts ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file define some types which define code generation concepts. For
// example, relocation model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CODEGEN_H
#define LLVM_SUPPORT_CODEGEN_H

namespace llvm {

  // Relocation model types.
  namespace Reloc {
    // Cannot be named PIC due to collision with -DPIC
    enum Model { Static, PIC_, DynamicNoPIC, ROPI, RWPI, ROPI_RWPI };
  }

  // Code model types.
  namespace CodeModel {
    // Sync changes with CodeGenCWrappers.h.
    enum Model { Tiny, Small, Kernel, Medium, Large };
  }

  namespace PICLevel {
    // This is used to map -fpic/-fPIC.
    enum Level { NotPIC=0, SmallPIC=1, BigPIC=2 };
  }

  namespace PIELevel {
    enum Level { Default=0, Small=1, Large=2 };
  }

  // TLS models.
  namespace TLSModel {
    enum Model {
      GeneralDynamic,
      LocalDynamic,
      InitialExec,
      LocalExec
    };
  }

  // Code generation optimization level.
  namespace CodeGenOpt {
    enum Level {
      None = 0,      // -O0
      Less = 1,      // -O1
      Default = 2,   // -O2, -Os
      Aggressive = 3 // -O3
    };
  }

  /// These enums are meant to be passed into addPassesToEmitFile to indicate
  /// what type of file to emit, and returned by it to indicate what type of
  /// file could actually be made.
  enum CodeGenFileType {
    CGFT_AssemblyFile,
    CGFT_ObjectFile,
    CGFT_Null         // Do not emit any output.
  };

  // Specify what functions should keep the frame pointer.
  enum class FramePointerKind { None, NonLeaf, All };

}  // end llvm namespace

#endif
