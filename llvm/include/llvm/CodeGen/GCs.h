//===-- GCs.h - Garbage collector linkage hacks ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains hack functions to force linking in the GC components.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GCS_H
#define LLVM_CODEGEN_GCS_H

namespace llvm {
  class GCStrategy;
  class GCMetadataPrinter;
  
  /// FIXME: Collector instances are not useful on their own. These no longer
  ///        serve any purpose except to link in the plugins.
  
  /// Creates an ocaml-compatible garbage collector.
  void linkOcamlGC();
  
  /// Creates an ocaml-compatible metadata printer.
  void linkOcamlGCPrinter();
  
  /// Creates a shadow stack garbage collector. This collector requires no code
  /// generator support.
  void linkShadowStackGC();
}

#endif
