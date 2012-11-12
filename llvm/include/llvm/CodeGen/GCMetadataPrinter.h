//===-- llvm/CodeGen/GCMetadataPrinter.h - Prints asm GC tables -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The abstract base class GCMetadataPrinter supports writing GC metadata tables
// as assembly code. This is a separate class from GCStrategy in order to allow
// users of the LLVM JIT to avoid linking with the AsmWriter.
//
// Subclasses of GCMetadataPrinter must be registered using the
// GCMetadataPrinterRegistry. This is separate from the GCStrategy itself
// because these subclasses are logically plugins for the AsmWriter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GCMETADATAPRINTER_H
#define LLVM_CODEGEN_GCMETADATAPRINTER_H

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/Support/Registry.h"

namespace llvm {

  class GCMetadataPrinter;

  /// GCMetadataPrinterRegistry - The GC assembly printer registry uses all the
  /// defaults from Registry.
  typedef Registry<GCMetadataPrinter> GCMetadataPrinterRegistry;

  /// GCMetadataPrinter - Emits GC metadata as assembly code.
  ///
  class GCMetadataPrinter {
  public:
    typedef GCStrategy::list_type list_type;
    typedef GCStrategy::iterator iterator;

  private:
    GCStrategy *S;

    friend class AsmPrinter;

  protected:
    // May only be subclassed.
    GCMetadataPrinter();

  private:
    GCMetadataPrinter(const GCMetadataPrinter &) LLVM_DELETED_FUNCTION;
    GCMetadataPrinter &
      operator=(const GCMetadataPrinter &) LLVM_DELETED_FUNCTION;

  public:
    GCStrategy &getStrategy() { return *S; }
    const Module &getModule() const { return S->getModule(); }

    /// begin/end - Iterate over the collected function metadata.
    iterator begin() { return S->begin(); }
    iterator end()   { return S->end();   }

    /// beginAssembly/finishAssembly - Emit module metadata as assembly code.
    virtual void beginAssembly(AsmPrinter &AP);

    virtual void finishAssembly(AsmPrinter &AP);

    virtual ~GCMetadataPrinter();
  };

}

#endif
