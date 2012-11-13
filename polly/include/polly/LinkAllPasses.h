//===- polly/LinkAllPasses.h ------------ Reference All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_LINKALLPASSES_H
#define POLLY_LINKALLPASSES_H

#include "polly/Config/config.h"
#include <cstdlib>

namespace llvm {
  class Pass;
  class PassInfo;
  class PassRegistry;
  class RegionPass;
}

namespace polly {
#ifdef CLOOG_FOUND
  llvm::Pass *createCloogExporterPass();
  llvm::Pass *createCloogInfoPass();
  llvm::Pass *createCodeGenerationPass();
#endif
  llvm::Pass *createCodePreparationPass();
  llvm::Pass *createDeadCodeElimPass();
  llvm::Pass *createDependencesPass();
  llvm::Pass *createDOTOnlyPrinterPass();
  llvm::Pass *createDOTOnlyViewerPass();
  llvm::Pass *createDOTPrinterPass();
  llvm::Pass *createDOTViewerPass();
  llvm::Pass *createIndependentBlocksPass();
  llvm::Pass *createIndVarSimplifyPass();
  llvm::Pass *createJSONExporterPass();
  llvm::Pass *createJSONImporterPass();
#ifdef PLUTO_FOUND
  llvm::Pass *createPlutoOptimizerPass();
#endif
  llvm::Pass *createRegionSimplifyPass();
  llvm::Pass *createScopDetectionPass();
  llvm::Pass *createScopInfoPass();
  llvm::Pass *createIslAstInfoPass();
  llvm::Pass *createIslCodeGenerationPass();
  llvm::Pass *createIslScheduleOptimizerPass();
  llvm::Pass *createTempScopInfoPass();

#ifdef OPENSCOP_FOUND
  llvm::Pass *createScopExporterPass();
  llvm::Pass *createScopImporterPass();
#endif

#ifdef SCOPLIB_FOUND
  llvm::Pass *createPoccPass();
  llvm::Pass *createScopLibExporterPass();
  llvm::Pass *createScopLibImporterPass();
#endif

  extern char &IndependentBlocksID;
  extern char &CodePreparationID;
}

using namespace polly;

namespace {
  struct PollyForcePassLinking {
    PollyForcePassLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

#ifdef CLOOG_FOUND
       createCloogExporterPass();
       createCloogInfoPass();
       createCodeGenerationPass();
#endif
       createCodePreparationPass();
       createDeadCodeElimPass();
       createDependencesPass();
       createDOTOnlyPrinterPass();
       createDOTOnlyViewerPass();
       createDOTPrinterPass();
       createDOTViewerPass();
       createIndependentBlocksPass();
       createIndVarSimplifyPass();
       createJSONExporterPass();
       createJSONImporterPass();
       createRegionSimplifyPass();
       createScopDetectionPass();
       createScopInfoPass();
#ifdef PLUTO_FOUND
       createPlutoOptimizerPass();
#endif
       createIslAstInfoPass();
       createIslCodeGenerationPass();
       createIslScheduleOptimizerPass();
       createTempScopInfoPass();

#ifdef OPENSCOP_FOUND
       createScopExporterPass();
       createScopImporterPass();
#endif
#ifdef SCOPLIB_FOUND
       createPoccPass();
       createScopLibExporterPass();
       createScopLibImporterPass();
#endif

    }
  } PollyForcePassLinking; // Force link by creating a global definition.
}

namespace llvm {
  class PassRegistry;
#ifdef CLOOG_FOUND
  void initializeCodeGenerationPass(llvm::PassRegistry&);
#endif
  void initializeCodePreparationPass(llvm::PassRegistry&);
  void initializeDeadCodeElimPass(llvm::PassRegistry&);
  void initializeIndependentBlocksPass(llvm::PassRegistry&);
  void initializeJSONExporterPass(llvm::PassRegistry&);
  void initializeJSONImporterPass(llvm::PassRegistry&);
  void initializeIslAstInfoPass(llvm::PassRegistry&);
  void initializeIslCodeGenerationPass(llvm::PassRegistry&);
  void initializeIslScheduleOptimizerPass(llvm::PassRegistry&);
#ifdef PLUTO_FOUND
  void initializePlutoOptimizerPass(llvm::PassRegistry&);
#endif
#ifdef SCOPLIB_FOUND
  void initializePoccPass(llvm::PassRegistry&);
#endif
  void initializePollyIndVarSimplifyPass(llvm::PassRegistry&);
  void initializeRegionSimplifyPass(llvm::PassRegistry&);
}

#endif
