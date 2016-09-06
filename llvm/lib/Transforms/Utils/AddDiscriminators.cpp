//===- AddDiscriminators.cpp - Insert DWARF path discriminators -----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file adds DWARF discriminators to the IR. Path discriminators are
// used to decide what CFG path was taken inside sub-graphs whose instructions
// share the same line and column number information.
//
// The main user of this is the sample profiler. Instruction samples are
// mapped to line number information. Since a single line may be spread
// out over several basic blocks, discriminators add more precise location
// for the samples.
//
// For example,
//
//   1  #define ASSERT(P)
//   2      if (!(P))
//   3        abort()
//   ...
//   100   while (true) {
//   101     ASSERT (sum < 0);
//   102     ...
//   130   }
//
// when converted to IR, this snippet looks something like:
//
// while.body:                                       ; preds = %entry, %if.end
//   %0 = load i32* %sum, align 4, !dbg !15
//   %cmp = icmp slt i32 %0, 0, !dbg !15
//   br i1 %cmp, label %if.end, label %if.then, !dbg !15
//
// if.then:                                          ; preds = %while.body
//   call void @abort(), !dbg !15
//   br label %if.end, !dbg !15
//
// Notice that all the instructions in blocks 'while.body' and 'if.then'
// have exactly the same debug information. When this program is sampled
// at runtime, the profiler will assume that all these instructions are
// equally frequent. This, in turn, will consider the edge while.body->if.then
// to be frequently taken (which is incorrect).
//
// By adding a discriminator value to the instructions in block 'if.then',
// we can distinguish instructions at line 101 with discriminator 0 from
// the instructions at line 101 with discriminator 1.
//
// For more details about DWARF discriminators, please visit
// http://wiki.dwarfstd.org/index.php?title=Path_Discriminators
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AddDiscriminators.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "add-discriminators"

namespace {
// The legacy pass of AddDiscriminators.
struct AddDiscriminatorsLegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  AddDiscriminatorsLegacyPass() : FunctionPass(ID) {
    initializeAddDiscriminatorsLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

char AddDiscriminatorsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AddDiscriminatorsLegacyPass, "add-discriminators",
                      "Add DWARF path discriminators", false, false)
INITIALIZE_PASS_END(AddDiscriminatorsLegacyPass, "add-discriminators",
                    "Add DWARF path discriminators", false, false)

// Command line option to disable discriminator generation even in the
// presence of debug information. This is only needed when debugging
// debug info generation issues.
static cl::opt<bool> NoDiscriminators(
    "no-discriminators", cl::init(false),
    cl::desc("Disable generation of discriminator information."));

// Create the legacy AddDiscriminatorsPass.
FunctionPass *llvm::createAddDiscriminatorsPass() {
  return new AddDiscriminatorsLegacyPass();
}

/// \brief Assign DWARF discriminators.
///
/// To assign discriminators, we examine the boundaries of every
/// basic block and its successors. Suppose there is a basic block B1
/// with successor B2. The last instruction I1 in B1 and the first
/// instruction I2 in B2 are located at the same file and line number.
/// This situation is illustrated in the following code snippet:
///
///       if (i < 10) x = i;
///
///     entry:
///       br i1 %cmp, label %if.then, label %if.end, !dbg !10
///     if.then:
///       %1 = load i32* %i.addr, align 4, !dbg !10
///       store i32 %1, i32* %x, align 4, !dbg !10
///       br label %if.end, !dbg !10
///     if.end:
///       ret void, !dbg !12
///
/// Notice how the branch instruction in block 'entry' and all the
/// instructions in block 'if.then' have the exact same debug location
/// information (!dbg !10).
///
/// To distinguish instructions in block 'entry' from instructions in
/// block 'if.then', we generate a new lexical block for all the
/// instruction in block 'if.then' that share the same file and line
/// location with the last instruction of block 'entry'.
///
/// This new lexical block will have the same location information as
/// the previous one, but with a new DWARF discriminator value.
///
/// One of the main uses of this discriminator value is in runtime
/// sample profilers. It allows the profiler to distinguish instructions
/// at location !dbg !10 that execute on different basic blocks. This is
/// important because while the predicate 'if (x < 10)' may have been
/// executed millions of times, the assignment 'x = i' may have only
/// executed a handful of times (meaning that the entry->if.then edge is
/// seldom taken).
///
/// If we did not have discriminator information, the profiler would
/// assign the same weight to both blocks 'entry' and 'if.then', which
/// in turn will make it conclude that the entry->if.then edge is very
/// hot.
///
/// To decide where to create new discriminator values, this function
/// traverses the CFG and examines instruction at basic block boundaries.
/// If the last instruction I1 of a block B1 is at the same file and line
/// location as instruction I2 of successor B2, then it creates a new
/// lexical block for I2 and all the instruction in B2 that share the same
/// file and line location as I2. This new lexical block will have a
/// different discriminator number than I1.
static bool addDiscriminators(Function &F) {
  // If the function has debug information, but the user has disabled
  // discriminators, do nothing.
  // Simlarly, if the function has no debug info, do nothing.
  // Finally, if this module is built with dwarf versions earlier than 4,
  // do nothing (discriminator support is a DWARF 4 feature).
  if (NoDiscriminators || !F.getSubprogram() ||
      F.getParent()->getDwarfVersion() < 4)
    return false;

  bool Changed = false;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  DIBuilder Builder(*M, /*AllowUnresolved*/ false);

  typedef std::pair<StringRef, unsigned> Location;
  typedef DenseMap<const BasicBlock *, Metadata *> BBScopeMap;
  typedef DenseMap<Location, BBScopeMap> LocationBBMap;
  typedef DenseMap<Location, unsigned> LocationDiscriminatorMap;
  typedef DenseSet<Location> LocationSet;

  LocationBBMap LBM;
  LocationDiscriminatorMap LDM;

  // Traverse all instructions in the function. If the source line location
  // of the instruction appears in other basic block, assign a new
  // discriminator for this instruction.
  for (BasicBlock &B : F) {
    for (auto &I : B.getInstList()) {
      if (isa<IntrinsicInst>(&I))
        continue;
      const DILocation *DIL = I.getDebugLoc();
      if (!DIL)
        continue;
      Location L = std::make_pair(DIL->getFilename(), DIL->getLine());
      auto &BBMap = LBM[L];
      auto R = BBMap.insert(std::make_pair(&B, (Metadata *)nullptr));
      if (BBMap.size() == 1)
        continue;
      bool InsertSuccess = R.second;
      Metadata *&NewScope = R.first->second;
      // If we could insert a different block in the same location, a
      // discriminator is needed to distinguish both instructions.
      if (InsertSuccess) {
        auto *Scope = DIL->getScope();
        auto *File =
            Builder.createFile(DIL->getFilename(), Scope->getDirectory());
        NewScope = Builder.createLexicalBlockFile(Scope, File, ++LDM[L]);
      }
      I.setDebugLoc(DILocation::get(Ctx, DIL->getLine(), DIL->getColumn(),
                                    NewScope, DIL->getInlinedAt()));
      DEBUG(dbgs() << DIL->getFilename() << ":" << DIL->getLine() << ":"
                   << DIL->getColumn() << ":"
                   << dyn_cast<DILexicalBlockFile>(NewScope)->getDiscriminator()
                   << I << "\n");
      Changed = true;
    }
  }

  // Traverse all instructions and assign new discriminators to call
  // instructions with the same lineno that are in the same basic block.
  // Sample base profile needs to distinguish different function calls within
  // a same source line for correct profile annotation.
  for (BasicBlock &B : F) {
    LocationSet CallLocations;
    for (auto &I : B.getInstList()) {
      CallInst *Current = dyn_cast<CallInst>(&I);
      if (!Current || isa<IntrinsicInst>(&I))
        continue;

      DILocation *CurrentDIL = Current->getDebugLoc();
      if (!CurrentDIL)
        continue;
      Location L =
          std::make_pair(CurrentDIL->getFilename(), CurrentDIL->getLine());
      if (!CallLocations.insert(L).second) {
        auto *Scope = CurrentDIL->getScope();
        auto *File = Builder.createFile(CurrentDIL->getFilename(),
                                        Scope->getDirectory());
        auto *NewScope = Builder.createLexicalBlockFile(Scope, File, ++LDM[L]);
        Current->setDebugLoc(DILocation::get(Ctx, CurrentDIL->getLine(),
                                             CurrentDIL->getColumn(), NewScope,
                                             CurrentDIL->getInlinedAt()));
        Changed = true;
      }
    }
  }
  return Changed;
}

bool AddDiscriminatorsLegacyPass::runOnFunction(Function &F) {
  return addDiscriminators(F);
}
PreservedAnalyses AddDiscriminatorsPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  if (!addDiscriminators(F))
    return PreservedAnalyses::all();

  // FIXME: should be all()
  return PreservedAnalyses::none();
}
