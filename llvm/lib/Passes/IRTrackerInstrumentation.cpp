//===- IRTrackerInstrumentation.cpp - IR tracker recorder -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/IRTrackerInstrumentation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// CLI option (the recorder owns its own flag so this TU is self-contained).
//===----------------------------------------------------------------------===//

static cl::opt<std::string> IRTrackerOutput(
    "ir-tracker-output",
    cl::desc("IR tracker: per-pass IR snapshot output path (TSV row format)"),
    cl::value_desc("file"), cl::init(""), cl::Hidden);

namespace {

//===----------------------------------------------------------------------===//
// Local copies of small helpers shared with StandardInstrumentations.cpp.
// Duplicated here so this TU is self-contained.
//===----------------------------------------------------------------------===//

template <typename IRUnitT> static const IRUnitT *unwrapIR(Any IR) {
  const IRUnitT **IRPtr = llvm::any_cast<const IRUnitT *>(&IR);
  return IRPtr ? *IRPtr : nullptr;
}

static std::string getIRName(Any IR) {
  if (unwrapIR<Module>(IR))
    return "[module]";
  if (const auto *F = unwrapIR<Function>(IR))
    return F->getName().str();
  if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR))
    return C->getName();
  if (const auto *L = unwrapIR<Loop>(IR))
    return "loop %" + L->getName().str() + " in function " +
           L->getHeader()->getParent()->getName().str();
  // Unknown IR-unit type. Mirrors the closed-set fallthrough in
  // shouldPrintIR: degrade gracefully (empty name -> P row carries an
  // empty ir_unit field) rather than aborting opt. The recorder is not
  // load-bearing for compilation correctness, so a hard crash here is
  // the wrong tradeoff if the new pass manager later grows a fifth IR
  // unit type.
  return {};
}

static bool moduleContainsFilterPrintFunc(const Module &M) {
  return any_of(M.functions(),
                [](const Function &F) {
                  return isFunctionInPrintList(F.getName());
                }) ||
         isFunctionInPrintList("*");
}

static bool sccContainsFilterPrintFunc(const LazyCallGraph::SCC &C) {
  return any_of(C,
                [](const LazyCallGraph::Node &N) {
                  return isFunctionInPrintList(N.getName());
                }) ||
         isFunctionInPrintList("*");
}

static bool shouldPrintIR(Any IR) {
  if (const auto *M = unwrapIR<Module>(IR))
    return moduleContainsFilterPrintFunc(*M);
  if (const auto *F = unwrapIR<Function>(IR))
    return isFunctionInPrintList(F->getName());
  if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR))
    return sccContainsFilterPrintFunc(*C);
  if (const auto *L = unwrapIR<Loop>(IR))
    return isFunctionInPrintList(L->getHeader()->getParent()->getName());
  return false;
}

static bool isIgnored(StringRef PassID) {
  return isSpecialPass(PassID,
                       {"PassManager", "PassAdaptor", "AnalysisManagerProxy",
                        "DevirtSCCRepeatedPass", "ModuleInlinerWrapperPass",
                        "VerifierPass", "PrintModulePass", "PrintMIRPass",
                        "PrintMIRPreparePass"});
}

//===----------------------------------------------------------------------===//
// Auto-synthesis of missing DILocations.
//
// Purpose: give every instruction a stable identifier that survives
// pass-driven cloning, moving, and rewriting, so the recorder can
// associate instructions that originate from the same source point across
// passes. The recorder uses an instruction's DILocation as that
// identifier -- when a later pass copies or rewrites an instruction, the
// resulting instruction's DILocation tells us which original instruction
// it descends from. Inputs without -g (bitcode, clang JIT, lld embedding,
// hand-written .ll) have no DILocations on most instructions, so the
// recorder loses identity across passes. This function fills the gaps
// with synthesized ordinal IDs on first sight per module.
//
// What it adds: for each function that lacks a DISubprogram, one synthetic
// DISubprogram whose file is "<ir-tracker-synthetic>"; and for each
// instruction that lacks a DebugLoc, a DILocation(line=N, column=0,
// scope=that DISubprogram) where N is a module-wide running counter.
// Real DI is never overwritten -- functions that already have a
// DISubprogram and instructions that already have a DebugLoc are skipped.
//
// Example. Input without debug info:
//
//   define i32 @add(i32 %a, i32 %b) {
//     %s = add i32 %a, %b
//     ret i32 %s
//   }
//
// After synthesizeMissingInstructionLocs:
//
//   define i32 @add(i32 %a, i32 %b) !dbg !2 {
//     %s = add i32 %a, %b, !dbg !3
//     ret i32 %s, !dbg !4
//   }
//   !1 = !DIFile(filename: "<ir-tracker-synthetic>", directory: ".")
//   !2 = distinct !DISubprogram(name: "add", scope: !0, file: !1, line: 1,
//                               ...)
//   !3 = !DILocation(line: 1, column: 0, scope: !2)
//   !4 = !DILocation(line: 2, column: 0, scope: !2)
//
// The line numbers are arbitrary running ordinals; they do not refer to
// anything in the original source. They only need to be stable across
// passes so the recorder can match a later pass's DILocation against an
// earlier one and conclude "same logical instruction".
//
// No-op when the module already has llvm.dbg.cu (a real -g build), or
// when the module has no functions.
//===----------------------------------------------------------------------===//

static void synthesizeMissingInstructionLocs(Module &M) {
  // Skip if the module already has real debug info; we never want to
  // override real DI.
  if (M.getNamedMetadata("llvm.dbg.cu"))
    return;
  // Skip empty modules (no functions to attach locs to).
  if (M.empty())
    return;

  if (!M.getModuleFlag("Debug Info Version"))
    M.addModuleFlag(Module::Warning, "Debug Info Version",
                    DEBUG_METADATA_VERSION);

  DIBuilder DIB(M);
  LLVMContext &Ctx = M.getContext();
  DIFile *File = DIB.createFile("<ir-tracker-synthetic>", ".");
  DICompileUnit *CU = DIB.createCompileUnit(
      dwarf::DW_LANG_C, File, "ir-tracker",
      /*isOptimized=*/true, "", /*RV=*/0, /*SplitName=*/"",
      DICompileUnit::FullDebug);
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  unsigned NextOrdinal = 1;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.getSubprogram())
      continue;
    unsigned FuncLine = NextOrdinal;
    DISubprogram::DISPFlags SPFlags =
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized;
    if (F.hasPrivateLinkage() || F.hasInternalLinkage())
      SPFlags |= DISubprogram::SPFlagLocalToUnit;
    DISubprogram *SP =
        DIB.createFunction(CU, F.getName(), F.getName(), File, FuncLine,
                           SPType, FuncLine, DINode::FlagZero, SPFlags);
    F.setSubprogram(SP);

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (I.getDebugLoc())
          continue;
        I.setDebugLoc(DILocation::get(Ctx, NextOrdinal, 0, SP));
        ++NextOrdinal;
      }
    }
  }
  DIB.finalize();
}

//===----------------------------------------------------------------------===//
// Recorder implementation.
//===----------------------------------------------------------------------===//

static std::string getIRTrackerFilePath(const DILocation *Loc) {
  if (!Loc)
    return {};

  StringRef Dir = Loc->getDirectory();
  StringRef File = Loc->getFilename();
  if (File.empty())
    return {};
  if (Dir.empty())
    return File.str();

  SmallString<256> Path(Dir);
  sys::path::append(Path, File);
  return std::string(Path);
}

static stable_hash hashTrackerIdentity(const DILocation *Loc);

static stable_hash hashValueIdentity(const Value &V) {
  if (const auto *Arg = dyn_cast<Argument>(&V))
    return stable_hash_combine(static_cast<stable_hash>(1), Arg->getArgNo());
  if (const auto *GV = dyn_cast<GlobalValue>(&V))
    return stable_hash_combine(static_cast<stable_hash>(2),
                               static_cast<stable_hash>(hash_value(GV->getName())));
  if (const auto *BB = dyn_cast<BasicBlock>(&V)) {
    if (BB->hasName())
      return stable_hash_combine(static_cast<stable_hash>(3),
                                 static_cast<stable_hash>(hash_value(BB->getName())));
    return 0;
  }
  if (const auto *I = dyn_cast<Instruction>(&V)) {
    if (const DILocation *Loc = I->getDebugLoc() ? I->getDebugLoc().get() : nullptr)
      return stable_hash_combine(static_cast<stable_hash>(4),
                                 hashTrackerIdentity(Loc));
    if (I->hasName())
      return stable_hash_combine(static_cast<stable_hash>(5),
                                 static_cast<stable_hash>(hash_value(I->getName())));
    return 0;
  }
  if (V.hasName())
    return stable_hash_combine(static_cast<stable_hash>(6),
                               static_cast<stable_hash>(hash_value(V.getName())));
  return 0;
}

/// Compute a stable structural fingerprint of an instruction.
///
/// Used by the per-instruction change-detection step: at each pass, the
/// recorder rehashes every changed-block instruction and compares against
/// the previous pass's hash for the same tracker ID. Equal hash → "did
/// not change" → skip emission.
///
/// Captures: opcode, result type, operand count, per-operand type, the
/// per-operand kind bits (is-Constant, is-Argument), a stable identity for
/// non-constant operands when one is available (argument number, global name,
/// basic-block name, instruction source-point identity), the commutative flag,
/// the CmpInst predicate, and the value of any ConstantInt operand.
///
/// The important property is that operand rewrites which change the rendered
/// instruction text should usually perturb the hash as well:
///
///   %a = add i32 %x, %y         hash X
///   %b = add i32 %x, %z         hash Y      (different source-point identity)
///   %c = add i32 %x, 1          hash Y      (kind bit changed)
///   %d = add i32 %x, 2          hash Z      (ConstantInt value changed)
///   %e = sub i32 %x, %y         hash W      (opcode changed)
///   %f = icmp eq i32 %x, %y     hash V
///   %g = icmp ne i32 %x, %y     hash U      (predicate changed)
///
/// Still missed: zero-loc unnamed instruction operands that have no stable
/// identity source, ConstantFP value changes, ConstantExpr changes, and
/// attached metadata changes. Metadata tracking is opt-in via separate flags
/// (deferred to a follow-up PR).
static stable_hash hashInstruction(const Instruction &I) {
  stable_hash H = stable_hash_combine(I.getOpcode(), I.getType()->getTypeID(),
                                      I.getNumOperands());
  for (const Use &U : I.operands()) {
    Value *V = U.get();
    H = stable_hash_combine(
        H, stable_hash_combine(
               static_cast<stable_hash>(V->getType()->getTypeID()),
               static_cast<stable_hash>(isa<Constant>(V) ? 1 : 0),
               static_cast<stable_hash>(isa<Argument>(V) ? 1 : 0),
               hashValueIdentity(*V)));
    if (auto *C = dyn_cast<ConstantInt>(V))
      H = stable_hash_combine(H,
                              static_cast<stable_hash>(hash_value(C->getValue())));
  }
  if (I.isCommutative())
    H = stable_hash_combine(H, 1);
  if (auto *CI = dyn_cast<CmpInst>(&I))
    H = stable_hash_combine(H, CI->getPredicate());
  return H;
}

/// Compute the hash that identifies "this source point" for tracker-ID
/// interning.
///
/// The recorder assigns one compact integer ID to each unique source
/// point and references that ID in instruction rows so the source
/// location appears once (in a metadata row) rather than per
/// instruction row. Two DILocations should map to the same ID iff
/// they refer to the same source point.
///
/// Combines: source file, line, column, and the declared line of the
/// enclosing DISubprogram. The file participates in the key because
/// mixed-file modules can legitimately contain the same (line, col,
/// scope-line) tuple in multiple translation units; without file
/// identity those instructions alias to the same tracker ID and share
/// change-detection state incorrectly. The subprogram's declared line
/// still disambiguates instructions at the same (line, col) coming
/// from different functions in the same file:
///
///   file.c:1 in function foo (foo is declared at line 1)
///     hash = combine("file.c", 1, 0, 1)
///   file.c:1 in function bar (bar is declared at line 3)
///     hash = combine("file.c", 1, 0, 3)        // distinct
///   other.c:1 in function foo (foo is declared at line 1)
///     hash = combine("other.c", 1, 0, 1)       // distinct
///
/// Without the scope-line component, inlined or template-instantiated
/// instructions at the same (line, col) would alias and be treated as
/// the same source point. Including the DISubprogram pointer directly
/// would make the hash run-unstable (pointer addresses are not stable
/// across LLVM invocations); using the file path plus the subprogram's
/// declared line gives us the needed disambiguation while keeping the
/// hash deterministic.
///
/// Returns 0 for a null DILocation. The recorder uses 0 as a sentinel
/// "no real source point" and routes such instructions through a
/// per-block temp-ID fallback path (synthesized phi nodes, etc.).
static stable_hash hashTrackerIdentity(const DILocation *Loc) {
  if (!Loc)
    return 0;
  stable_hash FileKey =
      stable_hash_combine(static_cast<stable_hash>(hash_value(Loc->getDirectory())),
                          static_cast<stable_hash>(hash_value(Loc->getFilename())));
  unsigned ScopeLine = 0;
  if (DISubprogram *SP = Loc->getScope()->getSubprogram())
    ScopeLine = SP->getLine();
  return stable_hash_combine(FileKey, Loc->getLine(), Loc->getColumn(),
                             ScopeLine);
}

static void printAPIntValue(raw_ostream &OS, const APInt &V) {
  SmallString<32> Tmp;
  if (V.isNegative())
    V.toStringSigned(Tmp, 10);
  else
    V.toStringUnsigned(Tmp, 10);
  OS << Tmp;
}

class IRTrackerRecorder {
  /// Open file handle for the TSV row stream. Opened by the constructor;
  /// closed when the IRTrackerRecorder is destroyed.
  std::unique_ptr<raw_fd_ostream> OS;

  /// Pass-record sequence number assigned to the next P row. Starts at 1
  /// because seq=0 is reserved for the implicit initial-capture pass that
  /// records the IR before any user pass runs.
  unsigned NextSeq = 1;

  /// Guard so the initial IR is captured exactly once, on the first
  /// non-skipped beforePass we see. Set to true after the initial capture
  /// runs.
  bool InitialCaptured = false;

  /// Counter that allocates fresh tracker IDs. ID 0 is the sentinel for
  /// "no real source location"; real IDs start at 1.
  unsigned NextTrackerID = 1;

  /// Cached pointer to the last module observed by afterPass, used by the
  /// destructor to emit a final full-instruction snapshot so downstream
  /// tooling can reconstruct the post-optimization IR from the stream
  /// alone (no separate -S re-run required).
  const Module *LastModule = nullptr;

  /// Set of modules whose missing DILocations have already been
  /// synthesized by ensureSyntheticLocs. Each module gets the synthesis
  /// walk at most once per recorder lifetime.
  DenseSet<const Module *> ModulesWithSynthesizedLocs;

  /// Per-function combined structural hash. Equal value across passes
  /// means the function produced the same per-block instruction hashes,
  /// so the detailed emission walk can be skipped.
  DenseMap<const Function *, stable_hash> FunctionHashes;

  /// Per-function vector of full per-block hashes. Indexed positionally
  /// by basic-block order. Recomputed every pass so function-level
  /// equality has no false negatives at block granularity.
  DenseMap<const Function *, SmallVector<stable_hash>> BlockHashes;

  /// Per-function, per-block bool flag: "does this block contain any
  /// instruction without a DILocation?" Indexed positionally. Drives the
  /// per-instruction temp-ID fallback path for zero-loc instructions
  /// (phi nodes, LCSSA-inserted ops).
  DenseMap<const Function *, SmallVector<bool>> BlockHasZeroIDs;

  /// Per-function, per-block, per-instruction structural hash. Used by
  /// the per-instruction skip layer ("emit a row only for instructions
  /// whose hash changed since last pass"). Outer index is the block
  /// position; inner index is the instruction position within the block.
  DenseMap<const Function *, SmallVector<SmallVector<stable_hash>>>
      BlockInstHashes;

  /// Per-function, per-block, per-instruction tracker ID. Same indexing
  /// as BlockInstHashes. Stores the temp ID assigned to each zero-loc
  /// instruction in a block so the recorder can refer to it consistently
  /// across passes.
  DenseMap<const Function *, SmallVector<SmallVector<unsigned>>> BlockTempIDs;

  /// Intern table from a source-point hash (hashTrackerIdentity) to the
  /// compact integer tracker ID. First time a source point is seen, a
  /// fresh ID is allocated; subsequent sightings of the same source
  /// point return the existing ID, giving stable cross-pass identity.
  DenseMap<stable_hash, unsigned> LocKeyToTrackerID;

  /// Per-tracker-ID memory of the last instruction-hash we emitted for
  /// that ID. The change-detection check is "is the current instruction
  /// hash equal to the value stored here?"; if equal, skip emission.
  DenseMap<unsigned, stable_hash> TrackerIDToPrevHash;

  /// Set of tracker IDs we have already emitted a T (metadata) row for.
  /// Used by writeTrackerRecord to dedup so each unique source point
  /// appears in exactly one T row over the recorder's lifetime.
  DenseSet<unsigned> EmittedTrackerMetadata;

  /// Emit one P (pass) row. P rows delimit the per-pass instruction
  /// records that follow.
  ///
  /// Format: ``P\t<seq>\t<phase>\t<pass_name>\t<ir_unit>``.
  ///
  /// * ``seq``: monotonically increasing pass index. 0 is the initial
  ///   capture, 1..N are normal passes, and one final "phase=final"
  ///   record is emitted at teardown.
  /// * ``phase``: ``initial``, ``after``, or ``final``.
  /// * ``pass_name``: pass class name as resolved by
  ///   PassInstrumentationCallbacks::getPassNameForClassName.
  /// * ``ir_unit``: human-readable IR unit name from getIRName ("[module]",
  ///   a function name, an SCC name, or "loop %X in function Y").
  ///
  /// Example output:
  ///
  ///   P\t0\tinitial\t<initial>\t[module]
  ///   P\t1\tafter\tmemprof-remove-attributes\t[module]
  ///   P\t5\tafter\tsroa\tcli_wcwidth
  void writePassRecord(unsigned Seq, StringRef Phase, StringRef PassName,
                       StringRef IRUnit) {
    *OS << "P\t" << Seq << '\t' << Phase << '\t' << PassName << '\t' << IRUnit
        << '\n';
  }

  /// Emit one T (tracker-metadata) row, at most once per tracker ID over
  /// the recorder's lifetime.
  ///
  /// Each T row binds a tracker ID to its source location so I rows
  /// (instruction rows) can reference the ID compactly without repeating
  /// the file/line/col text. The dedup is via EmittedTrackerMetadata; a
  /// second call with the same ID is a silent no-op.
  ///
  /// Format: ``T\t<id>\t<file>\t<line>\t<col>``.
  ///
  /// For instructions that have no DILocation (phi nodes, many LCSSA
  /// inserts) the recorder still allocates a temp tracker ID and calls
  /// this with Loc==nullptr. We emit the row anyway with placeholder
  /// "<synthetic>" / 0 / 0 so the downstream consumer's ID->location
  /// lookup is total -- if we silently dropped the T row, the consumer
  /// would have to special-case "I row references unknown ID".
  ///
  /// Example output:
  ///
  ///   T\t1\t/home/yaxunl/foo.c\t42\t7
  ///   T\t2\t/home/yaxunl/foo.c\t43\t3
  ///   T\t17\t<synthetic>\t0\t0
  void writeTrackerRecord(unsigned ID, const DILocation *Loc) {
    if (ID == 0 || !EmittedTrackerMetadata.insert(ID).second)
      return;
    std::string FilePath = Loc ? getIRTrackerFilePath(Loc) : "<synthetic>";
    unsigned LineN = Loc ? Loc->getLine() : 0;
    unsigned ColN = Loc ? Loc->getColumn() : 0;
    *OS << "T\t" << ID << '\t' << FilePath << '\t' << LineN << '\t'
        << ColN << '\n';
  }

  /// Look up (or assign) the tracker ID for a source location.
  ///
  /// Two DILocations referring to the same source point (per
  /// hashTrackerIdentity) return the same ID. The first sighting of a
  /// previously-unseen source point allocates and returns the next
  /// available ID. A null Loc returns the sentinel value 0, signaling
  /// "no real source point" -- the caller is responsible for routing
  /// such instructions through the per-block temp-ID fallback.
  ///
  /// Example. With LocKeyToTrackerID initially empty:
  ///
  ///   getOrCreateTrackerID(loc_at_foo_c_42_7)   -> 1   (newly minted)
  ///   getOrCreateTrackerID(loc_at_foo_c_43_3)   -> 2
  ///   getOrCreateTrackerID(loc_at_foo_c_42_7)   -> 1   (reused)
  ///   getOrCreateTrackerID(nullptr)             -> 0   (sentinel)
  ///   getOrCreateTrackerID(loc_at_bar_c_42_7)   -> 3   (different scope)
  unsigned getOrCreateTrackerID(const DILocation *Loc) {
    stable_hash Key = hashTrackerIdentity(Loc);
    if (Key == 0)
      return 0;
    auto It = LocKeyToTrackerID.find(Key);
    if (It != LocKeyToTrackerID.end())
      return It->second;
    unsigned ID = NextTrackerID++;
    LocKeyToTrackerID[Key] = ID;
    return ID;
  }

  /// Record one function's per-pass instruction-level diff into the TSV
  /// stream. Called from writeIR for every function reachable from the
  /// pass's IR unit.
  ///
  /// SkipUnchanged controls the change-detection mode:
  ///
  /// * SkipUnchanged=false: emit every instruction in the function (used
  ///   for the initial capture and the destructor's final snapshot).
  /// * SkipUnchanged=true (the normal per-pass path): only emit
  ///   instructions whose structural hash changed since the last
  ///   recorded pass.
  ///
  /// The implementation is staged so that cheap equality checks can skip
  /// the more expensive detailed emission work:
  ///
  ///   1. Recompute one structural hash per block from the current IR.
  ///      This is the conservative step that guarantees no false
  ///      negatives at block granularity.
  ///   2. Function-level hash skip. If the combined function hash
  ///      matches the previous pass's, return without emitting anything.
  ///   3. Per-block changed-or-not list (ChangedBlocks). Only blocks whose
  ///      full hash changed enter the detailed emission loop.
  ///   4. Per-instruction hash skip. For
  ///      each changed block, emit an I row only for instructions whose
  ///      structural hash differs from TrackerIDToPrevHash.
  ///
  /// Example. Pipeline with three passes A, B, C on a function with two
  /// blocks bb0, bb1, where pass B rewrites only bb1:
  ///
  ///   pass A (initial, SkipUnchanged=false):
  ///     emits I rows for every instruction in bb0 and bb1.
  ///   pass B (after, SkipUnchanged=true):
  ///     bb0 full block hash matches -> skipped.
  ///     bb1 full block hash differs -> walked, changed instructions emitted.
  ///   pass C (after, SkipUnchanged=true):
  ///     both block hashes match -> function-level FuncH matches ->
  ///     return immediately. Zero rows.
  void writeInstructionsInFunction(const Function &F, bool SkipUnchanged) {
    if (F.isDeclaration() || !isFunctionInPrintList(F.getName()))
      return;

    auto &PrevBlkH = BlockHashes[&F];
    auto &PrevBlkHasZeroIDs = BlockHasZeroIDs[&F];
    auto &PrevInstH = BlockInstHashes[&F];
    auto &PrevTempIDs = BlockTempIDs[&F];
    SmallVector<stable_hash> NewBlkH;
    SmallVector<bool> NewBlkHasZeroIDs;
    SmallVector<unsigned> ChangedBlocks;
    stable_hash FuncH = 0;

    unsigned BlkIdx = 0;
    for (const BasicBlock &BB : F) {
      stable_hash BlkH = 0;
      bool HasZeroID = false;
      for (const Instruction &I : BB) {
        stable_hash H = hashInstruction(I);
        BlkH = stable_hash_combine(BlkH, H);
        if (!I.getDebugLoc())
          HasZeroID = true;
      }
      NewBlkHasZeroIDs.push_back(HasZeroID);
      NewBlkH.push_back(BlkH);
      FuncH = stable_hash_combine(FuncH, BlkH);
      if (!SkipUnchanged || BlkIdx >= PrevBlkH.size() ||
          PrevBlkH[BlkIdx] != BlkH)
        ChangedBlocks.push_back(BlkIdx);
      ++BlkIdx;
    }

    if (SkipUnchanged) {
      auto It = FunctionHashes.find(&F);
      if (It != FunctionHashes.end() && It->second == FuncH) {
        return;
      }
      FunctionHashes[&F] = FuncH;
    } else {
      FunctionHashes[&F] = FuncH;
    }

    if (ChangedBlocks.empty()) {
      PrevBlkHasZeroIDs = std::move(NewBlkHasZeroIDs);
      PrevBlkH = std::move(NewBlkH);
      return;
    }

    // Local state shared by the three printer lambdas defined below.
    StringRef FunctionName = F.getName();
    // Reusable per-instruction text buffer; avoids reallocating per emit.
    SmallString<256> InstBuf;
    // Per-function intern table for the %u<N> fallback (used when a value
    // has no name, no global address, and no DILocation we can derive a
    // tracker ID from).
    DenseMap<const Value *, unsigned> LocalValueNames;
    unsigned NextLocalValueName = 0;

    /// Return a short text token for any Value reference, in priority
    /// order: named global -> ``@name``; named BB -> ``%name``; unnamed
    /// Argument -> ``%<argNo>``; any value with a name -> ``%name``;
    /// instruction with a DILocation -> ``%t<trackerID>`` (stable
    /// across passes); anything else -> ``%u<N>`` from the per-function
    /// fallback table.
    ///
    /// The ``%t<N>`` case is the key one: it gives instructions a
    /// stable handle that survives SSA renames, so the recorded text
    /// can be diffed across passes meaningfully.
    auto getValueName = [&](const Value *V) -> std::string {
      if (auto *GV = dyn_cast<GlobalValue>(V)) {
        if (GV->hasName())
          return (Twine("@") + GV->getName()).str();
      }
      if (auto *BB = dyn_cast<BasicBlock>(V)) {
        if (BB->hasName())
          return (Twine("%") + BB->getName()).str();
      }
      // Render unnamed function arguments using the textual-IR convention
      // (``%0``, ``%1``, ...) instead of the per-emission ``%u<N>`` fallback,
      // since the argument index is stable across passes.
      if (auto *Arg = dyn_cast<Argument>(V)) {
        if (!Arg->hasName())
          return (Twine("%") + Twine(Arg->getArgNo())).str();
      }
      if (V->hasName())
        return (Twine("%") + V->getName()).str();
      if (auto *I = dyn_cast<Instruction>(V)) {
        if (const DILocation *Loc =
                I->getDebugLoc() ? I->getDebugLoc().get() : nullptr) {
          unsigned ID = getOrCreateTrackerID(Loc);
          if (ID != 0)
            return (Twine("%t") + Twine(ID)).str();
        }
      }
      if (auto It = LocalValueNames.find(V); It != LocalValueNames.end())
        return (Twine("%u") + Twine(It->second)).str();
      unsigned ID = NextLocalValueName++;
      LocalValueNames[V] = ID;
      return (Twine("%u") + Twine(ID)).str();
    };

    /// Write one operand reference to ``OS``. Handles literal
    /// constants (ConstantInt, ConstantFP, null/undef/poison,
    /// zeroinitializer, string ConstantDataArray) inline; for
    /// everything else, falls through to ``getValueName``.
    ///
    /// Stored as ``std::function`` rather than ``auto`` so
    /// ``printInstructionText`` below can name its type when it
    /// captures it.
    std::function<void(raw_ostream &, const Value *)> writeValueRef =
        [&](raw_ostream &OS, const Value *V) {
          if (auto *CI = dyn_cast<ConstantInt>(V)) {
            printAPIntValue(OS, CI->getValue());
            return;
          }
          if (auto *CF = dyn_cast<ConstantFP>(V)) {
            SmallString<32> Tmp;
            CF->getValueAPF().toString(Tmp);
            OS << Tmp;
            return;
          }
          if (isa<ConstantPointerNull>(V)) {
            OS << "null";
            return;
          }
          if (isa<UndefValue>(V)) {
            OS << "undef";
            return;
          }
          if (isa<PoisonValue>(V)) {
            OS << "poison";
            return;
          }
          if (isa<ConstantAggregateZero>(V)) {
            OS << "zeroinitializer";
            return;
          }
          if (auto *CA = dyn_cast<ConstantDataArray>(V)) {
            if (CA->isString()) {
              OS << "c\"";
              printEscapedString(CA->getAsString(), OS);
              OS << "\"";
              return;
            }
          }
          OS << getValueName(V);
        };

    /// Format one instruction into ``OS`` as the structural-form text
    /// that lands in the I row. Walks the opcode-specific branches
    /// (ret, br, switch, phi, alloca, gep, load, store, call, invoke,
    /// cmp) to lay out operands in canonical order, then a generic
    /// fallback for arithmetic / cast / etc.
    ///
    /// Deliberately omits attribute lists, alignment suffixes,
    /// attached metadata, sync scope, atomic ordering, GEP inrange
    /// markers, fast-math flags, and ``tail`` markers -- the recorder
    /// records structural shape, not full LLVM IR text. The omissions
    /// are what keep the emitted text compact and inexpensive to produce.
    ///
    /// CurID is the current instruction's tracker ID; passed in so
    /// printInstructionText can put ``%t<CurID>`` as the result name
    /// for instructions that have a real DILocation.
    auto printInstructionText = [&](raw_ostream &OS, const Instruction &I,
                                    unsigned CurID) {
      if (!I.getType()->isVoidTy()) {
        if (I.hasName())
          OS << "%" << I.getName();
        else if (CurID != 0)
          OS << "%t" << CurID;
        else
          OS << getValueName(&I);
        OS << " = ";
      }

      OS << I.getOpcodeName();
      if (const auto *CI = dyn_cast<CmpInst>(&I))
        OS << ' ' << CI->getPredicate();

      if (const auto *RI = dyn_cast<ReturnInst>(&I)) {
        if (RI->getNumOperands() == 0) {
          OS << " void";
          return;
        }
        OS << ' ';
        RI->getReturnValue()->getType()->print(OS);
        OS << ' ';
        writeValueRef(OS, RI->getReturnValue());
        return;
      }

      if (const auto *BI = dyn_cast<BranchInst>(&I)) {
        if (BI->isUnconditional()) {
          OS << ' ';
          writeValueRef(OS, BI->getSuccessor(0));
        } else {
          OS << ' ';
          writeValueRef(OS, BI->getCondition());
          OS << ", ";
          writeValueRef(OS, BI->getSuccessor(0));
          OS << ", ";
          writeValueRef(OS, BI->getSuccessor(1));
        }
        return;
      }

      if (const auto *PN = dyn_cast<PHINode>(&I)) {
        OS << ' ';
        I.getType()->print(OS);
        bool First = true;
        for (unsigned Idx = 0; Idx < PN->getNumIncomingValues(); ++Idx) {
          OS << (First ? ' ' : ',');
          if (!First)
            OS << ' ';
          First = false;
          OS << "[ ";
          writeValueRef(OS, PN->getIncomingValue(Idx));
          OS << ", ";
          writeValueRef(OS, PN->getIncomingBlock(Idx));
          OS << " ]";
        }
        return;
      }

      if (const auto *CB = dyn_cast<CallBase>(&I)) {
        if (!CB->getType()->isVoidTy()) {
          OS << ' ';
          CB->getType()->print(OS);
        }
        OS << ' ';
        writeValueRef(OS, CB->getCalledOperand());
        OS << '(';
        for (unsigned Idx = 0; Idx < CB->arg_size(); ++Idx) {
          if (Idx)
            OS << ", ";
          writeValueRef(OS, CB->getArgOperand(Idx));
        }
        OS << ')';
        return;
      }

      if (I.getNumOperands()) {
        if (!I.getType()->isVoidTy()) {
          OS << ' ';
          I.getType()->print(OS);
        }
        OS << ' ';
        for (unsigned Idx = 0; Idx < I.getNumOperands(); ++Idx) {
          if (Idx)
            OS << ", ";
          writeValueRef(OS, I.getOperand(Idx));
        }
      }
    };
    BlkIdx = 0;
    unsigned ChangedBlockPos = 0;

    for (const BasicBlock &BB : F) {
      bool BlockChanged = ChangedBlockPos < ChangedBlocks.size() &&
                          ChangedBlocks[ChangedBlockPos] == BlkIdx;

      // Detailed per-instruction emission for one changed block.
      //
      // For each instruction we:
      //   1. Compute its structural hash (CurH).
      //   2. Resolve a stable CurID -- either from the DILocation
      //      (real tracker ID) or via the per-block temp-ID matching
      //      heuristic for zero-loc instructions (phi, LCSSA inserts).
      //   3. Decide if the instruction changed since last pass
      //      (InstChanged) by comparing CurH against the recorded
      //      previous hash for CurID (or against the same-position
      //      previous hash for unmatched zero-loc temps).
      //   4. If changed, format the instruction text via the
      //      lightweight printer and emit one I row.
      //   5. Update TrackerIDToPrevHash so the next pass's
      //      change-detection sees CurH as "previous".
      //
      // OldInstH / OldTempIDs point into the previous pass's per-block
      // vectors when available; nullptr on first pass or when caller
      // requested SkipUnchanged=false.
      if (BlockChanged) {
        ++ChangedBlockPos;
        StringRef BBLabel =
            BB.hasName() ? BB.getName() : StringRef("<unnamed>");
        bool NeedFallback = NewBlkHasZeroIDs[BlkIdx];
        SmallVector<stable_hash> CurInstH;
        SmallVector<unsigned> CurTempIDs;
        auto *OldInstH = (SkipUnchanged && BlkIdx < PrevInstH.size())
                             ? &PrevInstH[BlkIdx]
                             : nullptr;
        auto *OldTempIDs = (SkipUnchanged && BlkIdx < PrevTempIDs.size())
                               ? &PrevTempIDs[BlkIdx]
                               : nullptr;
        // Marks which previous-pass temp IDs have already been matched
        // to a current-pass instruction; prevents one previous ID from
        // being claimed by two current instructions.
        SmallVector<bool> UsedOldTempIDs;
        if (OldTempIDs)
          UsedOldTempIDs.assign(OldTempIDs->size(), false);
        unsigned InstSeq = 0;
        unsigned InstIdx = 0;
        for (Instruction &I : const_cast<BasicBlock &>(BB)) {
          stable_hash CurH = hashInstruction(I);
          if (NeedFallback)
            CurInstH.push_back(CurH);
          const DILocation *Loc =
              I.getDebugLoc() ? I.getDebugLoc().get() : nullptr;
          unsigned CurID = getOrCreateTrackerID(Loc);
          // Zero-loc instruction: try to inherit a previous-pass temp
          // ID so the same logical phi/LCSSA insert keeps a stable
          // identity across passes. Two-stage match:
          //   (a) Fast path: same position, unused, matching hash.
          //   (b) Slow path: nearest unused position with matching
          //       hash; reject ties to avoid guessing.
          // Fall back to allocating a fresh tracker ID if no match.
          if (CurID == 0) {
            int MatchedIdx = -1;
            if (OldTempIDs && OldInstH) {
              if (InstIdx < OldTempIDs->size() && InstIdx < OldInstH->size() &&
                  (*OldTempIDs)[InstIdx] != 0 && !UsedOldTempIDs[InstIdx] &&
                  (*OldInstH)[InstIdx] == CurH) {
                MatchedIdx = InstIdx;
              } else {
                int BestIdx = -1;
                int BestDist = std::numeric_limits<int>::max();
                bool AmbiguousBest = false;
                for (size_t J = 0,
                            E = std::min(OldTempIDs->size(), OldInstH->size());
                     J != E; ++J) {
                  if ((*OldTempIDs)[J] == 0 || UsedOldTempIDs[J] ||
                      (*OldInstH)[J] != CurH)
                    continue;
                  int Dist =
                      std::abs(static_cast<int>(J) - static_cast<int>(InstIdx));
                  if (Dist < BestDist) {
                    BestDist = Dist;
                    BestIdx = static_cast<int>(J);
                    AmbiguousBest = false;
                  } else if (Dist == BestDist) {
                    AmbiguousBest = true;
                  }
                }
                if (BestIdx >= 0 && !AmbiguousBest)
                  MatchedIdx = BestIdx;
              }
            }
            if (MatchedIdx >= 0) {
              CurID = (*OldTempIDs)[MatchedIdx];
              UsedOldTempIDs[MatchedIdx] = true;
            } else {
              CurID = NextTrackerID++;
            }
            CurTempIDs.push_back(CurID);
          } else if (NeedFallback) {
            CurTempIDs.push_back(0);
          }
          bool InstChanged = true;
          if (CurID != 0) {
            auto It = TrackerIDToPrevHash.find(CurID);
            InstChanged = It == TrackerIDToPrevHash.end() || It->second != CurH;
          } else {
            InstChanged = !OldInstH || InstIdx >= OldInstH->size() ||
                          (*OldInstH)[InstIdx] != CurH;
          }

          if (InstChanged) {
            InstBuf.clear();
            raw_svector_ostream IOS(InstBuf);
            printInstructionText(IOS, I, CurID);

            if (CurID != 0)
              writeTrackerRecord(CurID, Loc);

            *OS << "I\t" << FunctionName << '\t' << BBLabel << '\t' << InstSeq
                << '\t' << I.getOpcodeName() << '\t' << CurID << '\t' << InstBuf
                << '\n';
          }
          if (CurID != 0)
            TrackerIDToPrevHash[CurID] = CurH;
          ++InstSeq;
          ++InstIdx;
        }
        if (NeedFallback) {
          // Per-block writeback (only for blocks with zero-loc
          // instructions). The next pass will read PrevInstH and
          // PrevTempIDs to drive the temp-ID matching heuristic.
          // Resize on demand to handle functions whose block count
          // grew since the last pass.
          if (BlkIdx >= PrevInstH.size())
            PrevInstH.resize(BlkIdx + 1);
          PrevInstH[BlkIdx] = std::move(CurInstH);
          if (BlkIdx >= PrevTempIDs.size())
            PrevTempIDs.resize(BlkIdx + 1);
          PrevTempIDs[BlkIdx] = std::move(CurTempIDs);
        }
      }
      ++BlkIdx;
    }
    // Per-function writeback. PrevBlkH / PrevBlkHasZeroIDs
    // are references into the per-function DenseMaps grabbed at the top
    // of writeInstructionsInFunction; moving into them
    // mutates the maps directly, so the next pass on this function sees
    // the fresh state.
    PrevBlkHasZeroIDs = std::move(NewBlkHasZeroIDs);
    PrevBlkH = std::move(NewBlkH);
  }

  /// Emit one P row for the pass and dispatch the per-function recording
  /// work to writeInstructionsInFunction for every function reachable
  /// from the IR unit.
  ///
  /// The new pass manager passes IR through a type-erased Any container
  /// that may hold a Module*, Function*, LazyCallGraph::SCC*, or Loop*.
  /// We unwrap to whichever one applies and walk it accordingly.
  ///
  /// Loops are recorded at function granularity rather than at loop
  /// granularity because writeInstructionsInFunction's change-detection
  /// state is keyed on the function.
  ///
  /// Example. Suppose the pipeline contains a Module pass MP and a
  /// Function pass FP, and the module has functions foo, bar, baz:
  ///
  ///   writeIR(IR=Module, ...) for MP
  ///     -> P row, then writeInstructionsInFunction(foo / bar / baz)
  ///   writeIR(IR=Function foo, ...) for FP
  ///     -> P row, then writeInstructionsInFunction(foo)
  void writeIR(Any IR, unsigned Seq, StringRef Phase, StringRef PassName,
               StringRef IRUnit, bool SkipUnchanged) {
    writePassRecord(Seq, Phase, PassName, IRUnit);
    if (const auto *M = unwrapIR<Module>(IR)) {
      for (const Function &F : *M)
        writeInstructionsInFunction(F, SkipUnchanged);
      return;
    }
    if (const auto *F = unwrapIR<Function>(IR)) {
      writeInstructionsInFunction(*F, SkipUnchanged);
      return;
    }
    if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      for (const LazyCallGraph::Node &N : *C)
        writeInstructionsInFunction(N.getFunction(), SkipUnchanged);
      return;
    }
    if (const auto *L = unwrapIR<Loop>(IR))
      writeInstructionsInFunction(*L->getHeader()->getParent(), SkipUnchanged);
  }

  /// Return true iff every function in the given IR unit has already
  /// been seen and recorded at least once (i.e., has an entry in
  /// FunctionHashes). Used by afterPass for the C4 short-circuit:
  ///
  ///   if (PA.areAllPreserved() && allFunctionsKnown(IR))
  ///     write a P row and return; skip the per-function walk.
  ///
  /// PreservedAnalyses::areAllPreserved() means the pass declared it
  /// preserved everything (typically: did not transform the IR). When
  /// combined with allFunctionsKnown, we have a proof that there is
  /// nothing for the recorder to capture this pass.
  ///
  /// The allFunctionsKnown gate is what prevents the short-circuit from
  /// silently dropping the very first encounter of a function: until the
  /// first writeInstructionsInFunction call has populated FunctionHashes
  /// for F, we cannot skip a pass on F even if PA says nothing changed,
  /// because we have no recorded baseline yet.
  bool allFunctionsKnown(Any IR) {
    if (const auto *M = unwrapIR<Module>(IR)) {
      for (const Function &F : *M)
        if (!F.isDeclaration() && !FunctionHashes.count(&F))
          return false;
      return true;
    }
    if (const auto *F = unwrapIR<Function>(IR))
      return F->isDeclaration() || FunctionHashes.count(F);
    if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      for (const LazyCallGraph::Node &N : *C)
        if (!FunctionHashes.count(&N.getFunction()))
          return false;
      return true;
    }
    if (const auto *L = unwrapIR<Loop>(IR))
      return FunctionHashes.count(L->getHeader()->getParent());
    return false;
  }

public:
  /// Open the TSV output file. report_fatal_error if the path is not
  /// writable -- there is nothing useful the recorder can do without
  /// a working output sink.
  explicit IRTrackerRecorder(StringRef Path) {
    std::error_code EC;
    OS = std::make_unique<raw_fd_ostream>(Path, EC, sys::fs::OF_Text);
    if (EC)
      report_fatal_error(Twine("ir-tracker output open: ") + EC.message());
  }

  // Emit one synthetic "final" pass record on teardown covering every
  // function of the last-seen module with ``SkipUnchanged=false`` so the
  // post-optimization IR is fully materialized in the DB. The pass
  // manager's ``PassInstrumentationCallbacks`` (which owns the shared_ptr
  // holding this object) is destroyed before the module in the standard
  // opt/clang flow, so ``LastModule`` is still live at this point. The
  // cache reset is required because writeInstructionsInFunction also
  // short-circuits per-instruction on matching ``TrackerIDToPrevHash``
  // even when ``SkipUnchanged`` is false; without the reset the final
  // record would only contain whatever changed since the last pass.
  ~IRTrackerRecorder() {
    if (!LastModule)
      return;
    FunctionHashes.clear();
    BlockHashes.clear();
    BlockHasZeroIDs.clear();
    BlockInstHashes.clear();
    BlockTempIDs.clear();
    TrackerIDToPrevHash.clear();
    writePassRecord(NextSeq++, "final", "<final>", "[module]");
    for (const Function &F : *LastModule)
      writeInstructionsInFunction(F, /*SkipUnchanged=*/false);
  }

  // Before any other beforePass work runs, ensure the module containing
  // this IR unit has DILocations on every instruction. No-op if the
  // module already has real debug info or if we have already synthesized
  // for this module.
  void ensureSyntheticLocs(Any IR) {
    Module *M = nullptr;
    if (const auto *MM = unwrapIR<Module>(IR))
      M = const_cast<Module *>(MM);
    else if (const auto *F = unwrapIR<Function>(IR))
      M = const_cast<Module *>(F->getParent());
    else if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      if (C->begin() != C->end())
        M = const_cast<Module *>(C->begin()->getFunction().getParent());
    } else if (const auto *L = unwrapIR<Loop>(IR))
      M = const_cast<Module *>(
          L->getHeader()->getParent()->getParent());
    if (!M)
      return;
    if (!ModulesWithSynthesizedLocs.insert(M).second)
      return;
    synthesizeMissingInstructionLocs(*M);
  }

  /// Pre-pass callback. Skip wrapper passes and IR units filtered by
  /// -filter-print-funcs, then ensure the enclosing module has DILocations
  /// on every instruction (one-shot per module). On the very first
  /// non-skipped invocation, emit the seq=0 "initial" snapshot so the
  /// stream has a baseline against which subsequent per-pass diffs make
  /// sense; subsequent invocations are no-ops.
  void beforePass(StringRef PassID, Any IR) {
    if (isIgnored(PassID) || !shouldPrintIR(IR))
      return;
    ensureSyntheticLocs(IR);
    if (InitialCaptured)
      return;
    InitialCaptured = true;
    writeIR(IR, 0, "initial", "<initial>", getIRName(IR),
            /*SkipUnchanged=*/false);
  }

  /// Post-pass callback -- the per-pass workhorse. After the same two
  /// filters as beforePass, cache the enclosing module so the destructor
  /// can locate the final snapshot, resolve a friendly pass name, then
  /// either (C4 short-circuit) emit a bare P row when the pass preserved
  /// everything and we already have a baseline for every function, or
  /// dispatch to writeIR with SkipUnchanged=true so only changed blocks
  /// of changed functions show up as I rows.
  void afterPass(StringRef PassID, Any IR, PassInstrumentationCallbacks &PIC,
                 const PreservedAnalyses &PA) {
    if (isIgnored(PassID) || !shouldPrintIR(IR))
      return;

    // Track the enclosing module so the destructor can dump a final
    // full-instruction snapshot regardless of the IR unit type this pass
    // saw.
    if (const auto *M = unwrapIR<Module>(IR))
      LastModule = M;
    else if (const auto *F = unwrapIR<Function>(IR))
      LastModule = F->getParent();
    else if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      if (C->begin() != C->end())
        LastModule = C->begin()->getFunction().getParent();
    } else if (const auto *L = unwrapIR<Loop>(IR))
      LastModule = L->getHeader()->getParent()->getParent();

    StringRef PassName = PIC.getPassNameForClassName(PassID);
    if (PassName.empty())
      PassName = PassID;

    if (PA.areAllPreserved() && allFunctionsKnown(IR)) {
      writePassRecord(NextSeq++, "after", PassName, getIRName(IR));
      return;
    }

    writeIR(IR, NextSeq++, "after", PassName, getIRName(IR),
            /*SkipUnchanged=*/true);
  }
};

} // namespace

void IRTrackerInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  StringRef Path = IRTrackerOutput;
  if (Path.empty())
    return;

  auto State = std::make_shared<IRTrackerRecorder>(Path);
  PIC.registerBeforeNonSkippedPassCallback(
      [State](StringRef PassID, Any IR) { State->beforePass(PassID, IR); });
  PIC.registerAfterPassCallback(
      [State, &PIC](StringRef PassID, Any IR, const PreservedAnalyses &PA) {
        State->afterPass(PassID, IR, PIC, PA);
      });
}
