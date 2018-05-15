//===- GCOVProfiling.cpp - Insert edge counters for gcov profiling --------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements GCOV-style profiling. When this pass is run it emits
// "gcno" files next to the existing source, and instruments the code that runs
// to records the edges between blocks that run and emit a complementary "gcda"
// file on exit.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/GCOVProfiler.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
using namespace llvm;

#define DEBUG_TYPE "insert-gcov-profiling"

static cl::opt<std::string>
DefaultGCOVVersion("default-gcov-version", cl::init("402*"), cl::Hidden,
                   cl::ValueRequired);
static cl::opt<bool> DefaultExitBlockBeforeBody("gcov-exit-block-before-body",
                                                cl::init(false), cl::Hidden);

GCOVOptions GCOVOptions::getDefault() {
  GCOVOptions Options;
  Options.EmitNotes = true;
  Options.EmitData = true;
  Options.UseCfgChecksum = false;
  Options.NoRedZone = false;
  Options.FunctionNamesInData = true;
  Options.ExitBlockBeforeBody = DefaultExitBlockBeforeBody;

  if (DefaultGCOVVersion.size() != 4) {
    llvm::report_fatal_error(std::string("Invalid -default-gcov-version: ") +
                             DefaultGCOVVersion);
  }
  memcpy(Options.Version, DefaultGCOVVersion.c_str(), 4);
  return Options;
}

namespace {
class GCOVFunction;

class GCOVProfiler {
public:
  GCOVProfiler() : GCOVProfiler(GCOVOptions::getDefault()) {}
  GCOVProfiler(const GCOVOptions &Opts) : Options(Opts) {
    assert((Options.EmitNotes || Options.EmitData) &&
           "GCOVProfiler asked to do nothing?");
    ReversedVersion[0] = Options.Version[3];
    ReversedVersion[1] = Options.Version[2];
    ReversedVersion[2] = Options.Version[1];
    ReversedVersion[3] = Options.Version[0];
    ReversedVersion[4] = '\0';
  }
  bool runOnModule(Module &M);

private:
  // Create the .gcno files for the Module based on DebugInfo.
  void emitProfileNotes();

  // Modify the program to track transitions along edges and call into the
  // profiling runtime to emit .gcda files when run.
  bool emitProfileArcs();

  // Get pointers to the functions in the runtime library.
  Constant *getStartFileFunc();
  Constant *getIncrementIndirectCounterFunc();
  Constant *getEmitFunctionFunc();
  Constant *getEmitArcsFunc();
  Constant *getSummaryInfoFunc();
  Constant *getEndFileFunc();

  // Create or retrieve an i32 state value that is used to represent the
  // pred block number for certain non-trivial edges.
  GlobalVariable *getEdgeStateValue();

  // Produce a table of pointers to counters, by predecessor and successor
  // block number.
  GlobalVariable *buildEdgeLookupTable(Function *F, GlobalVariable *Counter,
                                       const UniqueVector<BasicBlock *> &Preds,
                                       const UniqueVector<BasicBlock *> &Succs);

  // Add the function to write out all our counters to the global destructor
  // list.
  Function *
  insertCounterWriteout(ArrayRef<std::pair<GlobalVariable *, MDNode *>>);
  Function *insertFlush(ArrayRef<std::pair<GlobalVariable *, MDNode *>>);
  void insertIndirectCounterIncrement();

  enum class GCovFileType { GCNO, GCDA };
  std::string mangleName(const DICompileUnit *CU, GCovFileType FileType);

  GCOVOptions Options;

  // Reversed, NUL-terminated copy of Options.Version.
  char ReversedVersion[5];
  // Checksum, produced by hash of EdgeDestinations
  SmallVector<uint32_t, 4> FileChecksums;

  Module *M;
  LLVMContext *Ctx;
  SmallVector<std::unique_ptr<GCOVFunction>, 16> Funcs;
};

class GCOVProfilerLegacyPass : public ModulePass {
public:
  static char ID;
  GCOVProfilerLegacyPass()
      : GCOVProfilerLegacyPass(GCOVOptions::getDefault()) {}
  GCOVProfilerLegacyPass(const GCOVOptions &Opts)
      : ModulePass(ID), Profiler(Opts) {
    initializeGCOVProfilerLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "GCOV Profiler"; }

  bool runOnModule(Module &M) override { return Profiler.runOnModule(M); }

private:
  GCOVProfiler Profiler;
};
}

char GCOVProfilerLegacyPass::ID = 0;
INITIALIZE_PASS(GCOVProfilerLegacyPass, "insert-gcov-profiling",
                "Insert instrumentation for GCOV profiling", false, false)

ModulePass *llvm::createGCOVProfilerPass(const GCOVOptions &Options) {
  return new GCOVProfilerLegacyPass(Options);
}

static StringRef getFunctionName(const DISubprogram *SP) {
  if (!SP->getLinkageName().empty())
    return SP->getLinkageName();
  return SP->getName();
}

namespace {
  class GCOVRecord {
   protected:
    static const char *const LinesTag;
    static const char *const FunctionTag;
    static const char *const BlockTag;
    static const char *const EdgeTag;

    GCOVRecord() = default;

    void writeBytes(const char *Bytes, int Size) {
      os->write(Bytes, Size);
    }

    void write(uint32_t i) {
      writeBytes(reinterpret_cast<char*>(&i), 4);
    }

    // Returns the length measured in 4-byte blocks that will be used to
    // represent this string in a GCOV file
    static unsigned lengthOfGCOVString(StringRef s) {
      // A GCOV string is a length, followed by a NUL, then between 0 and 3 NULs
      // padding out to the next 4-byte word. The length is measured in 4-byte
      // words including padding, not bytes of actual string.
      return (s.size() / 4) + 1;
    }

    void writeGCOVString(StringRef s) {
      uint32_t Len = lengthOfGCOVString(s);
      write(Len);
      writeBytes(s.data(), s.size());

      // Write 1 to 4 bytes of NUL padding.
      assert((unsigned)(4 - (s.size() % 4)) > 0);
      assert((unsigned)(4 - (s.size() % 4)) <= 4);
      writeBytes("\0\0\0\0", 4 - (s.size() % 4));
    }

    raw_ostream *os;
  };
  const char *const GCOVRecord::LinesTag = "\0\0\x45\x01";
  const char *const GCOVRecord::FunctionTag = "\0\0\0\1";
  const char *const GCOVRecord::BlockTag = "\0\0\x41\x01";
  const char *const GCOVRecord::EdgeTag = "\0\0\x43\x01";

  class GCOVFunction;
  class GCOVBlock;

  // Constructed only by requesting it from a GCOVBlock, this object stores a
  // list of line numbers and a single filename, representing lines that belong
  // to the block.
  class GCOVLines : public GCOVRecord {
   public:
    void addLine(uint32_t Line) {
      assert(Line != 0 && "Line zero is not a valid real line number.");
      Lines.push_back(Line);
    }

    uint32_t length() const {
      // Here 2 = 1 for string length + 1 for '0' id#.
      return lengthOfGCOVString(Filename) + 2 + Lines.size();
    }

    void writeOut() {
      write(0);
      writeGCOVString(Filename);
      for (int i = 0, e = Lines.size(); i != e; ++i)
        write(Lines[i]);
    }

    GCOVLines(StringRef F, raw_ostream *os)
      : Filename(F) {
      this->os = os;
    }

   private:
    StringRef Filename;
    SmallVector<uint32_t, 32> Lines;
  };


  // Represent a basic block in GCOV. Each block has a unique number in the
  // function, number of lines belonging to each block, and a set of edges to
  // other blocks.
  class GCOVBlock : public GCOVRecord {
   public:
    GCOVLines &getFile(StringRef Filename) {
      return LinesByFile.try_emplace(Filename, Filename, os).first->second;
    }

    void addEdge(GCOVBlock &Successor) {
      OutEdges.push_back(&Successor);
    }

    void writeOut() {
      uint32_t Len = 3;
      SmallVector<StringMapEntry<GCOVLines> *, 32> SortedLinesByFile;
      for (auto &I : LinesByFile) {
        Len += I.second.length();
        SortedLinesByFile.push_back(&I);
      }

      writeBytes(LinesTag, 4);
      write(Len);
      write(Number);

      llvm::sort(
          SortedLinesByFile.begin(), SortedLinesByFile.end(),
          [](StringMapEntry<GCOVLines> *LHS, StringMapEntry<GCOVLines> *RHS) {
            return LHS->getKey() < RHS->getKey();
          });
      for (auto &I : SortedLinesByFile)
        I->getValue().writeOut();
      write(0);
      write(0);
    }

    GCOVBlock(const GCOVBlock &RHS) : GCOVRecord(RHS), Number(RHS.Number) {
      // Only allow copy before edges and lines have been added. After that,
      // there are inter-block pointers (eg: edges) that won't take kindly to
      // blocks being copied or moved around.
      assert(LinesByFile.empty());
      assert(OutEdges.empty());
    }

   private:
    friend class GCOVFunction;

    GCOVBlock(uint32_t Number, raw_ostream *os)
        : Number(Number) {
      this->os = os;
    }

    uint32_t Number;
    StringMap<GCOVLines> LinesByFile;
    SmallVector<GCOVBlock *, 4> OutEdges;
  };

  // A function has a unique identifier, a checksum (we leave as zero) and a
  // set of blocks and a map of edges between blocks. This is the only GCOV
  // object users can construct, the blocks and lines will be rooted here.
  class GCOVFunction : public GCOVRecord {
   public:
     GCOVFunction(const DISubprogram *SP, Function *F, raw_ostream *os,
                  uint32_t Ident, bool UseCfgChecksum, bool ExitBlockBeforeBody)
         : SP(SP), Ident(Ident), UseCfgChecksum(UseCfgChecksum), CfgChecksum(0),
           ReturnBlock(1, os) {
      this->os = os;

      LLVM_DEBUG(dbgs() << "Function: " << getFunctionName(SP) << "\n");

      uint32_t i = 0;
      for (auto &BB : *F) {
        // Skip index 1 if it's assigned to the ReturnBlock.
        if (i == 1 && ExitBlockBeforeBody)
          ++i;
        Blocks.insert(std::make_pair(&BB, GCOVBlock(i++, os)));
      }
      if (!ExitBlockBeforeBody)
        ReturnBlock.Number = i;

      std::string FunctionNameAndLine;
      raw_string_ostream FNLOS(FunctionNameAndLine);
      FNLOS << getFunctionName(SP) << SP->getLine();
      FNLOS.flush();
      FuncChecksum = hash_value(FunctionNameAndLine);
    }

    GCOVBlock &getBlock(BasicBlock *BB) {
      return Blocks.find(BB)->second;
    }

    GCOVBlock &getReturnBlock() {
      return ReturnBlock;
    }

    std::string getEdgeDestinations() {
      std::string EdgeDestinations;
      raw_string_ostream EDOS(EdgeDestinations);
      Function *F = Blocks.begin()->first->getParent();
      for (BasicBlock &I : *F) {
        GCOVBlock &Block = getBlock(&I);
        for (int i = 0, e = Block.OutEdges.size(); i != e; ++i)
          EDOS << Block.OutEdges[i]->Number;
      }
      return EdgeDestinations;
    }

    uint32_t getFuncChecksum() {
      return FuncChecksum;
    }

    void setCfgChecksum(uint32_t Checksum) {
      CfgChecksum = Checksum;
    }

    void writeOut() {
      writeBytes(FunctionTag, 4);
      uint32_t BlockLen = 1 + 1 + 1 + lengthOfGCOVString(getFunctionName(SP)) +
                          1 + lengthOfGCOVString(SP->getFilename()) + 1;
      if (UseCfgChecksum)
        ++BlockLen;
      write(BlockLen);
      write(Ident);
      write(FuncChecksum);
      if (UseCfgChecksum)
        write(CfgChecksum);
      writeGCOVString(getFunctionName(SP));
      writeGCOVString(SP->getFilename());
      write(SP->getLine());

      // Emit count of blocks.
      writeBytes(BlockTag, 4);
      write(Blocks.size() + 1);
      for (int i = 0, e = Blocks.size() + 1; i != e; ++i) {
        write(0);  // No flags on our blocks.
      }
      LLVM_DEBUG(dbgs() << Blocks.size() << " blocks.\n");

      // Emit edges between blocks.
      if (Blocks.empty()) return;
      Function *F = Blocks.begin()->first->getParent();
      for (BasicBlock &I : *F) {
        GCOVBlock &Block = getBlock(&I);
        if (Block.OutEdges.empty()) continue;

        writeBytes(EdgeTag, 4);
        write(Block.OutEdges.size() * 2 + 1);
        write(Block.Number);
        for (int i = 0, e = Block.OutEdges.size(); i != e; ++i) {
          LLVM_DEBUG(dbgs() << Block.Number << " -> "
                            << Block.OutEdges[i]->Number << "\n");
          write(Block.OutEdges[i]->Number);
          write(0);  // no flags
        }
      }

      // Emit lines for each block.
      for (BasicBlock &I : *F)
        getBlock(&I).writeOut();
    }

   private:
     const DISubprogram *SP;
    uint32_t Ident;
    uint32_t FuncChecksum;
    bool UseCfgChecksum;
    uint32_t CfgChecksum;
    DenseMap<BasicBlock *, GCOVBlock> Blocks;
    GCOVBlock ReturnBlock;
  };
}

std::string GCOVProfiler::mangleName(const DICompileUnit *CU,
                                     GCovFileType OutputType) {
  bool Notes = OutputType == GCovFileType::GCNO;

  if (NamedMDNode *GCov = M->getNamedMetadata("llvm.gcov")) {
    for (int i = 0, e = GCov->getNumOperands(); i != e; ++i) {
      MDNode *N = GCov->getOperand(i);
      bool ThreeElement = N->getNumOperands() == 3;
      if (!ThreeElement && N->getNumOperands() != 2)
        continue;
      if (dyn_cast<MDNode>(N->getOperand(ThreeElement ? 2 : 1)) != CU)
        continue;

      if (ThreeElement) {
        // These nodes have no mangling to apply, it's stored mangled in the
        // bitcode.
        MDString *NotesFile = dyn_cast<MDString>(N->getOperand(0));
        MDString *DataFile = dyn_cast<MDString>(N->getOperand(1));
        if (!NotesFile || !DataFile)
          continue;
        return Notes ? NotesFile->getString() : DataFile->getString();
      }

      MDString *GCovFile = dyn_cast<MDString>(N->getOperand(0));
      if (!GCovFile)
        continue;

      SmallString<128> Filename = GCovFile->getString();
      sys::path::replace_extension(Filename, Notes ? "gcno" : "gcda");
      return Filename.str();
    }
  }

  SmallString<128> Filename = CU->getFilename();
  sys::path::replace_extension(Filename, Notes ? "gcno" : "gcda");
  StringRef FName = sys::path::filename(Filename);
  SmallString<128> CurPath;
  if (sys::fs::current_path(CurPath)) return FName;
  sys::path::append(CurPath, FName);
  return CurPath.str();
}

bool GCOVProfiler::runOnModule(Module &M) {
  this->M = &M;
  Ctx = &M.getContext();

  if (Options.EmitNotes) emitProfileNotes();
  if (Options.EmitData) return emitProfileArcs();
  return false;
}

PreservedAnalyses GCOVProfilerPass::run(Module &M,
                                        ModuleAnalysisManager &AM) {

  GCOVProfiler Profiler(GCOVOpts);

  if (!Profiler.runOnModule(M))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

static bool functionHasLines(Function &F) {
  // Check whether this function actually has any source lines. Not only
  // do these waste space, they also can crash gcov.
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Debug intrinsic locations correspond to the location of the
      // declaration, not necessarily any statements or expressions.
      if (isa<DbgInfoIntrinsic>(&I)) continue;

      const DebugLoc &Loc = I.getDebugLoc();
      if (!Loc)
        continue;

      // Artificial lines such as calls to the global constructors.
      if (Loc.getLine() == 0) continue;

      return true;
    }
  }
  return false;
}

static bool isUsingFuncletBasedEH(Function &F) {
  if (!F.hasPersonalityFn()) return false;

  EHPersonality Personality = classifyEHPersonality(F.getPersonalityFn());
  return isFuncletEHPersonality(Personality);
}

static bool shouldKeepInEntry(BasicBlock::iterator It) {
	if (isa<AllocaInst>(*It)) return true;
	if (isa<DbgInfoIntrinsic>(*It)) return true;
	if (auto *II = dyn_cast<IntrinsicInst>(It)) {
		if (II->getIntrinsicID() == llvm::Intrinsic::localescape) return true;
	}

	return false;
}

void GCOVProfiler::emitProfileNotes() {
  NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes) return;

  for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
    // Each compile unit gets its own .gcno file. This means that whether we run
    // this pass over the original .o's as they're produced, or run it after
    // LTO, we'll generate the same .gcno files.

    auto *CU = cast<DICompileUnit>(CU_Nodes->getOperand(i));

    // Skip module skeleton (and module) CUs.
    if (CU->getDWOId())
      continue;

    std::error_code EC;
    raw_fd_ostream out(mangleName(CU, GCovFileType::GCNO), EC, sys::fs::F_None);
    if (EC) {
      Ctx->emitError(Twine("failed to open coverage notes file for writing: ") +
                     EC.message());
      continue;
    }

    std::string EdgeDestinations;

    unsigned FunctionIdent = 0;
    for (auto &F : M->functions()) {
      DISubprogram *SP = F.getSubprogram();
      if (!SP) continue;
      if (!functionHasLines(F)) continue;
      // TODO: Functions using funclet-based EH are currently not supported.
      if (isUsingFuncletBasedEH(F)) continue;

      // gcov expects every function to start with an entry block that has a
      // single successor, so split the entry block to make sure of that.
      BasicBlock &EntryBlock = F.getEntryBlock();
      BasicBlock::iterator It = EntryBlock.begin();
      while (shouldKeepInEntry(It))
        ++It;
      EntryBlock.splitBasicBlock(It);

      Funcs.push_back(make_unique<GCOVFunction>(SP, &F, &out, FunctionIdent++,
                                                Options.UseCfgChecksum,
                                                Options.ExitBlockBeforeBody));
      GCOVFunction &Func = *Funcs.back();

      for (auto &BB : F) {
        GCOVBlock &Block = Func.getBlock(&BB);
        TerminatorInst *TI = BB.getTerminator();
        if (int successors = TI->getNumSuccessors()) {
          for (int i = 0; i != successors; ++i) {
            Block.addEdge(Func.getBlock(TI->getSuccessor(i)));
          }
        } else if (isa<ReturnInst>(TI)) {
          Block.addEdge(Func.getReturnBlock());
        }

        uint32_t Line = 0;
        for (auto &I : BB) {
          // Debug intrinsic locations correspond to the location of the
          // declaration, not necessarily any statements or expressions.
          if (isa<DbgInfoIntrinsic>(&I)) continue;

          const DebugLoc &Loc = I.getDebugLoc();
          if (!Loc)
            continue;

          // Artificial lines such as calls to the global constructors.
          if (Loc.getLine() == 0) continue;

          if (Line == Loc.getLine()) continue;
          Line = Loc.getLine();
          if (SP != getDISubprogram(Loc.getScope()))
            continue;

          GCOVLines &Lines = Block.getFile(SP->getFilename());
          Lines.addLine(Loc.getLine());
        }
      }
      EdgeDestinations += Func.getEdgeDestinations();
    }

    FileChecksums.push_back(hash_value(EdgeDestinations));
    out.write("oncg", 4);
    out.write(ReversedVersion, 4);
    out.write(reinterpret_cast<char*>(&FileChecksums.back()), 4);

    for (auto &Func : Funcs) {
      Func->setCfgChecksum(FileChecksums.back());
      Func->writeOut();
    }

    out.write("\0\0\0\0\0\0\0\0", 8);  // EOF
    out.close();
  }
}

bool GCOVProfiler::emitProfileArcs() {
  NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes) return false;

  bool Result = false;
  bool InsertIndCounterIncrCode = false;
  for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
    SmallVector<std::pair<GlobalVariable *, MDNode *>, 8> CountersBySP;
    for (auto &F : M->functions()) {
      DISubprogram *SP = F.getSubprogram();
      if (!SP) continue;
      if (!functionHasLines(F)) continue;
      // TODO: Functions using funclet-based EH are currently not supported.
      if (isUsingFuncletBasedEH(F)) continue;
      if (!Result) Result = true;

      unsigned Edges = 0;
      for (auto &BB : F) {
        TerminatorInst *TI = BB.getTerminator();
        if (isa<ReturnInst>(TI))
          ++Edges;
        else
          Edges += TI->getNumSuccessors();
      }

      ArrayType *CounterTy =
        ArrayType::get(Type::getInt64Ty(*Ctx), Edges);
      GlobalVariable *Counters =
        new GlobalVariable(*M, CounterTy, false,
                           GlobalValue::InternalLinkage,
                           Constant::getNullValue(CounterTy),
                           "__llvm_gcov_ctr");
      CountersBySP.push_back(std::make_pair(Counters, SP));

      UniqueVector<BasicBlock *> ComplexEdgePreds;
      UniqueVector<BasicBlock *> ComplexEdgeSuccs;

      unsigned Edge = 0;
      for (auto &BB : F) {
        TerminatorInst *TI = BB.getTerminator();
        int Successors = isa<ReturnInst>(TI) ? 1 : TI->getNumSuccessors();
        if (Successors) {
          if (Successors == 1) {
            IRBuilder<> Builder(&*BB.getFirstInsertionPt());
            Value *Counter = Builder.CreateConstInBoundsGEP2_64(Counters, 0,
                                                                Edge);
            Value *Count = Builder.CreateLoad(Counter);
            Count = Builder.CreateAdd(Count, Builder.getInt64(1));
            Builder.CreateStore(Count, Counter);
          } else if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
            IRBuilder<> Builder(BI);
            Value *Sel = Builder.CreateSelect(BI->getCondition(),
                                              Builder.getInt64(Edge),
                                              Builder.getInt64(Edge + 1));
            Value *Counter = Builder.CreateInBoundsGEP(
                Counters->getValueType(), Counters, {Builder.getInt64(0), Sel});
            Value *Count = Builder.CreateLoad(Counter);
            Count = Builder.CreateAdd(Count, Builder.getInt64(1));
            Builder.CreateStore(Count, Counter);
          } else {
            ComplexEdgePreds.insert(&BB);
            for (int i = 0; i != Successors; ++i)
              ComplexEdgeSuccs.insert(TI->getSuccessor(i));
          }

          Edge += Successors;
        }
      }

      if (!ComplexEdgePreds.empty()) {
        GlobalVariable *EdgeTable =
          buildEdgeLookupTable(&F, Counters,
                               ComplexEdgePreds, ComplexEdgeSuccs);
        GlobalVariable *EdgeState = getEdgeStateValue();

        for (int i = 0, e = ComplexEdgePreds.size(); i != e; ++i) {
          IRBuilder<> Builder(&*ComplexEdgePreds[i + 1]->getFirstInsertionPt());
          Builder.CreateStore(Builder.getInt32(i), EdgeState);
        }

        for (int i = 0, e = ComplexEdgeSuccs.size(); i != e; ++i) {
          // Call runtime to perform increment.
          IRBuilder<> Builder(&*ComplexEdgeSuccs[i + 1]->getFirstInsertionPt());
          Value *CounterPtrArray =
            Builder.CreateConstInBoundsGEP2_64(EdgeTable, 0,
                                               i * ComplexEdgePreds.size());

          // Build code to increment the counter.
          InsertIndCounterIncrCode = true;
          Builder.CreateCall(getIncrementIndirectCounterFunc(),
                             {EdgeState, CounterPtrArray});
        }
      }
    }

    Function *WriteoutF = insertCounterWriteout(CountersBySP);
    Function *FlushF = insertFlush(CountersBySP);

    // Create a small bit of code that registers the "__llvm_gcov_writeout" to
    // be executed at exit and the "__llvm_gcov_flush" function to be executed
    // when "__gcov_flush" is called.
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
    Function *F = Function::Create(FTy, GlobalValue::InternalLinkage,
                                   "__llvm_gcov_init", M);
    F->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    F->setLinkage(GlobalValue::InternalLinkage);
    F->addFnAttr(Attribute::NoInline);
    if (Options.NoRedZone)
      F->addFnAttr(Attribute::NoRedZone);

    BasicBlock *BB = BasicBlock::Create(*Ctx, "entry", F);
    IRBuilder<> Builder(BB);

    FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
    Type *Params[] = {
      PointerType::get(FTy, 0),
      PointerType::get(FTy, 0)
    };
    FTy = FunctionType::get(Builder.getVoidTy(), Params, false);

    // Initialize the environment and register the local writeout and flush
    // functions.
    Constant *GCOVInit = M->getOrInsertFunction("llvm_gcov_init", FTy);
    Builder.CreateCall(GCOVInit, {WriteoutF, FlushF});
    Builder.CreateRetVoid();

    appendToGlobalCtors(*M, F, 0);
  }

  if (InsertIndCounterIncrCode)
    insertIndirectCounterIncrement();

  return Result;
}

// All edges with successors that aren't branches are "complex", because it
// requires complex logic to pick which counter to update.
GlobalVariable *GCOVProfiler::buildEdgeLookupTable(
    Function *F,
    GlobalVariable *Counters,
    const UniqueVector<BasicBlock *> &Preds,
    const UniqueVector<BasicBlock *> &Succs) {
  // TODO: support invoke, threads. We rely on the fact that nothing can modify
  // the whole-Module pred edge# between the time we set it and the time we next
  // read it. Threads and invoke make this untrue.

  // emit [(succs * preds) x i64*], logically [succ x [pred x i64*]].
  size_t TableSize = Succs.size() * Preds.size();
  Type *Int64PtrTy = Type::getInt64PtrTy(*Ctx);
  ArrayType *EdgeTableTy = ArrayType::get(Int64PtrTy, TableSize);

  std::unique_ptr<Constant * []> EdgeTable(new Constant *[TableSize]);
  Constant *NullValue = Constant::getNullValue(Int64PtrTy);
  for (size_t i = 0; i != TableSize; ++i)
    EdgeTable[i] = NullValue;

  unsigned Edge = 0;
  for (BasicBlock &BB : *F) {
    TerminatorInst *TI = BB.getTerminator();
    int Successors = isa<ReturnInst>(TI) ? 1 : TI->getNumSuccessors();
    if (Successors > 1 && !isa<BranchInst>(TI) && !isa<ReturnInst>(TI)) {
      for (int i = 0; i != Successors; ++i) {
        BasicBlock *Succ = TI->getSuccessor(i);
        IRBuilder<> Builder(Succ);
        Value *Counter = Builder.CreateConstInBoundsGEP2_64(Counters, 0,
                                                            Edge + i);
        EdgeTable[((Succs.idFor(Succ) - 1) * Preds.size()) +
                  (Preds.idFor(&BB) - 1)] = cast<Constant>(Counter);
      }
    }
    Edge += Successors;
  }

  GlobalVariable *EdgeTableGV =
      new GlobalVariable(
          *M, EdgeTableTy, true, GlobalValue::InternalLinkage,
          ConstantArray::get(EdgeTableTy,
                             makeArrayRef(&EdgeTable[0],TableSize)),
          "__llvm_gcda_edge_table");
  EdgeTableGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return EdgeTableGV;
}

Constant *GCOVProfiler::getStartFileFunc() {
  Type *Args[] = {
    Type::getInt8PtrTy(*Ctx),  // const char *orig_filename
    Type::getInt8PtrTy(*Ctx),  // const char version[4]
    Type::getInt32Ty(*Ctx),    // uint32_t checksum
  };
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), Args, false);
  return M->getOrInsertFunction("llvm_gcda_start_file", FTy);
}

Constant *GCOVProfiler::getIncrementIndirectCounterFunc() {
  Type *Int32Ty = Type::getInt32Ty(*Ctx);
  Type *Int64Ty = Type::getInt64Ty(*Ctx);
  Type *Args[] = {
    Int32Ty->getPointerTo(),                // uint32_t *predecessor
    Int64Ty->getPointerTo()->getPointerTo() // uint64_t **counters
  };
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), Args, false);
  return M->getOrInsertFunction("__llvm_gcov_indirect_counter_increment", FTy);
}

Constant *GCOVProfiler::getEmitFunctionFunc() {
  Type *Args[] = {
    Type::getInt32Ty(*Ctx),    // uint32_t ident
    Type::getInt8PtrTy(*Ctx),  // const char *function_name
    Type::getInt32Ty(*Ctx),    // uint32_t func_checksum
    Type::getInt8Ty(*Ctx),     // uint8_t use_extra_checksum
    Type::getInt32Ty(*Ctx),    // uint32_t cfg_checksum
  };
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), Args, false);
  return M->getOrInsertFunction("llvm_gcda_emit_function", FTy);
}

Constant *GCOVProfiler::getEmitArcsFunc() {
  Type *Args[] = {
    Type::getInt32Ty(*Ctx),     // uint32_t num_counters
    Type::getInt64PtrTy(*Ctx),  // uint64_t *counters
  };
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), Args, false);
  return M->getOrInsertFunction("llvm_gcda_emit_arcs", FTy);
}

Constant *GCOVProfiler::getSummaryInfoFunc() {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
  return M->getOrInsertFunction("llvm_gcda_summary_info", FTy);
}

Constant *GCOVProfiler::getEndFileFunc() {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
  return M->getOrInsertFunction("llvm_gcda_end_file", FTy);
}

GlobalVariable *GCOVProfiler::getEdgeStateValue() {
  GlobalVariable *GV = M->getGlobalVariable("__llvm_gcov_global_state_pred");
  if (!GV) {
    GV = new GlobalVariable(*M, Type::getInt32Ty(*Ctx), false,
                            GlobalValue::InternalLinkage,
                            ConstantInt::get(Type::getInt32Ty(*Ctx),
                                             0xffffffff),
                            "__llvm_gcov_global_state_pred");
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  }
  return GV;
}

Function *GCOVProfiler::insertCounterWriteout(
    ArrayRef<std::pair<GlobalVariable *, MDNode *> > CountersBySP) {
  FunctionType *WriteoutFTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
  Function *WriteoutF = M->getFunction("__llvm_gcov_writeout");
  if (!WriteoutF)
    WriteoutF = Function::Create(WriteoutFTy, GlobalValue::InternalLinkage,
                                 "__llvm_gcov_writeout", M);
  WriteoutF->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  WriteoutF->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    WriteoutF->addFnAttr(Attribute::NoRedZone);

  BasicBlock *BB = BasicBlock::Create(*Ctx, "entry", WriteoutF);
  IRBuilder<> Builder(BB);

  Constant *StartFile = getStartFileFunc();
  Constant *EmitFunction = getEmitFunctionFunc();
  Constant *EmitArcs = getEmitArcsFunc();
  Constant *SummaryInfo = getSummaryInfoFunc();
  Constant *EndFile = getEndFileFunc();

  NamedMDNode *CUNodes = M->getNamedMetadata("llvm.dbg.cu");
  if (!CUNodes) {
    Builder.CreateRetVoid();
    return WriteoutF;
  }

  // Collect the relevant data into a large constant data structure that we can
  // walk to write out everything.
  StructType *StartFileCallArgsTy = StructType::create(
      {Builder.getInt8PtrTy(), Builder.getInt8PtrTy(), Builder.getInt32Ty()});
  StructType *EmitFunctionCallArgsTy = StructType::create(
      {Builder.getInt32Ty(), Builder.getInt8PtrTy(), Builder.getInt32Ty(),
       Builder.getInt8Ty(), Builder.getInt32Ty()});
  StructType *EmitArcsCallArgsTy = StructType::create(
      {Builder.getInt32Ty(), Builder.getInt64Ty()->getPointerTo()});
  StructType *FileInfoTy =
      StructType::create({StartFileCallArgsTy, Builder.getInt32Ty(),
                          EmitFunctionCallArgsTy->getPointerTo(),
                          EmitArcsCallArgsTy->getPointerTo()});

  Constant *Zero32 = Builder.getInt32(0);
  // Build an explicit array of two zeros for use in ConstantExpr GEP building.
  Constant *TwoZero32s[] = {Zero32, Zero32};

  SmallVector<Constant *, 8> FileInfos;
  for (int i : llvm::seq<int>(0, CUNodes->getNumOperands())) {
    auto *CU = cast<DICompileUnit>(CUNodes->getOperand(i));

    // Skip module skeleton (and module) CUs.
    if (CU->getDWOId())
      continue;

    std::string FilenameGcda = mangleName(CU, GCovFileType::GCDA);
    uint32_t CfgChecksum = FileChecksums.empty() ? 0 : FileChecksums[i];
    auto *StartFileCallArgs = ConstantStruct::get(
        StartFileCallArgsTy, {Builder.CreateGlobalStringPtr(FilenameGcda),
                              Builder.CreateGlobalStringPtr(ReversedVersion),
                              Builder.getInt32(CfgChecksum)});

    SmallVector<Constant *, 8> EmitFunctionCallArgsArray;
    SmallVector<Constant *, 8> EmitArcsCallArgsArray;
    for (int j : llvm::seq<int>(0, CountersBySP.size())) {
      auto *SP = cast_or_null<DISubprogram>(CountersBySP[j].second);
      uint32_t FuncChecksum = Funcs.empty() ? 0 : Funcs[j]->getFuncChecksum();
      EmitFunctionCallArgsArray.push_back(ConstantStruct::get(
          EmitFunctionCallArgsTy,
          {Builder.getInt32(j),
           Options.FunctionNamesInData
               ? Builder.CreateGlobalStringPtr(getFunctionName(SP))
               : Constant::getNullValue(Builder.getInt8PtrTy()),
           Builder.getInt32(FuncChecksum),
           Builder.getInt8(Options.UseCfgChecksum),
           Builder.getInt32(CfgChecksum)}));

      GlobalVariable *GV = CountersBySP[j].first;
      unsigned Arcs = cast<ArrayType>(GV->getValueType())->getNumElements();
      EmitArcsCallArgsArray.push_back(ConstantStruct::get(
          EmitArcsCallArgsTy,
          {Builder.getInt32(Arcs), ConstantExpr::getInBoundsGetElementPtr(
                                       GV->getValueType(), GV, TwoZero32s)}));
    }
    // Create global arrays for the two emit calls.
    int CountersSize = CountersBySP.size();
    assert(CountersSize == (int)EmitFunctionCallArgsArray.size() &&
           "Mismatched array size!");
    assert(CountersSize == (int)EmitArcsCallArgsArray.size() &&
           "Mismatched array size!");
    auto *EmitFunctionCallArgsArrayTy =
        ArrayType::get(EmitFunctionCallArgsTy, CountersSize);
    auto *EmitFunctionCallArgsArrayGV = new GlobalVariable(
        *M, EmitFunctionCallArgsArrayTy, /*isConstant*/ true,
        GlobalValue::InternalLinkage,
        ConstantArray::get(EmitFunctionCallArgsArrayTy,
                           EmitFunctionCallArgsArray),
        Twine("__llvm_internal_gcov_emit_function_args.") + Twine(i));
    auto *EmitArcsCallArgsArrayTy =
        ArrayType::get(EmitArcsCallArgsTy, CountersSize);
    EmitFunctionCallArgsArrayGV->setUnnamedAddr(
        GlobalValue::UnnamedAddr::Global);
    auto *EmitArcsCallArgsArrayGV = new GlobalVariable(
        *M, EmitArcsCallArgsArrayTy, /*isConstant*/ true,
        GlobalValue::InternalLinkage,
        ConstantArray::get(EmitArcsCallArgsArrayTy, EmitArcsCallArgsArray),
        Twine("__llvm_internal_gcov_emit_arcs_args.") + Twine(i));
    EmitArcsCallArgsArrayGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    FileInfos.push_back(ConstantStruct::get(
        FileInfoTy,
        {StartFileCallArgs, Builder.getInt32(CountersSize),
         ConstantExpr::getInBoundsGetElementPtr(EmitFunctionCallArgsArrayTy,
                                                EmitFunctionCallArgsArrayGV,
                                                TwoZero32s),
         ConstantExpr::getInBoundsGetElementPtr(
             EmitArcsCallArgsArrayTy, EmitArcsCallArgsArrayGV, TwoZero32s)}));
  }

  // If we didn't find anything to actually emit, bail on out.
  if (FileInfos.empty()) {
    Builder.CreateRetVoid();
    return WriteoutF;
  }

  // To simplify code, we cap the number of file infos we write out to fit
  // easily in a 32-bit signed integer. This gives consistent behavior between
  // 32-bit and 64-bit systems without requiring (potentially very slow) 64-bit
  // operations on 32-bit systems. It also seems unreasonable to try to handle
  // more than 2 billion files.
  if ((int64_t)FileInfos.size() > (int64_t)INT_MAX)
    FileInfos.resize(INT_MAX);

  // Create a global for the entire data structure so we can walk it more
  // easily.
  auto *FileInfoArrayTy = ArrayType::get(FileInfoTy, FileInfos.size());
  auto *FileInfoArrayGV = new GlobalVariable(
      *M, FileInfoArrayTy, /*isConstant*/ true, GlobalValue::InternalLinkage,
      ConstantArray::get(FileInfoArrayTy, FileInfos),
      "__llvm_internal_gcov_emit_file_info");
  FileInfoArrayGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  // Create the CFG for walking this data structure.
  auto *FileLoopHeader =
      BasicBlock::Create(*Ctx, "file.loop.header", WriteoutF);
  auto *CounterLoopHeader =
      BasicBlock::Create(*Ctx, "counter.loop.header", WriteoutF);
  auto *FileLoopLatch = BasicBlock::Create(*Ctx, "file.loop.latch", WriteoutF);
  auto *ExitBB = BasicBlock::Create(*Ctx, "exit", WriteoutF);

  // We always have at least one file, so just branch to the header.
  Builder.CreateBr(FileLoopHeader);

  // The index into the files structure is our loop induction variable.
  Builder.SetInsertPoint(FileLoopHeader);
  PHINode *IV =
      Builder.CreatePHI(Builder.getInt32Ty(), /*NumReservedValues*/ 2);
  IV->addIncoming(Builder.getInt32(0), BB);
  auto *FileInfoPtr =
      Builder.CreateInBoundsGEP(FileInfoArrayGV, {Builder.getInt32(0), IV});
  auto *StartFileCallArgsPtr = Builder.CreateStructGEP(FileInfoPtr, 0);
  Builder.CreateCall(
      StartFile,
      {Builder.CreateLoad(Builder.CreateStructGEP(StartFileCallArgsPtr, 0)),
       Builder.CreateLoad(Builder.CreateStructGEP(StartFileCallArgsPtr, 1)),
       Builder.CreateLoad(Builder.CreateStructGEP(StartFileCallArgsPtr, 2))});
  auto *NumCounters =
      Builder.CreateLoad(Builder.CreateStructGEP(FileInfoPtr, 1));
  auto *EmitFunctionCallArgsArray =
      Builder.CreateLoad(Builder.CreateStructGEP(FileInfoPtr, 2));
  auto *EmitArcsCallArgsArray =
      Builder.CreateLoad(Builder.CreateStructGEP(FileInfoPtr, 3));
  auto *EnterCounterLoopCond =
      Builder.CreateICmpSLT(Builder.getInt32(0), NumCounters);
  Builder.CreateCondBr(EnterCounterLoopCond, CounterLoopHeader, FileLoopLatch);

  Builder.SetInsertPoint(CounterLoopHeader);
  auto *JV = Builder.CreatePHI(Builder.getInt32Ty(), /*NumReservedValues*/ 2);
  JV->addIncoming(Builder.getInt32(0), FileLoopHeader);
  auto *EmitFunctionCallArgsPtr =
      Builder.CreateInBoundsGEP(EmitFunctionCallArgsArray, {JV});
  Builder.CreateCall(
      EmitFunction,
      {Builder.CreateLoad(Builder.CreateStructGEP(EmitFunctionCallArgsPtr, 0)),
       Builder.CreateLoad(Builder.CreateStructGEP(EmitFunctionCallArgsPtr, 1)),
       Builder.CreateLoad(Builder.CreateStructGEP(EmitFunctionCallArgsPtr, 2)),
       Builder.CreateLoad(Builder.CreateStructGEP(EmitFunctionCallArgsPtr, 3)),
       Builder.CreateLoad(
           Builder.CreateStructGEP(EmitFunctionCallArgsPtr, 4))});
  auto *EmitArcsCallArgsPtr =
      Builder.CreateInBoundsGEP(EmitArcsCallArgsArray, {JV});
  Builder.CreateCall(
      EmitArcs,
      {Builder.CreateLoad(Builder.CreateStructGEP(EmitArcsCallArgsPtr, 0)),
       Builder.CreateLoad(Builder.CreateStructGEP(EmitArcsCallArgsPtr, 1))});
  auto *NextJV = Builder.CreateAdd(JV, Builder.getInt32(1));
  auto *CounterLoopCond = Builder.CreateICmpSLT(NextJV, NumCounters);
  Builder.CreateCondBr(CounterLoopCond, CounterLoopHeader, FileLoopLatch);
  JV->addIncoming(NextJV, CounterLoopHeader);

  Builder.SetInsertPoint(FileLoopLatch);
  Builder.CreateCall(SummaryInfo, {});
  Builder.CreateCall(EndFile, {});
  auto *NextIV = Builder.CreateAdd(IV, Builder.getInt32(1));
  auto *FileLoopCond =
      Builder.CreateICmpSLT(NextIV, Builder.getInt32(FileInfos.size()));
  Builder.CreateCondBr(FileLoopCond, FileLoopHeader, ExitBB);
  IV->addIncoming(NextIV, FileLoopLatch);

  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  return WriteoutF;
}

void GCOVProfiler::insertIndirectCounterIncrement() {
  Function *Fn =
    cast<Function>(GCOVProfiler::getIncrementIndirectCounterFunc());
  Fn->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Fn->setLinkage(GlobalValue::InternalLinkage);
  Fn->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    Fn->addFnAttr(Attribute::NoRedZone);

  // Create basic blocks for function.
  BasicBlock *BB = BasicBlock::Create(*Ctx, "entry", Fn);
  IRBuilder<> Builder(BB);

  BasicBlock *PredNotNegOne = BasicBlock::Create(*Ctx, "", Fn);
  BasicBlock *CounterEnd = BasicBlock::Create(*Ctx, "", Fn);
  BasicBlock *Exit = BasicBlock::Create(*Ctx, "exit", Fn);

  // uint32_t pred = *predecessor;
  // if (pred == 0xffffffff) return;
  Argument *Arg = &*Fn->arg_begin();
  Arg->setName("predecessor");
  Value *Pred = Builder.CreateLoad(Arg, "pred");
  Value *Cond = Builder.CreateICmpEQ(Pred, Builder.getInt32(0xffffffff));
  BranchInst::Create(Exit, PredNotNegOne, Cond, BB);

  Builder.SetInsertPoint(PredNotNegOne);

  // uint64_t *counter = counters[pred];
  // if (!counter) return;
  Value *ZExtPred = Builder.CreateZExt(Pred, Builder.getInt64Ty());
  Arg = &*std::next(Fn->arg_begin());
  Arg->setName("counters");
  Value *GEP = Builder.CreateGEP(Type::getInt64PtrTy(*Ctx), Arg, ZExtPred);
  Value *Counter = Builder.CreateLoad(GEP, "counter");
  Cond = Builder.CreateICmpEQ(Counter,
                              Constant::getNullValue(
                                  Builder.getInt64Ty()->getPointerTo()));
  Builder.CreateCondBr(Cond, Exit, CounterEnd);

  // ++*counter;
  Builder.SetInsertPoint(CounterEnd);
  Value *Add = Builder.CreateAdd(Builder.CreateLoad(Counter),
                                 Builder.getInt64(1));
  Builder.CreateStore(Add, Counter);
  Builder.CreateBr(Exit);

  // Fill in the exit block.
  Builder.SetInsertPoint(Exit);
  Builder.CreateRetVoid();
}

Function *GCOVProfiler::
insertFlush(ArrayRef<std::pair<GlobalVariable*, MDNode*> > CountersBySP) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
  Function *FlushF = M->getFunction("__llvm_gcov_flush");
  if (!FlushF)
    FlushF = Function::Create(FTy, GlobalValue::InternalLinkage,
                              "__llvm_gcov_flush", M);
  else
    FlushF->setLinkage(GlobalValue::InternalLinkage);
  FlushF->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  FlushF->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    FlushF->addFnAttr(Attribute::NoRedZone);

  BasicBlock *Entry = BasicBlock::Create(*Ctx, "entry", FlushF);

  // Write out the current counters.
  Constant *WriteoutF = M->getFunction("__llvm_gcov_writeout");
  assert(WriteoutF && "Need to create the writeout function first!");

  IRBuilder<> Builder(Entry);
  Builder.CreateCall(WriteoutF, {});

  // Zero out the counters.
  for (const auto &I : CountersBySP) {
    GlobalVariable *GV = I.first;
    Constant *Null = Constant::getNullValue(GV->getValueType());
    Builder.CreateStore(Null, GV);
  }

  Type *RetTy = FlushF->getReturnType();
  if (RetTy == Type::getVoidTy(*Ctx))
    Builder.CreateRetVoid();
  else if (RetTy->isIntegerTy())
    // Used if __llvm_gcov_flush was implicitly declared.
    Builder.CreateRet(ConstantInt::get(RetTy, 0));
  else
    report_fatal_error("invalid return type for __llvm_gcov_flush");

  return FlushF;
}
