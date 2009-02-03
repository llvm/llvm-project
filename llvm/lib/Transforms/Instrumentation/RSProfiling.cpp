//===- RSProfiling.cpp - Various profiling using random sampling ----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These passes implement a random sampling based profiling.  Different methods
// of choosing when to sample are supported, as well as different types of
// profiling.  This is done as two passes.  The first is a sequence of profiling
// passes which insert profiling into the program, and remember what they 
// inserted.
//
// The second stage duplicates all instructions in a function, ignoring the 
// profiling code, then connects the two versions togeather at the entry and at
// backedges.  At each connection point a choice is made as to whether to jump
// to the profiled code (take a sample) or execute the unprofiled code.
//
// It is highly recommended that after this pass one runs mem2reg and adce
// (instcombine load-vn gdce dse also are good to run afterwards)
//
// This design is intended to make the profiling passes independent of the RS
// framework, but any profiling pass that implements the RSProfiling interface
// is compatible with the rs framework (and thus can be sampled)
//
// TODO: obviously the block and function profiling are almost identical to the
// existing ones, so they can be unified (esp since these passes are valid
// without the rs framework).
// TODO: Fix choice code so that frequency is not hard coded
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Instrumentation.h"
#include "RSProfiling.h"
#include <set>
#include <map>
#include <queue>
using namespace llvm;

namespace {
  enum RandomMeth {
    GBV, GBVO, HOSTCC
  };
}

static cl::opt<RandomMeth> RandomMethod("profile-randomness",
    cl::desc("How to randomly choose to profile:"),
    cl::values(
               clEnumValN(GBV, "global", "global counter"),
               clEnumValN(GBVO, "ra_global", 
                          "register allocated global counter"),
               clEnumValN(HOSTCC, "rdcc", "cycle counter"),
               clEnumValEnd));
  
namespace {
  /// NullProfilerRS - The basic profiler that does nothing.  It is the default
  /// profiler and thus terminates RSProfiler chains.  It is useful for 
  /// measuring framework overhead
  class VISIBILITY_HIDDEN NullProfilerRS : public RSProfilers {
  public:
    static char ID; // Pass identification, replacement for typeid
    bool isProfiling(Value* v) {
      return false;
    }
    bool runOnModule(Module &M) {
      return false;
    }
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

static RegisterAnalysisGroup<RSProfilers> A("Profiling passes");
static RegisterPass<NullProfilerRS> NP("insert-null-profiling-rs",
                                       "Measure profiling framework overhead");
static RegisterAnalysisGroup<RSProfilers, true> NPT(NP);

namespace {
  /// Chooser - Something that chooses when to make a sample of the profiled code
  class VISIBILITY_HIDDEN Chooser {
  public:
    /// ProcessChoicePoint - is called for each basic block inserted to choose 
    /// between normal and sample code
    virtual void ProcessChoicePoint(BasicBlock*) = 0;
    /// PrepFunction - is called once per function before other work is done.
    /// This gives the opertunity to insert new allocas and such.
    virtual void PrepFunction(Function*) = 0;
    virtual ~Chooser() {}
  };

  //Things that implement sampling policies
  //A global value that is read-mod-stored to choose when to sample.
  //A sample is taken when the global counter hits 0
  class VISIBILITY_HIDDEN GlobalRandomCounter : public Chooser {
    GlobalVariable* Counter;
    Value* ResetValue;
    const Type* T;
  public:
    GlobalRandomCounter(Module& M, const Type* t, uint64_t resetval);
    virtual ~GlobalRandomCounter();
    virtual void PrepFunction(Function* F);
    virtual void ProcessChoicePoint(BasicBlock* bb);
  };

  //Same is GRC, but allow register allocation of the global counter
  class VISIBILITY_HIDDEN GlobalRandomCounterOpt : public Chooser {
    GlobalVariable* Counter;
    Value* ResetValue;
    AllocaInst* AI;
    const Type* T;
  public:
    GlobalRandomCounterOpt(Module& M, const Type* t, uint64_t resetval);
    virtual ~GlobalRandomCounterOpt();
    virtual void PrepFunction(Function* F);
    virtual void ProcessChoicePoint(BasicBlock* bb);
  };

  //Use the cycle counter intrinsic as a source of pseudo randomness when
  //deciding when to sample.
  class VISIBILITY_HIDDEN CycleCounter : public Chooser {
    uint64_t rm;
    Constant *F;
  public:
    CycleCounter(Module& m, uint64_t resetmask);
    virtual ~CycleCounter();
    virtual void PrepFunction(Function* F);
    virtual void ProcessChoicePoint(BasicBlock* bb);
  };

  /// ProfilerRS - Insert the random sampling framework
  struct VISIBILITY_HIDDEN ProfilerRS : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    ProfilerRS() : FunctionPass(&ID) {}

    std::map<Value*, Value*> TransCache;
    std::set<BasicBlock*> ChoicePoints;
    Chooser* c;

    //Translate and duplicate values for the new profile free version of stuff
    Value* Translate(Value* v);
    //Duplicate an entire function (with out profiling)
    void Duplicate(Function& F, RSProfilers& LI);
    //Called once for each backedge, handle the insertion of choice points and
    //the interconection of the two versions of the code
    void ProcessBackEdge(BasicBlock* src, BasicBlock* dst, Function& F);
    bool runOnFunction(Function& F);
    bool doInitialization(Module &M);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  };
}

static RegisterPass<ProfilerRS>
X("insert-rs-profiling-framework",
  "Insert random sampling instrumentation framework");

char RSProfilers::ID = 0;
char NullProfilerRS::ID = 0;
char ProfilerRS::ID = 0;

//Local utilities
static void ReplacePhiPred(BasicBlock* btarget, 
                           BasicBlock* bold, BasicBlock* bnew);

static void CollapsePhi(BasicBlock* btarget, BasicBlock* bsrc);

template<class T>
static void recBackEdge(BasicBlock* bb, T& BackEdges, 
                        std::map<BasicBlock*, int>& color,
                        std::map<BasicBlock*, int>& depth,
                        std::map<BasicBlock*, int>& finish,
                        int& time);

//find the back edges and where they go to
template<class T>
static void getBackEdges(Function& F, T& BackEdges);


///////////////////////////////////////
// Methods of choosing when to profile
///////////////////////////////////////
  
GlobalRandomCounter::GlobalRandomCounter(Module& M, const Type* t, 
                                         uint64_t resetval) : T(t) {
  ConstantInt* Init = ConstantInt::get(T, resetval); 
  ResetValue = Init;
  Counter = new GlobalVariable(T, false, GlobalValue::InternalLinkage,
                               Init, "RandomSteeringCounter", &M);
}

GlobalRandomCounter::~GlobalRandomCounter() {}

void GlobalRandomCounter::PrepFunction(Function* F) {}

void GlobalRandomCounter::ProcessChoicePoint(BasicBlock* bb) {
  BranchInst* t = cast<BranchInst>(bb->getTerminator());
  
  //decrement counter
  LoadInst* l = new LoadInst(Counter, "counter", t);
  
  ICmpInst* s = new ICmpInst(ICmpInst::ICMP_EQ, l, ConstantInt::get(T, 0), 
                             "countercc", t);

  Value* nv = BinaryOperator::CreateSub(l, ConstantInt::get(T, 1),
                                        "counternew", t);
  new StoreInst(nv, Counter, t);
  t->setCondition(s);
  
  //reset counter
  BasicBlock* oldnext = t->getSuccessor(0);
  BasicBlock* resetblock = BasicBlock::Create("reset", oldnext->getParent(), 
                                              oldnext);
  TerminatorInst* t2 = BranchInst::Create(oldnext, resetblock);
  t->setSuccessor(0, resetblock);
  new StoreInst(ResetValue, Counter, t2);
  ReplacePhiPred(oldnext, bb, resetblock);
}

GlobalRandomCounterOpt::GlobalRandomCounterOpt(Module& M, const Type* t, 
                                               uint64_t resetval) 
  : AI(0), T(t) {
  ConstantInt* Init = ConstantInt::get(T, resetval);
  ResetValue  = Init;
  Counter = new GlobalVariable(T, false, GlobalValue::InternalLinkage,
                               Init, "RandomSteeringCounter", &M);
}

GlobalRandomCounterOpt::~GlobalRandomCounterOpt() {}

void GlobalRandomCounterOpt::PrepFunction(Function* F) {
  //make a local temporary to cache the global
  BasicBlock& bb = F->getEntryBlock();
  BasicBlock::iterator InsertPt = bb.begin();
  AI = new AllocaInst(T, 0, "localcounter", InsertPt);
  LoadInst* l = new LoadInst(Counter, "counterload", InsertPt);
  new StoreInst(l, AI, InsertPt);
  
  //modify all functions and return values to restore the local variable to/from
  //the global variable
  for(Function::iterator fib = F->begin(), fie = F->end();
      fib != fie; ++fib)
    for(BasicBlock::iterator bib = fib->begin(), bie = fib->end();
        bib != bie; ++bib)
      if (isa<CallInst>(bib)) {
        LoadInst* l = new LoadInst(AI, "counter", bib);
        new StoreInst(l, Counter, bib);
        l = new LoadInst(Counter, "counter", ++bib);
        new StoreInst(l, AI, bib--);
      } else if (isa<InvokeInst>(bib)) {
        LoadInst* l = new LoadInst(AI, "counter", bib);
        new StoreInst(l, Counter, bib);
        
        BasicBlock* bb = cast<InvokeInst>(bib)->getNormalDest();
        BasicBlock::iterator i = bb->getFirstNonPHI();
        l = new LoadInst(Counter, "counter", i);
        
        bb = cast<InvokeInst>(bib)->getUnwindDest();
        i = bb->getFirstNonPHI();
        l = new LoadInst(Counter, "counter", i);
        new StoreInst(l, AI, i);
      } else if (isa<UnwindInst>(&*bib) || isa<ReturnInst>(&*bib)) {
        LoadInst* l = new LoadInst(AI, "counter", bib);
        new StoreInst(l, Counter, bib);
      }
}

void GlobalRandomCounterOpt::ProcessChoicePoint(BasicBlock* bb) {
  BranchInst* t = cast<BranchInst>(bb->getTerminator());
  
  //decrement counter
  LoadInst* l = new LoadInst(AI, "counter", t);
  
  ICmpInst* s = new ICmpInst(ICmpInst::ICMP_EQ, l, ConstantInt::get(T, 0), 
                             "countercc", t);

  Value* nv = BinaryOperator::CreateSub(l, ConstantInt::get(T, 1),
                                        "counternew", t);
  new StoreInst(nv, AI, t);
  t->setCondition(s);
  
  //reset counter
  BasicBlock* oldnext = t->getSuccessor(0);
  BasicBlock* resetblock = BasicBlock::Create("reset", oldnext->getParent(), 
                                              oldnext);
  TerminatorInst* t2 = BranchInst::Create(oldnext, resetblock);
  t->setSuccessor(0, resetblock);
  new StoreInst(ResetValue, AI, t2);
  ReplacePhiPred(oldnext, bb, resetblock);
}


CycleCounter::CycleCounter(Module& m, uint64_t resetmask) : rm(resetmask) {
  F = Intrinsic::getDeclaration(&m, Intrinsic::readcyclecounter);
}

CycleCounter::~CycleCounter() {}

void CycleCounter::PrepFunction(Function* F) {}

void CycleCounter::ProcessChoicePoint(BasicBlock* bb) {
  BranchInst* t = cast<BranchInst>(bb->getTerminator());
  
  CallInst* c = CallInst::Create(F, "rdcc", t);
  BinaryOperator* b = 
    BinaryOperator::CreateAnd(c, ConstantInt::get(Type::Int64Ty, rm),
                              "mrdcc", t);
  
  ICmpInst *s = new ICmpInst(ICmpInst::ICMP_EQ, b,
                             ConstantInt::get(Type::Int64Ty, 0), 
                             "mrdccc", t);

  t->setCondition(s);
}

///////////////////////////////////////
// Profiling:
///////////////////////////////////////
bool RSProfilers_std::isProfiling(Value* v) {
  if (profcode.find(v) != profcode.end())
    return true;
  //else
  RSProfilers& LI = getAnalysis<RSProfilers>();
  return LI.isProfiling(v);
}

void RSProfilers_std::IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                                          GlobalValue *CounterArray) {
  // Insert the increment after any alloca or PHI instructions...
  BasicBlock::iterator InsertPos = BB->getFirstNonPHI();
  while (isa<AllocaInst>(InsertPos))
    ++InsertPos;
  
  // Create the getelementptr constant expression
  std::vector<Constant*> Indices(2);
  Indices[0] = Constant::getNullValue(Type::Int32Ty);
  Indices[1] = ConstantInt::get(Type::Int32Ty, CounterNum);
  Constant *ElementPtr = ConstantExpr::getGetElementPtr(CounterArray,
                                                        &Indices[0], 2);
  
  // Load, increment and store the value back.
  Value *OldVal = new LoadInst(ElementPtr, "OldCounter", InsertPos);
  profcode.insert(OldVal);
  Value *NewVal = BinaryOperator::CreateAdd(OldVal,
                                            ConstantInt::get(Type::Int32Ty, 1),
                                            "NewCounter", InsertPos);
  profcode.insert(NewVal);
  profcode.insert(new StoreInst(NewVal, ElementPtr, InsertPos));
}

void RSProfilers_std::getAnalysisUsage(AnalysisUsage &AU) const {
  //grab any outstanding profiler, or get the null one
  AU.addRequired<RSProfilers>();
}

///////////////////////////////////////
// RS Framework
///////////////////////////////////////

Value* ProfilerRS::Translate(Value* v) {
  if(TransCache[v])
    return TransCache[v];
  
  if (BasicBlock* bb = dyn_cast<BasicBlock>(v)) {
    if (bb == &bb->getParent()->getEntryBlock())
      TransCache[bb] = bb; //don't translate entry block
    else
      TransCache[bb] = BasicBlock::Create("dup_" + bb->getName(),
                                          bb->getParent(), NULL);
    return TransCache[bb];
  } else if (Instruction* i = dyn_cast<Instruction>(v)) {
    //we have already translated this
    //do not translate entry block allocas
    if(&i->getParent()->getParent()->getEntryBlock() == i->getParent()) {
      TransCache[i] = i;
      return i;
    } else {
      //translate this
      Instruction* i2 = i->clone();
      if (i->hasName())
        i2->setName("dup_" + i->getName());
      TransCache[i] = i2;
      //NumNewInst++;
      for (unsigned x = 0; x < i2->getNumOperands(); ++x)
        i2->setOperand(x, Translate(i2->getOperand(x)));
      return i2;
    }
  } else if (isa<Function>(v) || isa<Constant>(v) || isa<Argument>(v)) {
    TransCache[v] = v;
    return v;
  }
  assert(0 && "Value not handled");
  return 0;
}

void ProfilerRS::Duplicate(Function& F, RSProfilers& LI)
{
  //perform a breadth first search, building up a duplicate of the code
  std::queue<BasicBlock*> worklist;
  std::set<BasicBlock*> seen;
  
  //This loop ensures proper BB order, to help performance
  for (Function::iterator fib = F.begin(), fie = F.end(); fib != fie; ++fib)
    worklist.push(fib);
  while (!worklist.empty()) {
    Translate(worklist.front());
    worklist.pop();
  }
  
  //remember than reg2mem created a new entry block we don't want to duplicate
  worklist.push(F.getEntryBlock().getTerminator()->getSuccessor(0));
  seen.insert(&F.getEntryBlock());
  
  while (!worklist.empty()) {
    BasicBlock* bb = worklist.front();
    worklist.pop();
    if(seen.find(bb) == seen.end()) {
      BasicBlock* bbtarget = cast<BasicBlock>(Translate(bb));
      BasicBlock::InstListType& instlist = bbtarget->getInstList();
      for (BasicBlock::iterator iib = bb->begin(), iie = bb->end(); 
           iib != iie; ++iib) {
        //NumOldInst++;
        if (!LI.isProfiling(&*iib)) {
          Instruction* i = cast<Instruction>(Translate(iib));
          instlist.insert(bbtarget->end(), i);
        }
      }
      //updated search state;
      seen.insert(bb);
      TerminatorInst* ti = bb->getTerminator();
      for (unsigned x = 0; x < ti->getNumSuccessors(); ++x) {
        BasicBlock* bbs = ti->getSuccessor(x);
        if (seen.find(bbs) == seen.end()) {
          worklist.push(bbs);
        }
      }
    }
  }
}

void ProfilerRS::ProcessBackEdge(BasicBlock* src, BasicBlock* dst, Function& F) {
  //given a backedge from B -> A, and translations A' and B',
  //a: insert C and C'
  //b: add branches in C to A and A' and in C' to A and A'
  //c: mod terminators@B, replace A with C
  //d: mod terminators@B', replace A' with C'
  //e: mod phis@A for pred B to be pred C
  //       if multiple entries, simplify to one
  //f: mod phis@A' for pred B' to be pred C'
  //       if multiple entries, simplify to one
  //g: for all phis@A with pred C using x
  //       add in edge from C' using x'
  //       add in edge from C using x in A'
  
  //a:
  Function::iterator BBN = src; ++BBN;
  BasicBlock* bbC = BasicBlock::Create("choice", &F, BBN);
  //ChoicePoints.insert(bbC);
  BBN = cast<BasicBlock>(Translate(src));
  BasicBlock* bbCp = BasicBlock::Create("choice", &F, ++BBN);
  ChoicePoints.insert(bbCp);
  
  //b:
  BranchInst::Create(cast<BasicBlock>(Translate(dst)), bbC);
  BranchInst::Create(dst, cast<BasicBlock>(Translate(dst)), 
                     ConstantInt::get(Type::Int1Ty, true), bbCp);
  //c:
  {
    TerminatorInst* iB = src->getTerminator();
    for (unsigned x = 0; x < iB->getNumSuccessors(); ++x)
      if (iB->getSuccessor(x) == dst)
        iB->setSuccessor(x, bbC);
  }
  //d:
  {
    TerminatorInst* iBp = cast<TerminatorInst>(Translate(src->getTerminator()));
    for (unsigned x = 0; x < iBp->getNumSuccessors(); ++x)
      if (iBp->getSuccessor(x) == cast<BasicBlock>(Translate(dst)))
        iBp->setSuccessor(x, bbCp);
  }
  //e:
  ReplacePhiPred(dst, src, bbC);
  //src could be a switch, in which case we are replacing several edges with one
  //thus collapse those edges int the Phi
  CollapsePhi(dst, bbC);
  //f:
  ReplacePhiPred(cast<BasicBlock>(Translate(dst)),
                 cast<BasicBlock>(Translate(src)),bbCp);
  CollapsePhi(cast<BasicBlock>(Translate(dst)), bbCp);
  //g:
  for(BasicBlock::iterator ib = dst->begin(), ie = dst->end(); ib != ie;
      ++ib)
    if (PHINode* phi = dyn_cast<PHINode>(&*ib)) {
      for(unsigned x = 0; x < phi->getNumIncomingValues(); ++x)
        if(bbC == phi->getIncomingBlock(x)) {
          phi->addIncoming(Translate(phi->getIncomingValue(x)), bbCp);
          cast<PHINode>(Translate(phi))->addIncoming(phi->getIncomingValue(x), 
                                                     bbC);
        }
      phi->removeIncomingValue(bbC);
    }
}

bool ProfilerRS::runOnFunction(Function& F) {
  if (!F.isDeclaration()) {
    std::set<std::pair<BasicBlock*, BasicBlock*> > BackEdges;
    RSProfilers& LI = getAnalysis<RSProfilers>();
    
    getBackEdges(F, BackEdges);
    Duplicate(F, LI);
    //assume that stuff worked.  now connect the duplicated basic blocks 
    //with the originals in such a way as to preserve ssa.  yuk!
    for (std::set<std::pair<BasicBlock*, BasicBlock*> >::iterator 
           ib = BackEdges.begin(), ie = BackEdges.end(); ib != ie; ++ib)
      ProcessBackEdge(ib->first, ib->second, F);
    
    //oh, and add the edge from the reg2mem created entry node to the 
    //duplicated second node
    TerminatorInst* T = F.getEntryBlock().getTerminator();
    ReplaceInstWithInst(T, BranchInst::Create(T->getSuccessor(0),
                                              cast<BasicBlock>(
                                                Translate(T->getSuccessor(0))),
                                              ConstantInt::get(Type::Int1Ty,
                                                               true)));
    
    //do whatever is needed now that the function is duplicated
    c->PrepFunction(&F);
    
    //add entry node to choice points
    ChoicePoints.insert(&F.getEntryBlock());
    
    for (std::set<BasicBlock*>::iterator 
           ii = ChoicePoints.begin(), ie = ChoicePoints.end(); ii != ie; ++ii)
      c->ProcessChoicePoint(*ii);
    
    ChoicePoints.clear();
    TransCache.clear();
    
    return true;
  }
  return false;
}

bool ProfilerRS::doInitialization(Module &M) {
  switch (RandomMethod) {
  case GBV:
    c = new GlobalRandomCounter(M, Type::Int32Ty, (1 << 14) - 1);
    break;
  case GBVO:
    c = new GlobalRandomCounterOpt(M, Type::Int32Ty, (1 << 14) - 1);
    break;
  case HOSTCC:
    c = new CycleCounter(M, (1 << 14) - 1);
    break;
  };
  return true;
}

void ProfilerRS::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<RSProfilers>();
  AU.addRequiredID(DemoteRegisterToMemoryID);
}

///////////////////////////////////////
// Utilities:
///////////////////////////////////////
static void ReplacePhiPred(BasicBlock* btarget, 
                           BasicBlock* bold, BasicBlock* bnew) {
  for(BasicBlock::iterator ib = btarget->begin(), ie = btarget->end();
      ib != ie; ++ib)
    if (PHINode* phi = dyn_cast<PHINode>(&*ib)) {
      for(unsigned x = 0; x < phi->getNumIncomingValues(); ++x)
        if(bold == phi->getIncomingBlock(x))
          phi->setIncomingBlock(x, bnew);
    }
}

static void CollapsePhi(BasicBlock* btarget, BasicBlock* bsrc) {
  for(BasicBlock::iterator ib = btarget->begin(), ie = btarget->end();
      ib != ie; ++ib)
    if (PHINode* phi = dyn_cast<PHINode>(&*ib)) {
      std::map<BasicBlock*, Value*> counter;
      for(unsigned i = 0; i < phi->getNumIncomingValues(); ) {
        if (counter[phi->getIncomingBlock(i)]) {
          assert(phi->getIncomingValue(i) == counter[phi->getIncomingBlock(i)]);
          phi->removeIncomingValue(i, false);
        } else {
          counter[phi->getIncomingBlock(i)] = phi->getIncomingValue(i);
          ++i;
        }
      }
    } 
}

template<class T>
static void recBackEdge(BasicBlock* bb, T& BackEdges, 
                        std::map<BasicBlock*, int>& color,
                        std::map<BasicBlock*, int>& depth,
                        std::map<BasicBlock*, int>& finish,
                        int& time)
{
  color[bb] = 1;
  ++time;
  depth[bb] = time;
  TerminatorInst* t= bb->getTerminator();
  for(unsigned i = 0; i < t->getNumSuccessors(); ++i) {
    BasicBlock* bbnew = t->getSuccessor(i);
    if (color[bbnew] == 0)
      recBackEdge(bbnew, BackEdges, color, depth, finish, time);
    else if (color[bbnew] == 1) {
      BackEdges.insert(std::make_pair(bb, bbnew));
      //NumBackEdges++;
    }
  }
  color[bb] = 2;
  ++time;
  finish[bb] = time;
}



//find the back edges and where they go to
template<class T>
static void getBackEdges(Function& F, T& BackEdges) {
  std::map<BasicBlock*, int> color;
  std::map<BasicBlock*, int> depth;
  std::map<BasicBlock*, int> finish;
  int time = 0;
  recBackEdge(&F.getEntryBlock(), BackEdges, color, depth, finish, time);
  DOUT << F.getName() << " " << BackEdges.size() << "\n";
}


//Creation functions
ModulePass* llvm::createNullProfilerRSPass() {
  return new NullProfilerRS();
}

FunctionPass* llvm::createRSProfilingPass() {
  return new ProfilerRS();
}
