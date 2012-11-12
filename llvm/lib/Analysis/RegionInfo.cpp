//===- RegionInfo.cpp - SESE region detection analysis --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Detects single entry single exit regions in the control flow graph.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Assembly/Writer.h"

#define DEBUG_TYPE "region"
#include "llvm/Support/Debug.h"

#include <set>
#include <algorithm>

using namespace llvm;

// Always verify if expensive checking is enabled.
#ifdef XDEBUG
static bool VerifyRegionInfo = true;
#else
static bool VerifyRegionInfo = false;
#endif

static cl::opt<bool,true>
VerifyRegionInfoX("verify-region-info", cl::location(VerifyRegionInfo),
                cl::desc("Verify region info (time consuming)"));

STATISTIC(numRegions,       "The # of regions");
STATISTIC(numSimpleRegions, "The # of simple regions");

static cl::opt<enum Region::PrintStyle> printStyle("print-region-style",
  cl::Hidden,
  cl::desc("style of printing regions"),
  cl::values(
    clEnumValN(Region::PrintNone, "none",  "print no details"),
    clEnumValN(Region::PrintBB, "bb",
               "print regions in detail with block_iterator"),
    clEnumValN(Region::PrintRN, "rn",
               "print regions in detail with element_iterator"),
    clEnumValEnd));
//===----------------------------------------------------------------------===//
/// Region Implementation
Region::Region(BasicBlock *Entry, BasicBlock *Exit, RegionInfo* RInfo,
               DominatorTree *dt, Region *Parent)
               : RegionNode(Parent, Entry, 1), RI(RInfo), DT(dt), exit(Exit) {}

Region::~Region() {
  // Free the cached nodes.
  for (BBNodeMapT::iterator it = BBNodeMap.begin(),
         ie = BBNodeMap.end(); it != ie; ++it)
    delete it->second;

  // Only clean the cache for this Region. Caches of child Regions will be
  // cleaned when the child Regions are deleted.
  BBNodeMap.clear();

  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
}

void Region::replaceEntry(BasicBlock *BB) {
  entry.setPointer(BB);
}

void Region::replaceExit(BasicBlock *BB) {
  assert(exit && "No exit to replace!");
  exit = BB;
}

bool Region::contains(const BasicBlock *B) const {
  BasicBlock *BB = const_cast<BasicBlock*>(B);

  assert(DT->getNode(BB) && "BB not part of the dominance tree");

  BasicBlock *entry = getEntry(), *exit = getExit();

  // Toplevel region.
  if (!exit)
    return true;

  return (DT->dominates(entry, BB)
    && !(DT->dominates(exit, BB) && DT->dominates(entry, exit)));
}

bool Region::contains(const Loop *L) const {
  // BBs that are not part of any loop are element of the Loop
  // described by the NULL pointer. This loop is not part of any region,
  // except if the region describes the whole function.
  if (L == 0)
    return getExit() == 0;

  if (!contains(L->getHeader()))
    return false;

  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  for (SmallVectorImpl<BasicBlock*>::iterator BI = ExitingBlocks.begin(),
       BE = ExitingBlocks.end(); BI != BE; ++BI)
    if (!contains(*BI))
      return false;

  return true;
}

Loop *Region::outermostLoopInRegion(Loop *L) const {
  if (!contains(L))
    return 0;

  while (L && contains(L->getParentLoop())) {
    L = L->getParentLoop();
  }

  return L;
}

Loop *Region::outermostLoopInRegion(LoopInfo *LI, BasicBlock* BB) const {
  assert(LI && BB && "LI and BB cannot be null!");
  Loop *L = LI->getLoopFor(BB);
  return outermostLoopInRegion(L);
}

BasicBlock *Region::getEnteringBlock() const {
  BasicBlock *entry = getEntry();
  BasicBlock *Pred;
  BasicBlock *enteringBlock = 0;

  for (pred_iterator PI = pred_begin(entry), PE = pred_end(entry); PI != PE;
       ++PI) {
    Pred = *PI;
    if (DT->getNode(Pred) && !contains(Pred)) {
      if (enteringBlock)
        return 0;

      enteringBlock = Pred;
    }
  }

  return enteringBlock;
}

BasicBlock *Region::getExitingBlock() const {
  BasicBlock *exit = getExit();
  BasicBlock *Pred;
  BasicBlock *exitingBlock = 0;

  if (!exit)
    return 0;

  for (pred_iterator PI = pred_begin(exit), PE = pred_end(exit); PI != PE;
       ++PI) {
    Pred = *PI;
    if (contains(Pred)) {
      if (exitingBlock)
        return 0;

      exitingBlock = Pred;
    }
  }

  return exitingBlock;
}

bool Region::isSimple() const {
  return !isTopLevelRegion() && getEnteringBlock() && getExitingBlock();
}

std::string Region::getNameStr() const {
  std::string exitName;
  std::string entryName;

  if (getEntry()->getName().empty()) {
    raw_string_ostream OS(entryName);

    WriteAsOperand(OS, getEntry(), false);
  } else
    entryName = getEntry()->getName();

  if (getExit()) {
    if (getExit()->getName().empty()) {
      raw_string_ostream OS(exitName);

      WriteAsOperand(OS, getExit(), false);
    } else
      exitName = getExit()->getName();
  } else
    exitName = "<Function Return>";

  return entryName + " => " + exitName;
}

void Region::verifyBBInRegion(BasicBlock *BB) const {
  if (!contains(BB))
    llvm_unreachable("Broken region found!");

  BasicBlock *entry = getEntry(), *exit = getExit();

  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    if (!contains(*SI) && exit != *SI)
      llvm_unreachable("Broken region found!");

  if (entry != BB)
    for (pred_iterator SI = pred_begin(BB), SE = pred_end(BB); SI != SE; ++SI)
      if (!contains(*SI))
        llvm_unreachable("Broken region found!");
}

void Region::verifyWalk(BasicBlock *BB, std::set<BasicBlock*> *visited) const {
  BasicBlock *exit = getExit();

  visited->insert(BB);

  verifyBBInRegion(BB);

  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    if (*SI != exit && visited->find(*SI) == visited->end())
        verifyWalk(*SI, visited);
}

void Region::verifyRegion() const {
  // Only do verification when user wants to, otherwise this expensive
  // check will be invoked by PassManager.
  if (!VerifyRegionInfo) return;

  std::set<BasicBlock*> visited;
  verifyWalk(getEntry(), &visited);
}

void Region::verifyRegionNest() const {
  for (Region::const_iterator RI = begin(), RE = end(); RI != RE; ++RI)
    (*RI)->verifyRegionNest();

  verifyRegion();
}

Region::element_iterator Region::element_begin() {
  return GraphTraits<Region*>::nodes_begin(this);
}

Region::element_iterator Region::element_end() {
  return GraphTraits<Region*>::nodes_end(this);
}

Region::const_element_iterator Region::element_begin() const {
  return GraphTraits<const Region*>::nodes_begin(this);
}

Region::const_element_iterator Region::element_end() const {
  return GraphTraits<const Region*>::nodes_end(this);
}

Region* Region::getSubRegionNode(BasicBlock *BB) const {
  Region *R = RI->getRegionFor(BB);

  if (!R || R == this)
    return 0;

  // If we pass the BB out of this region, that means our code is broken.
  assert(contains(R) && "BB not in current region!");

  while (contains(R->getParent()) && R->getParent() != this)
    R = R->getParent();

  if (R->getEntry() != BB)
    return 0;

  return R;
}

RegionNode* Region::getBBNode(BasicBlock *BB) const {
  assert(contains(BB) && "Can get BB node out of this region!");

  BBNodeMapT::const_iterator at = BBNodeMap.find(BB);

  if (at != BBNodeMap.end())
    return at->second;

  RegionNode *NewNode = new RegionNode(const_cast<Region*>(this), BB);
  BBNodeMap.insert(std::make_pair(BB, NewNode));
  return NewNode;
}

RegionNode* Region::getNode(BasicBlock *BB) const {
  assert(contains(BB) && "Can get BB node out of this region!");
  if (Region* Child = getSubRegionNode(BB))
    return Child->getNode();

  return getBBNode(BB);
}

void Region::transferChildrenTo(Region *To) {
  for (iterator I = begin(), E = end(); I != E; ++I) {
    (*I)->parent = To;
    To->children.push_back(*I);
  }
  children.clear();
}

void Region::addSubRegion(Region *SubRegion, bool moveChildren) {
  assert(SubRegion->parent == 0 && "SubRegion already has a parent!");
  assert(std::find(begin(), end(), SubRegion) == children.end()
         && "Subregion already exists!");

  SubRegion->parent = this;
  children.push_back(SubRegion);

  if (!moveChildren)
    return;

  assert(SubRegion->children.size() == 0
         && "SubRegions that contain children are not supported");

  for (element_iterator I = element_begin(), E = element_end(); I != E; ++I)
    if (!(*I)->isSubRegion()) {
      BasicBlock *BB = (*I)->getNodeAs<BasicBlock>();

      if (SubRegion->contains(BB))
        RI->setRegionFor(BB, SubRegion);
    }

  std::vector<Region*> Keep;
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (SubRegion->contains(*I) && *I != SubRegion) {
      SubRegion->children.push_back(*I);
      (*I)->parent = SubRegion;
    } else
      Keep.push_back(*I);

  children.clear();
  children.insert(children.begin(), Keep.begin(), Keep.end());
}


Region *Region::removeSubRegion(Region *Child) {
  assert(Child->parent == this && "Child is not a child of this region!");
  Child->parent = 0;
  RegionSet::iterator I = std::find(children.begin(), children.end(), Child);
  assert(I != children.end() && "Region does not exit. Unable to remove.");
  children.erase(children.begin()+(I-begin()));
  return Child;
}

unsigned Region::getDepth() const {
  unsigned Depth = 0;

  for (Region *R = parent; R != 0; R = R->parent)
    ++Depth;

  return Depth;
}

Region *Region::getExpandedRegion() const {
  unsigned NumSuccessors = exit->getTerminator()->getNumSuccessors();

  if (NumSuccessors == 0)
    return NULL;

  for (pred_iterator PI = pred_begin(getExit()), PE = pred_end(getExit());
       PI != PE; ++PI)
    if (!DT->dominates(getEntry(), *PI))
      return NULL;

  Region *R = RI->getRegionFor(exit);

  if (R->getEntry() != exit) {
    if (exit->getTerminator()->getNumSuccessors() == 1)
      return new Region(getEntry(), *succ_begin(exit), RI, DT);
    else
      return NULL;
  }

  while (R->getParent() && R->getParent()->getEntry() == exit)
    R = R->getParent();

  if (!DT->dominates(getEntry(), R->getExit()))
    for (pred_iterator PI = pred_begin(getExit()), PE = pred_end(getExit());
         PI != PE; ++PI)
    if (!DT->dominates(R->getExit(), *PI))
      return NULL;

  return new Region(getEntry(), R->getExit(), RI, DT);
}

void Region::print(raw_ostream &OS, bool print_tree, unsigned level,
                   enum PrintStyle Style) const {
  if (print_tree)
    OS.indent(level*2) << "[" << level << "] " << getNameStr();
  else
    OS.indent(level*2) << getNameStr();

  OS << "\n";


  if (Style != PrintNone) {
    OS.indent(level*2) << "{\n";
    OS.indent(level*2 + 2);

    if (Style == PrintBB) {
      for (const_block_iterator I = block_begin(), E = block_end(); I != E; ++I)
        OS << (*I)->getName() << ", "; // TODO: remove the last ","
    } else if (Style == PrintRN) {
      for (const_element_iterator I = element_begin(), E = element_end(); I!=E; ++I)
        OS << **I << ", "; // TODO: remove the last ",
    }

    OS << "\n";
  }

  if (print_tree)
    for (const_iterator RI = begin(), RE = end(); RI != RE; ++RI)
      (*RI)->print(OS, print_tree, level+1, Style);

  if (Style != PrintNone)
    OS.indent(level*2) << "} \n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void Region::dump() const {
  print(dbgs(), true, getDepth(), printStyle.getValue());
}
#endif

void Region::clearNodeCache() {
  // Free the cached nodes.
  for (BBNodeMapT::iterator I = BBNodeMap.begin(),
       IE = BBNodeMap.end(); I != IE; ++I)
    delete I->second;

  BBNodeMap.clear();
  for (Region::iterator RI = begin(), RE = end(); RI != RE; ++RI)
    (*RI)->clearNodeCache();
}

//===----------------------------------------------------------------------===//
// RegionInfo implementation
//

bool RegionInfo::isCommonDomFrontier(BasicBlock *BB, BasicBlock *entry,
                                     BasicBlock *exit) const {
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
    BasicBlock *P = *PI;
    if (DT->dominates(entry, P) && !DT->dominates(exit, P))
      return false;
  }
  return true;
}

bool RegionInfo::isRegion(BasicBlock *entry, BasicBlock *exit) const {
  assert(entry && exit && "entry and exit must not be null!");
  typedef DominanceFrontier::DomSetType DST;

  DST *entrySuccs = &DF->find(entry)->second;

  // Exit is the header of a loop that contains the entry. In this case,
  // the dominance frontier must only contain the exit.
  if (!DT->dominates(entry, exit)) {
    for (DST::iterator SI = entrySuccs->begin(), SE = entrySuccs->end();
         SI != SE; ++SI)
      if (*SI != exit && *SI != entry)
        return false;

    return true;
  }

  DST *exitSuccs = &DF->find(exit)->second;

  // Do not allow edges leaving the region.
  for (DST::iterator SI = entrySuccs->begin(), SE = entrySuccs->end();
       SI != SE; ++SI) {
    if (*SI == exit || *SI == entry)
      continue;
    if (exitSuccs->find(*SI) == exitSuccs->end())
      return false;
    if (!isCommonDomFrontier(*SI, entry, exit))
      return false;
  }

  // Do not allow edges pointing into the region.
  for (DST::iterator SI = exitSuccs->begin(), SE = exitSuccs->end();
       SI != SE; ++SI)
    if (DT->properlyDominates(entry, *SI) && *SI != exit)
      return false;


  return true;
}

void RegionInfo::insertShortCut(BasicBlock *entry, BasicBlock *exit,
                             BBtoBBMap *ShortCut) const {
  assert(entry && exit && "entry and exit must not be null!");

  BBtoBBMap::iterator e = ShortCut->find(exit);

  if (e == ShortCut->end())
    // No further region at exit available.
    (*ShortCut)[entry] = exit;
  else {
    // We found a region e that starts at exit. Therefore (entry, e->second)
    // is also a region, that is larger than (entry, exit). Insert the
    // larger one.
    BasicBlock *BB = e->second;
    (*ShortCut)[entry] = BB;
  }
}

DomTreeNode* RegionInfo::getNextPostDom(DomTreeNode* N,
                                        BBtoBBMap *ShortCut) const {
  BBtoBBMap::iterator e = ShortCut->find(N->getBlock());

  if (e == ShortCut->end())
    return N->getIDom();

  return PDT->getNode(e->second)->getIDom();
}

bool RegionInfo::isTrivialRegion(BasicBlock *entry, BasicBlock *exit) const {
  assert(entry && exit && "entry and exit must not be null!");

  unsigned num_successors = succ_end(entry) - succ_begin(entry);

  if (num_successors <= 1 && exit == *(succ_begin(entry)))
    return true;

  return false;
}

void RegionInfo::updateStatistics(Region *R) {
  ++numRegions;

  // TODO: Slow. Should only be enabled if -stats is used.
  if (R->isSimple()) ++numSimpleRegions;
}

Region *RegionInfo::createRegion(BasicBlock *entry, BasicBlock *exit) {
  assert(entry && exit && "entry and exit must not be null!");

  if (isTrivialRegion(entry, exit))
    return 0;

  Region *region = new Region(entry, exit, this, DT);
  BBtoRegion.insert(std::make_pair(entry, region));

 #ifdef XDEBUG
    region->verifyRegion();
 #else
    DEBUG(region->verifyRegion());
 #endif

  updateStatistics(region);
  return region;
}

void RegionInfo::findRegionsWithEntry(BasicBlock *entry, BBtoBBMap *ShortCut) {
  assert(entry);

  DomTreeNode *N = PDT->getNode(entry);

  if (!N)
    return;

  Region *lastRegion= 0;
  BasicBlock *lastExit = entry;

  // As only a BasicBlock that postdominates entry can finish a region, walk the
  // post dominance tree upwards.
  while ((N = getNextPostDom(N, ShortCut))) {
    BasicBlock *exit = N->getBlock();

    if (!exit)
      break;

    if (isRegion(entry, exit)) {
      Region *newRegion = createRegion(entry, exit);

      if (lastRegion)
        newRegion->addSubRegion(lastRegion);

      lastRegion = newRegion;
      lastExit = exit;
    }

    // This can never be a region, so stop the search.
    if (!DT->dominates(entry, exit))
      break;
  }

  // Tried to create regions from entry to lastExit.  Next time take a
  // shortcut from entry to lastExit.
  if (lastExit != entry)
    insertShortCut(entry, lastExit, ShortCut);
}

void RegionInfo::scanForRegions(Function &F, BBtoBBMap *ShortCut) {
  BasicBlock *entry = &(F.getEntryBlock());
  DomTreeNode *N = DT->getNode(entry);

  // Iterate over the dominance tree in post order to start with the small
  // regions from the bottom of the dominance tree.  If the small regions are
  // detected first, detection of bigger regions is faster, as we can jump
  // over the small regions.
  for (po_iterator<DomTreeNode*> FI = po_begin(N), FE = po_end(N); FI != FE;
    ++FI) {
    findRegionsWithEntry(FI->getBlock(), ShortCut);
  }
}

Region *RegionInfo::getTopMostParent(Region *region) {
  while (region->parent)
    region = region->getParent();

  return region;
}

void RegionInfo::buildRegionsTree(DomTreeNode *N, Region *region) {
  BasicBlock *BB = N->getBlock();

  // Passed region exit
  while (BB == region->getExit())
    region = region->getParent();

  BBtoRegionMap::iterator it = BBtoRegion.find(BB);

  // This basic block is a start block of a region. It is already in the
  // BBtoRegion relation. Only the child basic blocks have to be updated.
  if (it != BBtoRegion.end()) {
    Region *newRegion = it->second;
    region->addSubRegion(getTopMostParent(newRegion));
    region = newRegion;
  } else {
    BBtoRegion[BB] = region;
  }

  for (DomTreeNode::iterator CI = N->begin(), CE = N->end(); CI != CE; ++CI)
    buildRegionsTree(*CI, region);
}

void RegionInfo::releaseMemory() {
  BBtoRegion.clear();
  if (TopLevelRegion)
    delete TopLevelRegion;
  TopLevelRegion = 0;
}

RegionInfo::RegionInfo() : FunctionPass(ID) {
  initializeRegionInfoPass(*PassRegistry::getPassRegistry());
  TopLevelRegion = 0;
}

RegionInfo::~RegionInfo() {
  releaseMemory();
}

void RegionInfo::Calculate(Function &F) {
  // ShortCut a function where for every BB the exit of the largest region
  // starting with BB is stored. These regions can be threated as single BBS.
  // This improves performance on linear CFGs.
  BBtoBBMap ShortCut;

  scanForRegions(F, &ShortCut);
  BasicBlock *BB = &F.getEntryBlock();
  buildRegionsTree(DT->getNode(BB), TopLevelRegion);
}

bool RegionInfo::runOnFunction(Function &F) {
  releaseMemory();

  DT = &getAnalysis<DominatorTree>();
  PDT = &getAnalysis<PostDominatorTree>();
  DF = &getAnalysis<DominanceFrontier>();

  TopLevelRegion = new Region(&F.getEntryBlock(), 0, this, DT, 0);
  updateStatistics(TopLevelRegion);

  Calculate(F);

  return false;
}

void RegionInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTree>();
  AU.addRequired<PostDominatorTree>();
  AU.addRequired<DominanceFrontier>();
}

void RegionInfo::print(raw_ostream &OS, const Module *) const {
  OS << "Region tree:\n";
  TopLevelRegion->print(OS, true, 0, printStyle.getValue());
  OS << "End region tree\n";
}

void RegionInfo::verifyAnalysis() const {
  // Only do verification when user wants to, otherwise this expensive check
  // will be invoked by PMDataManager::verifyPreservedAnalysis when
  // a regionpass (marked PreservedAll) finish.
  if (!VerifyRegionInfo) return;

  TopLevelRegion->verifyRegionNest();
}

// Region pass manager support.
Region *RegionInfo::getRegionFor(BasicBlock *BB) const {
  BBtoRegionMap::const_iterator I=
    BBtoRegion.find(BB);
  return I != BBtoRegion.end() ? I->second : 0;
}

void RegionInfo::setRegionFor(BasicBlock *BB, Region *R) {
  BBtoRegion[BB] = R;
}

Region *RegionInfo::operator[](BasicBlock *BB) const {
  return getRegionFor(BB);
}

BasicBlock *RegionInfo::getMaxRegionExit(BasicBlock *BB) const {
  BasicBlock *Exit = NULL;

  while (true) {
    // Get largest region that starts at BB.
    Region *R = getRegionFor(BB);
    while (R && R->getParent() && R->getParent()->getEntry() == BB)
      R = R->getParent();

    // Get the single exit of BB.
    if (R && R->getEntry() == BB)
      Exit = R->getExit();
    else if (++succ_begin(BB) == succ_end(BB))
      Exit = *succ_begin(BB);
    else // No single exit exists.
      return Exit;

    // Get largest region that starts at Exit.
    Region *ExitR = getRegionFor(Exit);
    while (ExitR && ExitR->getParent()
           && ExitR->getParent()->getEntry() == Exit)
      ExitR = ExitR->getParent();

    for (pred_iterator PI = pred_begin(Exit), PE = pred_end(Exit); PI != PE;
         ++PI)
      if (!R->contains(*PI) && !ExitR->contains(*PI))
        break;

    // This stops infinite cycles.
    if (DT->dominates(Exit, BB))
      break;

    BB = Exit;
  }

  return Exit;
}

Region*
RegionInfo::getCommonRegion(Region *A, Region *B) const {
  assert (A && B && "One of the Regions is NULL");

  if (A->contains(B)) return A;

  while (!B->contains(A))
    B = B->getParent();

  return B;
}

Region*
RegionInfo::getCommonRegion(SmallVectorImpl<Region*> &Regions) const {
  Region* ret = Regions.back();
  Regions.pop_back();

  for (SmallVectorImpl<Region*>::const_iterator I = Regions.begin(),
       E = Regions.end(); I != E; ++I)
      ret = getCommonRegion(ret, *I);

  return ret;
}

Region*
RegionInfo::getCommonRegion(SmallVectorImpl<BasicBlock*> &BBs) const {
  Region* ret = getRegionFor(BBs.back());
  BBs.pop_back();

  for (SmallVectorImpl<BasicBlock*>::const_iterator I = BBs.begin(),
       E = BBs.end(); I != E; ++I)
      ret = getCommonRegion(ret, getRegionFor(*I));

  return ret;
}

void RegionInfo::splitBlock(BasicBlock* NewBB, BasicBlock *OldBB)
{
  Region *R = getRegionFor(OldBB);

  setRegionFor(NewBB, R);

  while (R->getEntry() == OldBB && !R->isTopLevelRegion()) {
    R->replaceEntry(NewBB);
    R = R->getParent();
  }

  setRegionFor(OldBB, R);
}

char RegionInfo::ID = 0;
INITIALIZE_PASS_BEGIN(RegionInfo, "regions",
                "Detect single entry single exit regions", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(DominanceFrontier)
INITIALIZE_PASS_END(RegionInfo, "regions",
                "Detect single entry single exit regions", true, true)

// Create methods available outside of this file, to use them
// "include/llvm/LinkAllPasses.h". Otherwise the pass would be deleted by
// the link time optimization.

namespace llvm {
  FunctionPass *createRegionInfoPass() {
    return new RegionInfo();
  }
}

