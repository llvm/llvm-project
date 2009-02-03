//===-- PredicateSimplifier.cpp - Path Sensitive Simplifier ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Path-sensitive optimizer. In a branch where x == y, replace uses of
// x with y. Permits further optimization, such as the elimination of
// the unreachable call:
//
// void test(int *p, int *q)
// {
//   if (p != q)
//     return;
// 
//   if (*p != *q)
//     foo(); // unreachable
// }
//
//===----------------------------------------------------------------------===//
//
// The InequalityGraph focusses on four properties; equals, not equals,
// less-than and less-than-or-equals-to. The greater-than forms are also held
// just to allow walking from a lesser node to a greater one. These properties
// are stored in a lattice; LE can become LT or EQ, NE can become LT or GT.
//
// These relationships define a graph between values of the same type. Each
// Value is stored in a map table that retrieves the associated Node. This
// is how EQ relationships are stored; the map contains pointers from equal
// Value to the same node. The node contains a most canonical Value* form
// and the list of known relationships with other nodes.
//
// If two nodes are known to be inequal, then they will contain pointers to
// each other with an "NE" relationship. If node getNode(%x) is less than
// getNode(%y), then the %x node will contain <%y, GT> and %y will contain
// <%x, LT>. This allows us to tie nodes together into a graph like this:
//
//   %a < %b < %c < %d
//
// with four nodes representing the properties. The InequalityGraph provides
// querying with "isRelatedBy" and mutators "addEquality" and "addInequality".
// To find a relationship, we start with one of the nodes any binary search
// through its list to find where the relationships with the second node start.
// Then we iterate through those to find the first relationship that dominates
// our context node.
//
// To create these properties, we wait until a branch or switch instruction
// implies that a particular value is true (or false). The VRPSolver is
// responsible for analyzing the variable and seeing what new inferences
// can be made from each property. For example:
//
//   %P = icmp ne i32* %ptr, null
//   %a = and i1 %P, %Q
//   br i1 %a label %cond_true, label %cond_false
//
// For the true branch, the VRPSolver will start with %a EQ true and look at
// the definition of %a and find that it can infer that %P and %Q are both
// true. From %P being true, it can infer that %ptr NE null. For the false
// branch it can't infer anything from the "and" instruction.
//
// Besides branches, we can also infer properties from instruction that may
// have undefined behaviour in certain cases. For example, the dividend of
// a division may never be zero. After the division instruction, we may assume
// that the dividend is not equal to zero.
//
//===----------------------------------------------------------------------===//
//
// The ValueRanges class stores the known integer bounds of a Value. When we
// encounter i8 %a u< %b, the ValueRanges stores that %a = [1, 255] and
// %b = [0, 254].
//
// It never stores an empty range, because that means that the code is
// unreachable. It never stores a single-element range since that's an equality
// relationship and better stored in the InequalityGraph, nor an empty range
// since that is better stored in UnreachableBlocks.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "predsimplify"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <deque>
#include <stack>
using namespace llvm;

STATISTIC(NumVarsReplaced, "Number of argument substitutions");
STATISTIC(NumInstruction , "Number of instructions removed");
STATISTIC(NumSimple      , "Number of simple replacements");
STATISTIC(NumBlocks      , "Number of blocks marked unreachable");
STATISTIC(NumSnuggle     , "Number of comparisons snuggled");

namespace {
  class DomTreeDFS {
  public:
    class Node {
      friend class DomTreeDFS;
    public:
      typedef std::vector<Node *>::iterator       iterator;
      typedef std::vector<Node *>::const_iterator const_iterator;

      unsigned getDFSNumIn()  const { return DFSin;  }
      unsigned getDFSNumOut() const { return DFSout; }

      BasicBlock *getBlock() const { return BB; }

      iterator begin() { return Children.begin(); }
      iterator end()   { return Children.end();   }

      const_iterator begin() const { return Children.begin(); }
      const_iterator end()   const { return Children.end();   }

      bool dominates(const Node *N) const {
        return DFSin <= N->DFSin && DFSout >= N->DFSout;
      }

      bool DominatedBy(const Node *N) const {
        return N->dominates(this);
      }

      /// Sorts by the number of descendants. With this, you can iterate
      /// through a sorted list and the first matching entry is the most
      /// specific match for your basic block. The order provided is stable;
      /// DomTreeDFS::Nodes with the same number of descendants are sorted by
      /// DFS in number.
      bool operator<(const Node &N) const {
        unsigned   spread =   DFSout -   DFSin;
        unsigned N_spread = N.DFSout - N.DFSin;
        if (spread == N_spread) return DFSin < N.DFSin;
        return spread < N_spread;
      }
      bool operator>(const Node &N) const { return N < *this; }

    private:
      unsigned DFSin, DFSout;
      BasicBlock *BB;

      std::vector<Node *> Children;
    };

    // XXX: this may be slow. Instead of using "new" for each node, consider
    // putting them in a vector to keep them contiguous.
    explicit DomTreeDFS(DominatorTree *DT) {
      std::stack<std::pair<Node *, DomTreeNode *> > S;

      Entry = new Node;
      Entry->BB = DT->getRootNode()->getBlock();
      S.push(std::make_pair(Entry, DT->getRootNode()));

      NodeMap[Entry->BB] = Entry;

      while (!S.empty()) {
        std::pair<Node *, DomTreeNode *> &Pair = S.top();
        Node *N = Pair.first;
        DomTreeNode *DTNode = Pair.second;
        S.pop();

        for (DomTreeNode::iterator I = DTNode->begin(), E = DTNode->end();
             I != E; ++I) {
          Node *NewNode = new Node;
          NewNode->BB = (*I)->getBlock();
          N->Children.push_back(NewNode);
          S.push(std::make_pair(NewNode, *I));

          NodeMap[NewNode->BB] = NewNode;
        }
      }

      renumber();

#ifndef NDEBUG
      DEBUG(dump());
#endif
    }

#ifndef NDEBUG
    virtual
#endif
    ~DomTreeDFS() {
      std::stack<Node *> S;

      S.push(Entry);
      while (!S.empty()) {
        Node *N = S.top(); S.pop();

        for (Node::iterator I = N->begin(), E = N->end(); I != E; ++I)
          S.push(*I);

        delete N;
      }
    }

    /// getRootNode - This returns the entry node for the CFG of the function.
    Node *getRootNode() const { return Entry; }

    /// getNodeForBlock - return the node for the specified basic block.
    Node *getNodeForBlock(BasicBlock *BB) const {
      if (!NodeMap.count(BB)) return 0;
      return const_cast<DomTreeDFS*>(this)->NodeMap[BB];
    }

    /// dominates - returns true if the basic block for I1 dominates that of
    /// the basic block for I2. If the instructions belong to the same basic
    /// block, the instruction first instruction sequentially in the block is
    /// considered dominating.
    bool dominates(Instruction *I1, Instruction *I2) {
      BasicBlock *BB1 = I1->getParent(),
                 *BB2 = I2->getParent();
      if (BB1 == BB2) {
        if (isa<TerminatorInst>(I1)) return false;
        if (isa<TerminatorInst>(I2)) return true;
        if ( isa<PHINode>(I1) && !isa<PHINode>(I2)) return true;
        if (!isa<PHINode>(I1) &&  isa<PHINode>(I2)) return false;

        for (BasicBlock::const_iterator I = BB2->begin(), E = BB2->end();
             I != E; ++I) {
          if (&*I == I1) return true;
          else if (&*I == I2) return false;
        }
        assert(!"Instructions not found in parent BasicBlock?");
      } else {
        Node *Node1 = getNodeForBlock(BB1),
             *Node2 = getNodeForBlock(BB2);
        return Node1 && Node2 && Node1->dominates(Node2);
      }
      return false; // Not reached
    }

  private:
    /// renumber - calculates the depth first search numberings and applies
    /// them onto the nodes.
    void renumber() {
      std::stack<std::pair<Node *, Node::iterator> > S;
      unsigned n = 0;

      Entry->DFSin = ++n;
      S.push(std::make_pair(Entry, Entry->begin()));

      while (!S.empty()) {
        std::pair<Node *, Node::iterator> &Pair = S.top();
        Node *N = Pair.first;
        Node::iterator &I = Pair.second;

        if (I == N->end()) {
          N->DFSout = ++n;
          S.pop();
        } else {
          Node *Next = *I++;
          Next->DFSin = ++n;
          S.push(std::make_pair(Next, Next->begin()));
        }
      }
    }

#ifndef NDEBUG
    virtual void dump() const {
      dump(*cerr.stream());
    }

    void dump(std::ostream &os) const {
      os << "Predicate simplifier DomTreeDFS: \n";
      dump(Entry, 0, os);
      os << "\n\n";
    }

    void dump(Node *N, int depth, std::ostream &os) const {
      ++depth;
      for (int i = 0; i < depth; ++i) { os << " "; }
      os << "[" << depth << "] ";

      os << N->getBlock()->getName() << " (" << N->getDFSNumIn()
         << ", " << N->getDFSNumOut() << ")\n";

      for (Node::iterator I = N->begin(), E = N->end(); I != E; ++I)
        dump(*I, depth, os);
    }
#endif

    Node *Entry;
    std::map<BasicBlock *, Node *> NodeMap;
  };

  // SLT SGT ULT UGT EQ
  //   0   1   0   1  0 -- GT                  10
  //   0   1   0   1  1 -- GE                  11
  //   0   1   1   0  0 -- SGTULT              12
  //   0   1   1   0  1 -- SGEULE              13
  //   0   1   1   1  0 -- SGT                 14
  //   0   1   1   1  1 -- SGE                 15
  //   1   0   0   1  0 -- SLTUGT              18
  //   1   0   0   1  1 -- SLEUGE              19
  //   1   0   1   0  0 -- LT                  20
  //   1   0   1   0  1 -- LE                  21
  //   1   0   1   1  0 -- SLT                 22
  //   1   0   1   1  1 -- SLE                 23
  //   1   1   0   1  0 -- UGT                 26
  //   1   1   0   1  1 -- UGE                 27
  //   1   1   1   0  0 -- ULT                 28
  //   1   1   1   0  1 -- ULE                 29
  //   1   1   1   1  0 -- NE                  30
  enum LatticeBits {
    EQ_BIT = 1, UGT_BIT = 2, ULT_BIT = 4, SGT_BIT = 8, SLT_BIT = 16
  };
  enum LatticeVal {
    GT = SGT_BIT | UGT_BIT,
    GE = GT | EQ_BIT,
    LT = SLT_BIT | ULT_BIT,
    LE = LT | EQ_BIT,
    NE = SLT_BIT | SGT_BIT | ULT_BIT | UGT_BIT,
    SGTULT = SGT_BIT | ULT_BIT,
    SGEULE = SGTULT | EQ_BIT,
    SLTUGT = SLT_BIT | UGT_BIT,
    SLEUGE = SLTUGT | EQ_BIT,
    ULT = SLT_BIT | SGT_BIT | ULT_BIT,
    UGT = SLT_BIT | SGT_BIT | UGT_BIT,
    SLT = SLT_BIT | ULT_BIT | UGT_BIT,
    SGT = SGT_BIT | ULT_BIT | UGT_BIT,
    SLE = SLT | EQ_BIT,
    SGE = SGT | EQ_BIT,
    ULE = ULT | EQ_BIT,
    UGE = UGT | EQ_BIT
  };

#ifndef NDEBUG
  /// validPredicate - determines whether a given value is actually a lattice
  /// value. Only used in assertions or debugging.
  static bool validPredicate(LatticeVal LV) {
    switch (LV) {
      case GT: case GE: case LT: case LE: case NE:
      case SGTULT: case SGT: case SGEULE:
      case SLTUGT: case SLT: case SLEUGE:
      case ULT: case UGT:
      case SLE: case SGE: case ULE: case UGE:
        return true;
      default:
        return false;
    }
  }
#endif

  /// reversePredicate - reverse the direction of the inequality
  static LatticeVal reversePredicate(LatticeVal LV) {
    unsigned reverse = LV ^ (SLT_BIT|SGT_BIT|ULT_BIT|UGT_BIT); //preserve EQ_BIT

    if ((reverse & (SLT_BIT|SGT_BIT)) == 0)
      reverse |= (SLT_BIT|SGT_BIT);

    if ((reverse & (ULT_BIT|UGT_BIT)) == 0)
      reverse |= (ULT_BIT|UGT_BIT);

    LatticeVal Rev = static_cast<LatticeVal>(reverse);
    assert(validPredicate(Rev) && "Failed reversing predicate.");
    return Rev;
  }

  /// ValueNumbering stores the scope-specific value numbers for a given Value.
  class VISIBILITY_HIDDEN ValueNumbering {

    /// VNPair is a tuple of {Value, index number, DomTreeDFS::Node}. It
    /// includes the comparison operators necessary to allow you to store it
    /// in a sorted vector.
    class VISIBILITY_HIDDEN VNPair {
    public:
      Value *V;
      unsigned index;
      DomTreeDFS::Node *Subtree;

      VNPair(Value *V, unsigned index, DomTreeDFS::Node *Subtree)
        : V(V), index(index), Subtree(Subtree) {}

      bool operator==(const VNPair &RHS) const {
        return V == RHS.V && Subtree == RHS.Subtree;
      }

      bool operator<(const VNPair &RHS) const {
        if (V != RHS.V) return V < RHS.V;
        return *Subtree < *RHS.Subtree;
      }

      bool operator<(Value *RHS) const {
        return V < RHS;
      }

      bool operator>(Value *RHS) const {
        return V > RHS;
      }

      friend bool operator<(Value *RHS, const VNPair &pair) {
        return pair.operator>(RHS);
      }
    };

    typedef std::vector<VNPair> VNMapType;
    VNMapType VNMap;

    /// The canonical choice for value number at index.
    std::vector<Value *> Values;

    DomTreeDFS *DTDFS;

  public:
#ifndef NDEBUG
    virtual ~ValueNumbering() {}
    virtual void dump() {
      dump(*cerr.stream());
    }

    void dump(std::ostream &os) {
      for (unsigned i = 1; i <= Values.size(); ++i) {
        os << i << " = ";
        WriteAsOperand(os, Values[i-1]);
        os << " {";
        for (unsigned j = 0; j < VNMap.size(); ++j) {
          if (VNMap[j].index == i) {
            WriteAsOperand(os, VNMap[j].V);
            os << " (" << VNMap[j].Subtree->getDFSNumIn() << ")  ";
          }
        }
        os << "}\n";
      }
    }
#endif

    /// compare - returns true if V1 is a better canonical value than V2.
    bool compare(Value *V1, Value *V2) const {
      if (isa<Constant>(V1))
        return !isa<Constant>(V2);
      else if (isa<Constant>(V2))
        return false;
      else if (isa<Argument>(V1))
        return !isa<Argument>(V2);
      else if (isa<Argument>(V2))
        return false;

      Instruction *I1 = dyn_cast<Instruction>(V1);
      Instruction *I2 = dyn_cast<Instruction>(V2);

      if (!I1 || !I2)
        return V1->getNumUses() < V2->getNumUses();

      return DTDFS->dominates(I1, I2);
    }

    ValueNumbering(DomTreeDFS *DTDFS) : DTDFS(DTDFS) {}

    /// valueNumber - finds the value number for V under the Subtree. If
    /// there is no value number, returns zero.
    unsigned valueNumber(Value *V, DomTreeDFS::Node *Subtree) {
      if (!(isa<Constant>(V) || isa<Argument>(V) || isa<Instruction>(V))
          || V->getType() == Type::VoidTy) return 0;

      VNMapType::iterator E = VNMap.end();
      VNPair pair(V, 0, Subtree);
      VNMapType::iterator I = std::lower_bound(VNMap.begin(), E, pair);
      while (I != E && I->V == V) {
        if (I->Subtree->dominates(Subtree))
          return I->index;
        ++I;
      }
      return 0;
    }

    /// getOrInsertVN - always returns a value number, creating it if necessary.
    unsigned getOrInsertVN(Value *V, DomTreeDFS::Node *Subtree) {
      if (unsigned n = valueNumber(V, Subtree))
        return n;
      else
        return newVN(V);
    }

    /// newVN - creates a new value number. Value V must not already have a
    /// value number assigned.
    unsigned newVN(Value *V) {
      assert((isa<Constant>(V) || isa<Argument>(V) || isa<Instruction>(V)) &&
             "Bad Value for value numbering.");
      assert(V->getType() != Type::VoidTy && "Won't value number a void value");

      Values.push_back(V);

      VNPair pair = VNPair(V, Values.size(), DTDFS->getRootNode());
      VNMapType::iterator I = std::lower_bound(VNMap.begin(), VNMap.end(), pair);
      assert((I == VNMap.end() || value(I->index) != V) &&
             "Attempt to create a duplicate value number.");
      VNMap.insert(I, pair);

      return Values.size();
    }

    /// value - returns the Value associated with a value number.
    Value *value(unsigned index) const {
      assert(index != 0 && "Zero index is reserved for not found.");
      assert(index <= Values.size() && "Index out of range.");
      return Values[index-1];
    }

    /// canonicalize - return a Value that is equal to V under Subtree.
    Value *canonicalize(Value *V, DomTreeDFS::Node *Subtree) {
      if (isa<Constant>(V)) return V;

      if (unsigned n = valueNumber(V, Subtree))
        return value(n);
      else
        return V;
    }

    /// addEquality - adds that value V belongs to the set of equivalent
    /// values defined by value number n under Subtree.
    void addEquality(unsigned n, Value *V, DomTreeDFS::Node *Subtree) {
      assert(canonicalize(value(n), Subtree) == value(n) &&
             "Node's 'canonical' choice isn't best within this subtree.");

      // Suppose that we are given "%x -> node #1 (%y)". The problem is that
      // we may already have "%z -> node #2 (%x)" somewhere above us in the
      // graph. We need to find those edges and add "%z -> node #1 (%y)"
      // to keep the lookups canonical.

      std::vector<Value *> ToRepoint(1, V);

      if (unsigned Conflict = valueNumber(V, Subtree)) {
        for (VNMapType::iterator I = VNMap.begin(), E = VNMap.end();
             I != E; ++I) {
          if (I->index == Conflict && I->Subtree->dominates(Subtree))
            ToRepoint.push_back(I->V);
        }
      }

      for (std::vector<Value *>::iterator VI = ToRepoint.begin(),
           VE = ToRepoint.end(); VI != VE; ++VI) {
        Value *V = *VI;

        VNPair pair(V, n, Subtree);
        VNMapType::iterator B = VNMap.begin(), E = VNMap.end();
        VNMapType::iterator I = std::lower_bound(B, E, pair);
        if (I != E && I->V == V && I->Subtree == Subtree)
          I->index = n; // Update best choice
        else
          VNMap.insert(I, pair); // New Value

        // XXX: we currently don't have to worry about updating values with
        // more specific Subtrees, but we will need to for PHI node support.

#ifndef NDEBUG
        Value *V_n = value(n);
        if (isa<Constant>(V) && isa<Constant>(V_n)) {
          assert(V == V_n && "Constant equals different constant?");
        }
#endif
      }
    }

    /// remove - removes all references to value V.
    void remove(Value *V) {
      VNMapType::iterator B = VNMap.begin(), E = VNMap.end();
      VNPair pair(V, 0, DTDFS->getRootNode());
      VNMapType::iterator J = std::upper_bound(B, E, pair);
      VNMapType::iterator I = J;

      while (I != B && (I == E || I->V == V)) --I;

      VNMap.erase(I, J);
    }
  };

  /// The InequalityGraph stores the relationships between values.
  /// Each Value in the graph is assigned to a Node. Nodes are pointer
  /// comparable for equality. The caller is expected to maintain the logical
  /// consistency of the system.
  ///
  /// The InequalityGraph class may invalidate Node*s after any mutator call.
  /// @brief The InequalityGraph stores the relationships between values.
  class VISIBILITY_HIDDEN InequalityGraph {
    ValueNumbering &VN;
    DomTreeDFS::Node *TreeRoot;

    InequalityGraph();                  // DO NOT IMPLEMENT
    InequalityGraph(InequalityGraph &); // DO NOT IMPLEMENT
  public:
    InequalityGraph(ValueNumbering &VN, DomTreeDFS::Node *TreeRoot)
      : VN(VN), TreeRoot(TreeRoot) {}

    class Node;

    /// An Edge is contained inside a Node making one end of the edge implicit
    /// and contains a pointer to the other end. The edge contains a lattice
    /// value specifying the relationship and an DomTreeDFS::Node specifying
    /// the root in the dominator tree to which this edge applies.
    class VISIBILITY_HIDDEN Edge {
    public:
      Edge(unsigned T, LatticeVal V, DomTreeDFS::Node *ST)
        : To(T), LV(V), Subtree(ST) {}

      unsigned To;
      LatticeVal LV;
      DomTreeDFS::Node *Subtree;

      bool operator<(const Edge &edge) const {
        if (To != edge.To) return To < edge.To;
        return *Subtree < *edge.Subtree;
      }

      bool operator<(unsigned to) const {
        return To < to;
      }

      bool operator>(unsigned to) const {
        return To > to;
      }

      friend bool operator<(unsigned to, const Edge &edge) {
        return edge.operator>(to);
      }
    };

    /// A single node in the InequalityGraph. This stores the canonical Value
    /// for the node, as well as the relationships with the neighbours.
    ///
    /// @brief A single node in the InequalityGraph.
    class VISIBILITY_HIDDEN Node {
      friend class InequalityGraph;

      typedef SmallVector<Edge, 4> RelationsType;
      RelationsType Relations;

      // TODO: can this idea improve performance?
      //friend class std::vector<Node>;
      //Node(Node &N) { RelationsType.swap(N.RelationsType); }

    public:
      typedef RelationsType::iterator       iterator;
      typedef RelationsType::const_iterator const_iterator;

#ifndef NDEBUG
      virtual ~Node() {}
      virtual void dump() const {
        dump(*cerr.stream());
      }
    private:
      void dump(std::ostream &os) const {
        static const std::string names[32] =
          { "000000", "000001", "000002", "000003", "000004", "000005",
            "000006", "000007", "000008", "000009", "     >", "    >=",
            "  s>u<", "s>=u<=", "    s>", "   s>=", "000016", "000017",
            "  s<u>", "s<=u>=", "     <", "    <=", "    s<", "   s<=",
            "000024", "000025", "    u>", "   u>=", "    u<", "   u<=",
            "    !=", "000031" };
        for (Node::const_iterator NI = begin(), NE = end(); NI != NE; ++NI) {
          os << names[NI->LV] << " " << NI->To
             << " (" << NI->Subtree->getDFSNumIn() << "), ";
        }
      }
    public:
#endif

      iterator begin()             { return Relations.begin(); }
      iterator end()               { return Relations.end();   }
      const_iterator begin() const { return Relations.begin(); }
      const_iterator end()   const { return Relations.end();   }

      iterator find(unsigned n, DomTreeDFS::Node *Subtree) {
        iterator E = end();
        for (iterator I = std::lower_bound(begin(), E, n);
             I != E && I->To == n; ++I) {
          if (Subtree->DominatedBy(I->Subtree))
            return I;
        }
        return E;
      }

      const_iterator find(unsigned n, DomTreeDFS::Node *Subtree) const {
        const_iterator E = end();
        for (const_iterator I = std::lower_bound(begin(), E, n);
             I != E && I->To == n; ++I) {
          if (Subtree->DominatedBy(I->Subtree))
            return I;
        }
        return E;
      }

      /// update - updates the lattice value for a given node, creating a new
      /// entry if one doesn't exist. The new lattice value must not be
      /// inconsistent with any previously existing value.
      void update(unsigned n, LatticeVal R, DomTreeDFS::Node *Subtree) {
        assert(validPredicate(R) && "Invalid predicate.");

        Edge edge(n, R, Subtree);
        iterator B = begin(), E = end();
        iterator I = std::lower_bound(B, E, edge);

        iterator J = I;
        while (J != E && J->To == n) {
          if (Subtree->DominatedBy(J->Subtree))
            break;
          ++J;
        }

        if (J != E && J->To == n) {
          edge.LV = static_cast<LatticeVal>(J->LV & R);
          assert(validPredicate(edge.LV) && "Invalid union of lattice values.");

          if (edge.LV == J->LV)
            return; // This update adds nothing new.
        }

        if (I != B) {
          // We also have to tighten any edge beneath our update.
          for (iterator K = I - 1; K->To == n; --K) {
            if (K->Subtree->DominatedBy(Subtree)) {
              LatticeVal LV = static_cast<LatticeVal>(K->LV & edge.LV);
              assert(validPredicate(LV) && "Invalid union of lattice values");
              K->LV = LV;
            }
            if (K == B) break;
          }
        }

        // Insert new edge at Subtree if it isn't already there.
        if (I == E || I->To != n || Subtree != I->Subtree)
          Relations.insert(I, edge);
      }
    };

  private:

    std::vector<Node> Nodes;

  public:
    /// node - returns the node object at a given value number. The pointer
    /// returned may be invalidated on the next call to node().
    Node *node(unsigned index) {
      assert(VN.value(index)); // This triggers the necessary checks.
      if (Nodes.size() < index) Nodes.resize(index);
      return &Nodes[index-1];
    }

    /// isRelatedBy - true iff n1 op n2
    bool isRelatedBy(unsigned n1, unsigned n2, DomTreeDFS::Node *Subtree,
                     LatticeVal LV) {
      if (n1 == n2) return LV & EQ_BIT;

      Node *N1 = node(n1);
      Node::iterator I = N1->find(n2, Subtree), E = N1->end();
      if (I != E) return (I->LV & LV) == I->LV;

      return false;
    }

    // The add* methods assume that your input is logically valid and may 
    // assertion-fail or infinitely loop if you attempt a contradiction.

    /// addInequality - Sets n1 op n2.
    /// It is also an error to call this on an inequality that is already true.
    void addInequality(unsigned n1, unsigned n2, DomTreeDFS::Node *Subtree,
                       LatticeVal LV1) {
      assert(n1 != n2 && "A node can't be inequal to itself.");

      if (LV1 != NE)
        assert(!isRelatedBy(n1, n2, Subtree, reversePredicate(LV1)) &&
               "Contradictory inequality.");

      // Suppose we're adding %n1 < %n2. Find all the %a < %n1 and
      // add %a < %n2 too. This keeps the graph fully connected.
      if (LV1 != NE) {
        // Break up the relationship into signed and unsigned comparison parts.
        // If the signed parts of %a op1 %n1 match that of %n1 op2 %n2, and
        // op1 and op2 aren't NE, then add %a op3 %n2. The new relationship
        // should have the EQ_BIT iff it's set for both op1 and op2.

        unsigned LV1_s = LV1 & (SLT_BIT|SGT_BIT);
        unsigned LV1_u = LV1 & (ULT_BIT|UGT_BIT);

        for (Node::iterator I = node(n1)->begin(), E = node(n1)->end(); I != E; ++I) {
          if (I->LV != NE && I->To != n2) {

            DomTreeDFS::Node *Local_Subtree = NULL;
            if (Subtree->DominatedBy(I->Subtree))
              Local_Subtree = Subtree;
            else if (I->Subtree->DominatedBy(Subtree))
              Local_Subtree = I->Subtree;

            if (Local_Subtree) {
              unsigned new_relationship = 0;
              LatticeVal ILV = reversePredicate(I->LV);
              unsigned ILV_s = ILV & (SLT_BIT|SGT_BIT);
              unsigned ILV_u = ILV & (ULT_BIT|UGT_BIT);

              if (LV1_s != (SLT_BIT|SGT_BIT) && ILV_s == LV1_s)
                new_relationship |= ILV_s;
              if (LV1_u != (ULT_BIT|UGT_BIT) && ILV_u == LV1_u)
                new_relationship |= ILV_u;

              if (new_relationship) {
                if ((new_relationship & (SLT_BIT|SGT_BIT)) == 0)
                  new_relationship |= (SLT_BIT|SGT_BIT);
                if ((new_relationship & (ULT_BIT|UGT_BIT)) == 0)
                  new_relationship |= (ULT_BIT|UGT_BIT);
                if ((LV1 & EQ_BIT) && (ILV & EQ_BIT))
                  new_relationship |= EQ_BIT;

                LatticeVal NewLV = static_cast<LatticeVal>(new_relationship);

                node(I->To)->update(n2, NewLV, Local_Subtree);
                node(n2)->update(I->To, reversePredicate(NewLV), Local_Subtree);
              }
            }
          }
        }

        for (Node::iterator I = node(n2)->begin(), E = node(n2)->end(); I != E; ++I) {
          if (I->LV != NE && I->To != n1) {
            DomTreeDFS::Node *Local_Subtree = NULL;
            if (Subtree->DominatedBy(I->Subtree))
              Local_Subtree = Subtree;
            else if (I->Subtree->DominatedBy(Subtree))
              Local_Subtree = I->Subtree;

            if (Local_Subtree) {
              unsigned new_relationship = 0;
              unsigned ILV_s = I->LV & (SLT_BIT|SGT_BIT);
              unsigned ILV_u = I->LV & (ULT_BIT|UGT_BIT);

              if (LV1_s != (SLT_BIT|SGT_BIT) && ILV_s == LV1_s)
                new_relationship |= ILV_s;

              if (LV1_u != (ULT_BIT|UGT_BIT) && ILV_u == LV1_u)
                new_relationship |= ILV_u;

              if (new_relationship) {
                if ((new_relationship & (SLT_BIT|SGT_BIT)) == 0)
                  new_relationship |= (SLT_BIT|SGT_BIT);
                if ((new_relationship & (ULT_BIT|UGT_BIT)) == 0)
                  new_relationship |= (ULT_BIT|UGT_BIT);
                if ((LV1 & EQ_BIT) && (I->LV & EQ_BIT))
                  new_relationship |= EQ_BIT;

                LatticeVal NewLV = static_cast<LatticeVal>(new_relationship);

                node(n1)->update(I->To, NewLV, Local_Subtree);
                node(I->To)->update(n1, reversePredicate(NewLV), Local_Subtree);
              }
            }
          }
        }
      }

      node(n1)->update(n2, LV1, Subtree);
      node(n2)->update(n1, reversePredicate(LV1), Subtree);
    }

    /// remove - removes a node from the graph by removing all references to
    /// and from it.
    void remove(unsigned n) {
      Node *N = node(n);
      for (Node::iterator NI = N->begin(), NE = N->end(); NI != NE; ++NI) {
        Node::iterator Iter = node(NI->To)->find(n, TreeRoot);
        do {
          node(NI->To)->Relations.erase(Iter);
          Iter = node(NI->To)->find(n, TreeRoot);
        } while (Iter != node(NI->To)->end());
      }
      N->Relations.clear();
    }

#ifndef NDEBUG
    virtual ~InequalityGraph() {}
    virtual void dump() {
      dump(*cerr.stream());
    }

    void dump(std::ostream &os) {
      for (unsigned i = 1; i <= Nodes.size(); ++i) {
        os << i << " = {";
        node(i)->dump(os);
        os << "}\n";
      }
    }
#endif
  };

  class VRPSolver;

  /// ValueRanges tracks the known integer ranges and anti-ranges of the nodes
  /// in the InequalityGraph.
  class VISIBILITY_HIDDEN ValueRanges {
    ValueNumbering &VN;
    TargetData *TD;

    class VISIBILITY_HIDDEN ScopedRange {
      typedef std::vector<std::pair<DomTreeDFS::Node *, ConstantRange> >
              RangeListType;
      RangeListType RangeList;

      static bool swo(const std::pair<DomTreeDFS::Node *, ConstantRange> &LHS,
                      const std::pair<DomTreeDFS::Node *, ConstantRange> &RHS) {
        return *LHS.first < *RHS.first;
      }

    public:
#ifndef NDEBUG
      virtual ~ScopedRange() {}
      virtual void dump() const {
        dump(*cerr.stream());
      }

      void dump(std::ostream &os) const {
        os << "{";
        for (const_iterator I = begin(), E = end(); I != E; ++I) {
          os << &I->second << " (" << I->first->getDFSNumIn() << "), ";
        }
        os << "}";
      }
#endif

      typedef RangeListType::iterator       iterator;
      typedef RangeListType::const_iterator const_iterator;

      iterator begin() { return RangeList.begin(); }
      iterator end()   { return RangeList.end(); }
      const_iterator begin() const { return RangeList.begin(); }
      const_iterator end()   const { return RangeList.end(); }

      iterator find(DomTreeDFS::Node *Subtree) {
        static ConstantRange empty(1, false);
        iterator E = end();
        iterator I = std::lower_bound(begin(), E,
                                      std::make_pair(Subtree, empty), swo);

        while (I != E && !I->first->dominates(Subtree)) ++I;
        return I;
      }

      const_iterator find(DomTreeDFS::Node *Subtree) const {
        static const ConstantRange empty(1, false);
        const_iterator E = end();
        const_iterator I = std::lower_bound(begin(), E,
                                            std::make_pair(Subtree, empty), swo);

        while (I != E && !I->first->dominates(Subtree)) ++I;
        return I;
      }

      void update(const ConstantRange &CR, DomTreeDFS::Node *Subtree) {
        assert(!CR.isEmptySet() && "Empty ConstantRange.");
        assert(!CR.isSingleElement() && "Refusing to store single element.");

        static ConstantRange empty(1, false);
        iterator E = end();
        iterator I =
            std::lower_bound(begin(), E, std::make_pair(Subtree, empty), swo);

        if (I != end() && I->first == Subtree) {
          ConstantRange CR2 = I->second.maximalIntersectWith(CR);
          assert(!CR2.isEmptySet() && !CR2.isSingleElement() &&
                 "Invalid union of ranges.");
          I->second = CR2;
        } else
          RangeList.insert(I, std::make_pair(Subtree, CR));
      }
    };

    std::vector<ScopedRange> Ranges;

    void update(unsigned n, const ConstantRange &CR, DomTreeDFS::Node *Subtree){
      if (CR.isFullSet()) return;
      if (Ranges.size() < n) Ranges.resize(n);
      Ranges[n-1].update(CR, Subtree);
    }

    /// create - Creates a ConstantRange that matches the given LatticeVal
    /// relation with a given integer.
    ConstantRange create(LatticeVal LV, const ConstantRange &CR) {
      assert(!CR.isEmptySet() && "Can't deal with empty set.");

      if (LV == NE)
        return makeConstantRange(ICmpInst::ICMP_NE, CR);

      unsigned LV_s = LV & (SGT_BIT|SLT_BIT);
      unsigned LV_u = LV & (UGT_BIT|ULT_BIT);
      bool hasEQ = LV & EQ_BIT;

      ConstantRange Range(CR.getBitWidth());

      if (LV_s == SGT_BIT) {
        Range = Range.maximalIntersectWith(makeConstantRange(
                    hasEQ ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_SGT, CR));
      } else if (LV_s == SLT_BIT) {
        Range = Range.maximalIntersectWith(makeConstantRange(
                    hasEQ ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_SLT, CR));
      }

      if (LV_u == UGT_BIT) {
        Range = Range.maximalIntersectWith(makeConstantRange(
                    hasEQ ? ICmpInst::ICMP_UGE : ICmpInst::ICMP_UGT, CR));
      } else if (LV_u == ULT_BIT) {
        Range = Range.maximalIntersectWith(makeConstantRange(
                    hasEQ ? ICmpInst::ICMP_ULE : ICmpInst::ICMP_ULT, CR));
      }

      return Range;
    }

    /// makeConstantRange - Creates a ConstantRange representing the set of all
    /// value that match the ICmpInst::Predicate with any of the values in CR.
    ConstantRange makeConstantRange(ICmpInst::Predicate ICmpOpcode,
                                    const ConstantRange &CR) {
      uint32_t W = CR.getBitWidth();
      switch (ICmpOpcode) {
        default: assert(!"Invalid ICmp opcode to makeConstantRange()");
        case ICmpInst::ICMP_EQ:
          return ConstantRange(CR.getLower(), CR.getUpper());
        case ICmpInst::ICMP_NE:
          if (CR.isSingleElement())
            return ConstantRange(CR.getUpper(), CR.getLower());
          return ConstantRange(W);
        case ICmpInst::ICMP_ULT:
          return ConstantRange(APInt::getMinValue(W), CR.getUnsignedMax());
        case ICmpInst::ICMP_SLT:
          return ConstantRange(APInt::getSignedMinValue(W), CR.getSignedMax());
        case ICmpInst::ICMP_ULE: {
          APInt UMax(CR.getUnsignedMax());
          if (UMax.isMaxValue())
            return ConstantRange(W);
          return ConstantRange(APInt::getMinValue(W), UMax + 1);
        }
        case ICmpInst::ICMP_SLE: {
          APInt SMax(CR.getSignedMax());
          if (SMax.isMaxSignedValue() || (SMax+1).isMaxSignedValue())
            return ConstantRange(W);
          return ConstantRange(APInt::getSignedMinValue(W), SMax + 1);
        }
        case ICmpInst::ICMP_UGT:
          return ConstantRange(CR.getUnsignedMin() + 1, APInt::getNullValue(W));
        case ICmpInst::ICMP_SGT:
          return ConstantRange(CR.getSignedMin() + 1,
                               APInt::getSignedMinValue(W));
        case ICmpInst::ICMP_UGE: {
          APInt UMin(CR.getUnsignedMin());
          if (UMin.isMinValue())
            return ConstantRange(W);
          return ConstantRange(UMin, APInt::getNullValue(W));
        }
        case ICmpInst::ICMP_SGE: {
          APInt SMin(CR.getSignedMin());
          if (SMin.isMinSignedValue())
            return ConstantRange(W);
          return ConstantRange(SMin, APInt::getSignedMinValue(W));
        }
      }
    }

#ifndef NDEBUG
    bool isCanonical(Value *V, DomTreeDFS::Node *Subtree) {
      return V == VN.canonicalize(V, Subtree);
    }
#endif

  public:

    ValueRanges(ValueNumbering &VN, TargetData *TD) : VN(VN), TD(TD) {}

#ifndef NDEBUG
    virtual ~ValueRanges() {}

    virtual void dump() const {
      dump(*cerr.stream());
    }

    void dump(std::ostream &os) const {
      for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
        os << (i+1) << " = ";
        Ranges[i].dump(os);
        os << "\n";
      }
    }
#endif

    /// range - looks up the ConstantRange associated with a value number.
    ConstantRange range(unsigned n, DomTreeDFS::Node *Subtree) {
      assert(VN.value(n)); // performs range checks

      if (n <= Ranges.size()) {
        ScopedRange::iterator I = Ranges[n-1].find(Subtree);
        if (I != Ranges[n-1].end()) return I->second;
      }

      Value *V = VN.value(n);
      ConstantRange CR = range(V);
      return CR;
    }

    /// range - determine a range from a Value without performing any lookups.
    ConstantRange range(Value *V) const {
      if (ConstantInt *C = dyn_cast<ConstantInt>(V))
        return ConstantRange(C->getValue());
      else if (isa<ConstantPointerNull>(V))
        return ConstantRange(APInt::getNullValue(typeToWidth(V->getType())));
      else
        return ConstantRange(typeToWidth(V->getType()));
    }

    // typeToWidth - returns the number of bits necessary to store a value of
    // this type, or zero if unknown.
    uint32_t typeToWidth(const Type *Ty) const {
      if (TD)
        return TD->getTypeSizeInBits(Ty);
      else
        return Ty->getPrimitiveSizeInBits();
    }

    static bool isRelatedBy(const ConstantRange &CR1, const ConstantRange &CR2,
                            LatticeVal LV) {
      switch (LV) {
      default: assert(!"Impossible lattice value!");
      case NE:
        return CR1.maximalIntersectWith(CR2).isEmptySet();
      case ULT:
        return CR1.getUnsignedMax().ult(CR2.getUnsignedMin());
      case ULE:
        return CR1.getUnsignedMax().ule(CR2.getUnsignedMin());
      case UGT:
        return CR1.getUnsignedMin().ugt(CR2.getUnsignedMax());
      case UGE:
        return CR1.getUnsignedMin().uge(CR2.getUnsignedMax());
      case SLT:
        return CR1.getSignedMax().slt(CR2.getSignedMin());
      case SLE:
        return CR1.getSignedMax().sle(CR2.getSignedMin());
      case SGT:
        return CR1.getSignedMin().sgt(CR2.getSignedMax());
      case SGE:
        return CR1.getSignedMin().sge(CR2.getSignedMax());
      case LT:
        return CR1.getUnsignedMax().ult(CR2.getUnsignedMin()) &&
               CR1.getSignedMax().slt(CR2.getUnsignedMin());
      case LE:
        return CR1.getUnsignedMax().ule(CR2.getUnsignedMin()) &&
               CR1.getSignedMax().sle(CR2.getUnsignedMin());
      case GT:
        return CR1.getUnsignedMin().ugt(CR2.getUnsignedMax()) &&
               CR1.getSignedMin().sgt(CR2.getSignedMax());
      case GE:
        return CR1.getUnsignedMin().uge(CR2.getUnsignedMax()) &&
               CR1.getSignedMin().sge(CR2.getSignedMax());
      case SLTUGT:
        return CR1.getSignedMax().slt(CR2.getSignedMin()) &&
               CR1.getUnsignedMin().ugt(CR2.getUnsignedMax());
      case SLEUGE:
        return CR1.getSignedMax().sle(CR2.getSignedMin()) &&
               CR1.getUnsignedMin().uge(CR2.getUnsignedMax());
      case SGTULT:
        return CR1.getSignedMin().sgt(CR2.getSignedMax()) &&
               CR1.getUnsignedMax().ult(CR2.getUnsignedMin());
      case SGEULE:
        return CR1.getSignedMin().sge(CR2.getSignedMax()) &&
               CR1.getUnsignedMax().ule(CR2.getUnsignedMin());
      }
    }

    bool isRelatedBy(unsigned n1, unsigned n2, DomTreeDFS::Node *Subtree,
                     LatticeVal LV) {
      ConstantRange CR1 = range(n1, Subtree);
      ConstantRange CR2 = range(n2, Subtree);

      // True iff all values in CR1 are LV to all values in CR2.
      return isRelatedBy(CR1, CR2, LV);
    }

    void addToWorklist(Value *V, Constant *C, ICmpInst::Predicate Pred,
                       VRPSolver *VRP);
    void markBlock(VRPSolver *VRP);

    void mergeInto(Value **I, unsigned n, unsigned New,
                   DomTreeDFS::Node *Subtree, VRPSolver *VRP) {
      ConstantRange CR_New = range(New, Subtree);
      ConstantRange Merged = CR_New;

      for (; n != 0; ++I, --n) {
        unsigned i = VN.valueNumber(*I, Subtree);
        ConstantRange CR_Kill = i ? range(i, Subtree) : range(*I);
        if (CR_Kill.isFullSet()) continue;
        Merged = Merged.maximalIntersectWith(CR_Kill);
      }

      if (Merged.isFullSet() || Merged == CR_New) return;

      applyRange(New, Merged, Subtree, VRP);
    }

    void applyRange(unsigned n, const ConstantRange &CR,
                    DomTreeDFS::Node *Subtree, VRPSolver *VRP) {
      ConstantRange Merged = CR.maximalIntersectWith(range(n, Subtree));
      if (Merged.isEmptySet()) {
        markBlock(VRP);
        return;
      }

      if (const APInt *I = Merged.getSingleElement()) {
        Value *V = VN.value(n); // XXX: redesign worklist.
        const Type *Ty = V->getType();
        if (Ty->isInteger()) {
          addToWorklist(V, ConstantInt::get(*I), ICmpInst::ICMP_EQ, VRP);
          return;
        } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
          assert(*I == 0 && "Pointer is null but not zero?");
          addToWorklist(V, ConstantPointerNull::get(PTy),
                        ICmpInst::ICMP_EQ, VRP);
          return;
        }
      }

      update(n, Merged, Subtree);
    }

    void addNotEquals(unsigned n1, unsigned n2, DomTreeDFS::Node *Subtree,
                      VRPSolver *VRP) {
      ConstantRange CR1 = range(n1, Subtree);
      ConstantRange CR2 = range(n2, Subtree);

      uint32_t W = CR1.getBitWidth();

      if (const APInt *I = CR1.getSingleElement()) {
        if (CR2.isFullSet()) {
          ConstantRange NewCR2(CR1.getUpper(), CR1.getLower());
          applyRange(n2, NewCR2, Subtree, VRP);
        } else if (*I == CR2.getLower()) {
          APInt NewLower(CR2.getLower() + 1),
                NewUpper(CR2.getUpper());
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR2(NewLower, NewUpper);
          applyRange(n2, NewCR2, Subtree, VRP);
        } else if (*I == CR2.getUpper() - 1) {
          APInt NewLower(CR2.getLower()),
                NewUpper(CR2.getUpper() - 1);
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR2(NewLower, NewUpper);
          applyRange(n2, NewCR2, Subtree, VRP);
        }
      }

      if (const APInt *I = CR2.getSingleElement()) {
        if (CR1.isFullSet()) {
          ConstantRange NewCR1(CR2.getUpper(), CR2.getLower());
          applyRange(n1, NewCR1, Subtree, VRP);
        } else if (*I == CR1.getLower()) {
          APInt NewLower(CR1.getLower() + 1),
                NewUpper(CR1.getUpper());
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR1(NewLower, NewUpper);
          applyRange(n1, NewCR1, Subtree, VRP);
        } else if (*I == CR1.getUpper() - 1) {
          APInt NewLower(CR1.getLower()),
                NewUpper(CR1.getUpper() - 1);
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR1(NewLower, NewUpper);
          applyRange(n1, NewCR1, Subtree, VRP);
        }
      }
    }

    void addInequality(unsigned n1, unsigned n2, DomTreeDFS::Node *Subtree,
                       LatticeVal LV, VRPSolver *VRP) {
      assert(!isRelatedBy(n1, n2, Subtree, LV) && "Asked to do useless work.");

      if (LV == NE) {
        addNotEquals(n1, n2, Subtree, VRP);
        return;
      }

      ConstantRange CR1 = range(n1, Subtree);
      ConstantRange CR2 = range(n2, Subtree);

      if (!CR1.isSingleElement()) {
        ConstantRange NewCR1 = CR1.maximalIntersectWith(create(LV, CR2));
        if (NewCR1 != CR1)
          applyRange(n1, NewCR1, Subtree, VRP);
      }

      if (!CR2.isSingleElement()) {
        ConstantRange NewCR2 = CR2.maximalIntersectWith(
                                       create(reversePredicate(LV), CR1));
        if (NewCR2 != CR2)
          applyRange(n2, NewCR2, Subtree, VRP);
      }
    }
  };

  /// UnreachableBlocks keeps tracks of blocks that are for one reason or
  /// another discovered to be unreachable. This is used to cull the graph when
  /// analyzing instructions, and to mark blocks with the "unreachable"
  /// terminator instruction after the function has executed.
  class VISIBILITY_HIDDEN UnreachableBlocks {
  private:
    std::vector<BasicBlock *> DeadBlocks;

  public:
    /// mark - mark a block as dead
    void mark(BasicBlock *BB) {
      std::vector<BasicBlock *>::iterator E = DeadBlocks.end();
      std::vector<BasicBlock *>::iterator I =
        std::lower_bound(DeadBlocks.begin(), E, BB);

      if (I == E || *I != BB) DeadBlocks.insert(I, BB);
    }

    /// isDead - returns whether a block is known to be dead already
    bool isDead(BasicBlock *BB) {
      std::vector<BasicBlock *>::iterator E = DeadBlocks.end();
      std::vector<BasicBlock *>::iterator I =
        std::lower_bound(DeadBlocks.begin(), E, BB);

      return I != E && *I == BB;
    }

    /// kill - replace the dead blocks' terminator with an UnreachableInst.
    bool kill() {
      bool modified = false;
      for (std::vector<BasicBlock *>::iterator I = DeadBlocks.begin(),
           E = DeadBlocks.end(); I != E; ++I) {
        BasicBlock *BB = *I;

        DOUT << "unreachable block: " << BB->getName() << "\n";

        for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB);
             SI != SE; ++SI) {
          BasicBlock *Succ = *SI;
          Succ->removePredecessor(BB);
        }

        TerminatorInst *TI = BB->getTerminator();
        TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
        TI->eraseFromParent();
        new UnreachableInst(BB);
        ++NumBlocks;
        modified = true;
      }
      DeadBlocks.clear();
      return modified;
    }
  };

  /// VRPSolver keeps track of how changes to one variable affect other
  /// variables, and forwards changes along to the InequalityGraph. It
  /// also maintains the correct choice for "canonical" in the IG.
  /// @brief VRPSolver calculates inferences from a new relationship.
  class VISIBILITY_HIDDEN VRPSolver {
  private:
    friend class ValueRanges;

    struct Operation {
      Value *LHS, *RHS;
      ICmpInst::Predicate Op;

      BasicBlock *ContextBB; // XXX use a DomTreeDFS::Node instead
      Instruction *ContextInst;
    };
    std::deque<Operation> WorkList;

    ValueNumbering &VN;
    InequalityGraph &IG;
    UnreachableBlocks &UB;
    ValueRanges &VR;
    DomTreeDFS *DTDFS;
    DomTreeDFS::Node *Top;
    BasicBlock *TopBB;
    Instruction *TopInst;
    bool &modified;

    typedef InequalityGraph::Node Node;

    // below - true if the Instruction is dominated by the current context
    // block or instruction
    bool below(Instruction *I) {
      BasicBlock *BB = I->getParent();
      if (TopInst && TopInst->getParent() == BB) {
        if (isa<TerminatorInst>(TopInst)) return false;
        if (isa<TerminatorInst>(I)) return true;
        if ( isa<PHINode>(TopInst) && !isa<PHINode>(I)) return true;
        if (!isa<PHINode>(TopInst) &&  isa<PHINode>(I)) return false;

        for (BasicBlock::const_iterator Iter = BB->begin(), E = BB->end();
             Iter != E; ++Iter) {
          if (&*Iter == TopInst) return true;
          else if (&*Iter == I) return false;
        }
        assert(!"Instructions not found in parent BasicBlock?");
      } else {
        DomTreeDFS::Node *Node = DTDFS->getNodeForBlock(BB);
        if (!Node) return false;
        return Top->dominates(Node);
      }
      return false; // Not reached
    }

    // aboveOrBelow - true if the Instruction either dominates or is dominated
    // by the current context block or instruction
    bool aboveOrBelow(Instruction *I) {
      BasicBlock *BB = I->getParent();
      DomTreeDFS::Node *Node = DTDFS->getNodeForBlock(BB);
      if (!Node) return false;

      return Top == Node || Top->dominates(Node) || Node->dominates(Top);
    }

    bool makeEqual(Value *V1, Value *V2) {
      DOUT << "makeEqual(" << *V1 << ", " << *V2 << ")\n";
      DOUT << "context is ";
      if (TopInst) DOUT << "I: " << *TopInst << "\n";
      else DOUT << "BB: " << TopBB->getName()
                << "(" << Top->getDFSNumIn() << ")\n";

      assert(V1->getType() == V2->getType() &&
             "Can't make two values with different types equal.");

      if (V1 == V2) return true;

      if (isa<Constant>(V1) && isa<Constant>(V2))
        return false;

      unsigned n1 = VN.valueNumber(V1, Top), n2 = VN.valueNumber(V2, Top);

      if (n1 && n2) {
        if (n1 == n2) return true;
        if (IG.isRelatedBy(n1, n2, Top, NE)) return false;
      }

      if (n1) assert(V1 == VN.value(n1) && "Value isn't canonical.");
      if (n2) assert(V2 == VN.value(n2) && "Value isn't canonical.");

      assert(!VN.compare(V2, V1) && "Please order parameters to makeEqual.");

      assert(!isa<Constant>(V2) && "Tried to remove a constant.");

      SetVector<unsigned> Remove;
      if (n2) Remove.insert(n2);

      if (n1 && n2) {
        // Suppose we're being told that %x == %y, and %x <= %z and %y >= %z.
        // We can't just merge %x and %y because the relationship with %z would
        // be EQ and that's invalid. What we're doing is looking for any nodes
        // %z such that %x <= %z and %y >= %z, and vice versa.

        Node::iterator end = IG.node(n2)->end();

        // Find the intersection between N1 and N2 which is dominated by
        // Top. If we find %x where N1 <= %x <= N2 (or >=) then add %x to
        // Remove.
        for (Node::iterator I = IG.node(n1)->begin(), E = IG.node(n1)->end();
             I != E; ++I) {
          if (!(I->LV & EQ_BIT) || !Top->DominatedBy(I->Subtree)) continue;

          unsigned ILV_s = I->LV & (SLT_BIT|SGT_BIT);
          unsigned ILV_u = I->LV & (ULT_BIT|UGT_BIT);
          Node::iterator NI = IG.node(n2)->find(I->To, Top);
          if (NI != end) {
            LatticeVal NILV = reversePredicate(NI->LV);
            unsigned NILV_s = NILV & (SLT_BIT|SGT_BIT);
            unsigned NILV_u = NILV & (ULT_BIT|UGT_BIT);

            if ((ILV_s != (SLT_BIT|SGT_BIT) && ILV_s == NILV_s) ||
                (ILV_u != (ULT_BIT|UGT_BIT) && ILV_u == NILV_u))
              Remove.insert(I->To);
          }
        }

        // See if one of the nodes about to be removed is actually a better
        // canonical choice than n1.
        unsigned orig_n1 = n1;
        SetVector<unsigned>::iterator DontRemove = Remove.end();
        for (SetVector<unsigned>::iterator I = Remove.begin()+1 /* skip n2 */,
             E = Remove.end(); I != E; ++I) {
          unsigned n = *I;
          Value *V = VN.value(n);
          if (VN.compare(V, V1)) {
            V1 = V;
            n1 = n;
            DontRemove = I;
          }
        }
        if (DontRemove != Remove.end()) {
          unsigned n = *DontRemove;
          Remove.remove(n);
          Remove.insert(orig_n1);
        }
      }

      // We'd like to allow makeEqual on two values to perform a simple
      // substitution without creating nodes in the IG whenever possible.
      //
      // The first iteration through this loop operates on V2 before going
      // through the Remove list and operating on those too. If all of the
      // iterations performed simple replacements then we exit early.
      bool mergeIGNode = false;
      unsigned i = 0;
      for (Value *R = V2; i == 0 || i < Remove.size(); ++i) {
        if (i) R = VN.value(Remove[i]); // skip n2.

        // Try to replace the whole instruction. If we can, we're done.
        Instruction *I2 = dyn_cast<Instruction>(R);
        if (I2 && below(I2)) {
          std::vector<Instruction *> ToNotify;
          for (Value::use_iterator UI = R->use_begin(), UE = R->use_end();
               UI != UE;) {
            Use &TheUse = UI.getUse();
            ++UI;
            if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser()))
              ToNotify.push_back(I);
          }

          DOUT << "Simply removing " << *I2
               << ", replacing with " << *V1 << "\n";
          I2->replaceAllUsesWith(V1);
          // leave it dead; it'll get erased later.
          ++NumInstruction;
          modified = true;

          for (std::vector<Instruction *>::iterator II = ToNotify.begin(),
               IE = ToNotify.end(); II != IE; ++II) {
            opsToDef(*II);
          }

          continue;
        }

        // Otherwise, replace all dominated uses.
        for (Value::use_iterator UI = R->use_begin(), UE = R->use_end();
             UI != UE;) {
          Use &TheUse = UI.getUse();
          ++UI;
          if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
            if (below(I)) {
              TheUse.set(V1);
              modified = true;
              ++NumVarsReplaced;
              opsToDef(I);
            }
          }
        }

        // If that killed the instruction, stop here.
        if (I2 && isInstructionTriviallyDead(I2)) {
          DOUT << "Killed all uses of " << *I2
               << ", replacing with " << *V1 << "\n";
          continue;
        }

        // If we make it to here, then we will need to create a node for N1.
        // Otherwise, we can skip out early!
        mergeIGNode = true;
      }

      if (!isa<Constant>(V1)) {
        if (Remove.empty()) {
          VR.mergeInto(&V2, 1, VN.getOrInsertVN(V1, Top), Top, this);
        } else {
          std::vector<Value*> RemoveVals;
          RemoveVals.reserve(Remove.size());

          for (SetVector<unsigned>::iterator I = Remove.begin(),
               E = Remove.end(); I != E; ++I) {
            Value *V = VN.value(*I);
            if (!V->use_empty())
              RemoveVals.push_back(V);
          }
          VR.mergeInto(&RemoveVals[0], RemoveVals.size(), 
                       VN.getOrInsertVN(V1, Top), Top, this);
        }
      }

      if (mergeIGNode) {
        // Create N1.
        if (!n1) n1 = VN.getOrInsertVN(V1, Top);
        IG.node(n1); // Ensure that IG.Nodes won't get resized

        // Migrate relationships from removed nodes to N1.
        for (SetVector<unsigned>::iterator I = Remove.begin(), E = Remove.end();
             I != E; ++I) {
          unsigned n = *I;
          for (Node::iterator NI = IG.node(n)->begin(), NE = IG.node(n)->end();
               NI != NE; ++NI) {
            if (NI->Subtree->DominatedBy(Top)) {
              if (NI->To == n1) {
                assert((NI->LV & EQ_BIT) && "Node inequal to itself.");
                continue;
              }
              if (Remove.count(NI->To))
                continue;

              IG.node(NI->To)->update(n1, reversePredicate(NI->LV), Top);
              IG.node(n1)->update(NI->To, NI->LV, Top);
            }
          }
        }

        // Point V2 (and all items in Remove) to N1.
        if (!n2)
          VN.addEquality(n1, V2, Top);
        else {
          for (SetVector<unsigned>::iterator I = Remove.begin(),
               E = Remove.end(); I != E; ++I) {
            VN.addEquality(n1, VN.value(*I), Top);
          }
        }

        // If !Remove.empty() then V2 = Remove[0]->getValue().
        // Even when Remove is empty, we still want to process V2.
        i = 0;
        for (Value *R = V2; i == 0 || i < Remove.size(); ++i) {
          if (i) R = VN.value(Remove[i]); // skip n2.

          if (Instruction *I2 = dyn_cast<Instruction>(R)) {
            if (aboveOrBelow(I2))
            defToOps(I2);
          }
          for (Value::use_iterator UI = V2->use_begin(), UE = V2->use_end();
               UI != UE;) {
            Use &TheUse = UI.getUse();
            ++UI;
            if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
              if (aboveOrBelow(I))
                opsToDef(I);
            }
          }
        }
      }

      // re-opsToDef all dominated users of V1.
      if (Instruction *I = dyn_cast<Instruction>(V1)) {
        for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
             UI != UE;) {
          Use &TheUse = UI.getUse();
          ++UI;
          Value *V = TheUse.getUser();
          if (!V->use_empty()) {
            if (Instruction *Inst = dyn_cast<Instruction>(V)) {
              if (aboveOrBelow(Inst))
                opsToDef(Inst);
            }
          }
        }
      }

      return true;
    }

    /// cmpInstToLattice - converts an CmpInst::Predicate to lattice value
    /// Requires that the lattice value be valid; does not accept ICMP_EQ.
    static LatticeVal cmpInstToLattice(ICmpInst::Predicate Pred) {
      switch (Pred) {
        case ICmpInst::ICMP_EQ:
          assert(!"No matching lattice value.");
          return static_cast<LatticeVal>(EQ_BIT);
        default:
          assert(!"Invalid 'icmp' predicate.");
        case ICmpInst::ICMP_NE:
          return NE;
        case ICmpInst::ICMP_UGT:
          return UGT;
        case ICmpInst::ICMP_UGE:
          return UGE;
        case ICmpInst::ICMP_ULT:
          return ULT;
        case ICmpInst::ICMP_ULE:
          return ULE;
        case ICmpInst::ICMP_SGT:
          return SGT;
        case ICmpInst::ICMP_SGE:
          return SGE;
        case ICmpInst::ICMP_SLT:
          return SLT;
        case ICmpInst::ICMP_SLE:
          return SLE;
      }
    }

  public:
    VRPSolver(ValueNumbering &VN, InequalityGraph &IG, UnreachableBlocks &UB,
              ValueRanges &VR, DomTreeDFS *DTDFS, bool &modified,
              BasicBlock *TopBB)
      : VN(VN),
        IG(IG),
        UB(UB),
        VR(VR),
        DTDFS(DTDFS),
        Top(DTDFS->getNodeForBlock(TopBB)),
        TopBB(TopBB),
        TopInst(NULL),
        modified(modified)
    {
      assert(Top && "VRPSolver created for unreachable basic block.");
    }

    VRPSolver(ValueNumbering &VN, InequalityGraph &IG, UnreachableBlocks &UB,
              ValueRanges &VR, DomTreeDFS *DTDFS, bool &modified,
              Instruction *TopInst)
      : VN(VN),
        IG(IG),
        UB(UB),
        VR(VR),
        DTDFS(DTDFS),
        Top(DTDFS->getNodeForBlock(TopInst->getParent())),
        TopBB(TopInst->getParent()),
        TopInst(TopInst),
        modified(modified)
    {
      assert(Top && "VRPSolver created for unreachable basic block.");
      assert(Top->getBlock() == TopInst->getParent() && "Context mismatch.");
    }

    bool isRelatedBy(Value *V1, Value *V2, ICmpInst::Predicate Pred) const {
      if (Constant *C1 = dyn_cast<Constant>(V1))
        if (Constant *C2 = dyn_cast<Constant>(V2))
          return ConstantExpr::getCompare(Pred, C1, C2) ==
                 ConstantInt::getTrue();

      unsigned n1 = VN.valueNumber(V1, Top);
      unsigned n2 = VN.valueNumber(V2, Top);

      if (n1 && n2) {
        if (n1 == n2) return Pred == ICmpInst::ICMP_EQ ||
                             Pred == ICmpInst::ICMP_ULE ||
                             Pred == ICmpInst::ICMP_UGE ||
                             Pred == ICmpInst::ICMP_SLE ||
                             Pred == ICmpInst::ICMP_SGE;
        if (Pred == ICmpInst::ICMP_EQ) return false;
        if (IG.isRelatedBy(n1, n2, Top, cmpInstToLattice(Pred))) return true;
        if (VR.isRelatedBy(n1, n2, Top, cmpInstToLattice(Pred))) return true;
      }

      if ((n1 && !n2 && isa<Constant>(V2)) ||
          (n2 && !n1 && isa<Constant>(V1))) {
        ConstantRange CR1 = n1 ? VR.range(n1, Top) : VR.range(V1);
        ConstantRange CR2 = n2 ? VR.range(n2, Top) : VR.range(V2);

        if (Pred == ICmpInst::ICMP_EQ)
          return CR1.isSingleElement() &&
                 CR1.getSingleElement() == CR2.getSingleElement();

        return VR.isRelatedBy(CR1, CR2, cmpInstToLattice(Pred));
      }
      if (Pred == ICmpInst::ICMP_EQ) return V1 == V2;
      return false;
    }

    /// add - adds a new property to the work queue
    void add(Value *V1, Value *V2, ICmpInst::Predicate Pred,
             Instruction *I = NULL) {
      DOUT << "adding " << *V1 << " " << Pred << " " << *V2;
      if (I) DOUT << " context: " << *I;
      else DOUT << " default context (" << Top->getDFSNumIn() << ")";
      DOUT << "\n";

      assert(V1->getType() == V2->getType() &&
             "Can't relate two values with different types.");

      WorkList.push_back(Operation());
      Operation &O = WorkList.back();
      O.LHS = V1, O.RHS = V2, O.Op = Pred, O.ContextInst = I;
      O.ContextBB = I ? I->getParent() : TopBB;
    }

    /// defToOps - Given an instruction definition that we've learned something
    /// new about, find any new relationships between its operands.
    void defToOps(Instruction *I) {
      Instruction *NewContext = below(I) ? I : TopInst;
      Value *Canonical = VN.canonicalize(I, Top);

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        const Type *Ty = BO->getType();
        assert(!Ty->isFPOrFPVector() && "Float in work queue!");

        Value *Op0 = VN.canonicalize(BO->getOperand(0), Top);
        Value *Op1 = VN.canonicalize(BO->getOperand(1), Top);

        // TODO: "and i32 -1, %x" EQ %y then %x EQ %y.

        switch (BO->getOpcode()) {
          case Instruction::And: {
            // "and i32 %a, %b" EQ -1 then %a EQ -1 and %b EQ -1
            ConstantInt *CI = ConstantInt::getAllOnesValue(Ty);
            if (Canonical == CI) {
              add(CI, Op0, ICmpInst::ICMP_EQ, NewContext);
              add(CI, Op1, ICmpInst::ICMP_EQ, NewContext);
            }
          } break;
          case Instruction::Or: {
            // "or i32 %a, %b" EQ 0 then %a EQ 0 and %b EQ 0
            Constant *Zero = Constant::getNullValue(Ty);
            if (Canonical == Zero) {
              add(Zero, Op0, ICmpInst::ICMP_EQ, NewContext);
              add(Zero, Op1, ICmpInst::ICMP_EQ, NewContext);
            }
          } break;
          case Instruction::Xor: {
            // "xor i32 %c, %a" EQ %b then %a EQ %c ^ %b
            // "xor i32 %c, %a" EQ %c then %a EQ 0
            // "xor i32 %c, %a" NE %c then %a NE 0
            // Repeat the above, with order of operands reversed.
            Value *LHS = Op0;
            Value *RHS = Op1;
            if (!isa<Constant>(LHS)) std::swap(LHS, RHS);

            if (ConstantInt *CI = dyn_cast<ConstantInt>(Canonical)) {
              if (ConstantInt *Arg = dyn_cast<ConstantInt>(LHS)) {
                add(RHS, ConstantInt::get(CI->getValue() ^ Arg->getValue()),
                    ICmpInst::ICMP_EQ, NewContext);
              }
            }
            if (Canonical == LHS) {
              if (isa<ConstantInt>(Canonical))
                add(RHS, Constant::getNullValue(Ty), ICmpInst::ICMP_EQ,
                    NewContext);
            } else if (isRelatedBy(LHS, Canonical, ICmpInst::ICMP_NE)) {
              add(RHS, Constant::getNullValue(Ty), ICmpInst::ICMP_NE,
                  NewContext);
            }
          } break;
          default:
            break;
        }
      } else if (ICmpInst *IC = dyn_cast<ICmpInst>(I)) {
        // "icmp ult i32 %a, %y" EQ true then %a u< y
        // etc.

        if (Canonical == ConstantInt::getTrue()) {
          add(IC->getOperand(0), IC->getOperand(1), IC->getPredicate(),
              NewContext);
        } else if (Canonical == ConstantInt::getFalse()) {
          add(IC->getOperand(0), IC->getOperand(1),
              ICmpInst::getInversePredicate(IC->getPredicate()), NewContext);
        }
      } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        if (I->getType()->isFPOrFPVector()) return;

        // Given: "%a = select i1 %x, i32 %b, i32 %c"
        // %a EQ %b and %b NE %c then %x EQ true
        // %a EQ %c and %b NE %c then %x EQ false

        Value *True  = SI->getTrueValue();
        Value *False = SI->getFalseValue();
        if (isRelatedBy(True, False, ICmpInst::ICMP_NE)) {
          if (Canonical == VN.canonicalize(True, Top) ||
              isRelatedBy(Canonical, False, ICmpInst::ICMP_NE))
            add(SI->getCondition(), ConstantInt::getTrue(),
                ICmpInst::ICMP_EQ, NewContext);
          else if (Canonical == VN.canonicalize(False, Top) ||
                   isRelatedBy(Canonical, True, ICmpInst::ICMP_NE))
            add(SI->getCondition(), ConstantInt::getFalse(),
                ICmpInst::ICMP_EQ, NewContext);
        }
      } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
        for (GetElementPtrInst::op_iterator OI = GEPI->idx_begin(),
             OE = GEPI->idx_end(); OI != OE; ++OI) {
          ConstantInt *Op = dyn_cast<ConstantInt>(VN.canonicalize(*OI, Top));
          if (!Op || !Op->isZero()) return;
        }
        // TODO: The GEPI indices are all zero. Copy from definition to operand,
        // jumping the type plane as needed.
        if (isRelatedBy(GEPI, Constant::getNullValue(GEPI->getType()),
                        ICmpInst::ICMP_NE)) {
          Value *Ptr = GEPI->getPointerOperand();
          add(Ptr, Constant::getNullValue(Ptr->getType()), ICmpInst::ICMP_NE,
              NewContext);
        }
      } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
        const Type *SrcTy = CI->getSrcTy();

        unsigned ci = VN.getOrInsertVN(CI, Top);
        uint32_t W = VR.typeToWidth(SrcTy);
        if (!W) return;
        ConstantRange CR = VR.range(ci, Top);

        if (CR.isFullSet()) return;

        switch (CI->getOpcode()) {
          default: break;
          case Instruction::ZExt:
          case Instruction::SExt:
            VR.applyRange(VN.getOrInsertVN(CI->getOperand(0), Top),
                          CR.truncate(W), Top, this);
            break;
          case Instruction::BitCast:
            VR.applyRange(VN.getOrInsertVN(CI->getOperand(0), Top),
                          CR, Top, this);
            break;
        }
      }
    }

    /// opsToDef - A new relationship was discovered involving one of this
    /// instruction's operands. Find any new relationship involving the
    /// definition, or another operand.
    void opsToDef(Instruction *I) {
      Instruction *NewContext = below(I) ? I : TopInst;

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        Value *Op0 = VN.canonicalize(BO->getOperand(0), Top);
        Value *Op1 = VN.canonicalize(BO->getOperand(1), Top);

        if (ConstantInt *CI0 = dyn_cast<ConstantInt>(Op0))
          if (ConstantInt *CI1 = dyn_cast<ConstantInt>(Op1)) {
            add(BO, ConstantExpr::get(BO->getOpcode(), CI0, CI1),
                ICmpInst::ICMP_EQ, NewContext);
            return;
          }

        // "%y = and i1 true, %x" then %x EQ %y
        // "%y = or i1 false, %x" then %x EQ %y
        // "%x = add i32 %y, 0" then %x EQ %y
        // "%x = mul i32 %y, 0" then %x EQ 0

        Instruction::BinaryOps Opcode = BO->getOpcode();
        const Type *Ty = BO->getType();
        assert(!Ty->isFPOrFPVector() && "Float in work queue!");

        Constant *Zero = Constant::getNullValue(Ty);
        Constant *One = ConstantInt::get(Ty, 1);
        ConstantInt *AllOnes = ConstantInt::getAllOnesValue(Ty);

        switch (Opcode) {
          default: break;
          case Instruction::LShr:
          case Instruction::AShr:
          case Instruction::Shl:
            if (Op1 == Zero) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
          case Instruction::Sub:
            if (Op1 == Zero) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            if (ConstantInt *CI0 = dyn_cast<ConstantInt>(Op0)) {
              unsigned n_ci0 = VN.getOrInsertVN(Op1, Top);
              ConstantRange CR = VR.range(n_ci0, Top);
              if (!CR.isFullSet()) {
                CR.subtract(CI0->getValue());
                unsigned n_bo = VN.getOrInsertVN(BO, Top);
                VR.applyRange(n_bo, CR, Top, this);
                return;
              }
            }
            if (ConstantInt *CI1 = dyn_cast<ConstantInt>(Op1)) {
              unsigned n_ci1 = VN.getOrInsertVN(Op0, Top);
              ConstantRange CR = VR.range(n_ci1, Top);
              if (!CR.isFullSet()) {
                CR.subtract(CI1->getValue());
                unsigned n_bo = VN.getOrInsertVN(BO, Top);
                VR.applyRange(n_bo, CR, Top, this);
                return;
              }
            }
            break;
          case Instruction::Or:
            if (Op0 == AllOnes || Op1 == AllOnes) {
              add(BO, AllOnes, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            if (Op0 == Zero) {
              add(BO, Op1, ICmpInst::ICMP_EQ, NewContext);
              return;
            } else if (Op1 == Zero) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
          case Instruction::Add:
            if (ConstantInt *CI0 = dyn_cast<ConstantInt>(Op0)) {
              unsigned n_ci0 = VN.getOrInsertVN(Op1, Top);
              ConstantRange CR = VR.range(n_ci0, Top);
              if (!CR.isFullSet()) {
                CR.subtract(-CI0->getValue());
                unsigned n_bo = VN.getOrInsertVN(BO, Top);
                VR.applyRange(n_bo, CR, Top, this);
                return;
              }
            }
            if (ConstantInt *CI1 = dyn_cast<ConstantInt>(Op1)) {
              unsigned n_ci1 = VN.getOrInsertVN(Op0, Top);
              ConstantRange CR = VR.range(n_ci1, Top);
              if (!CR.isFullSet()) {
                CR.subtract(-CI1->getValue());
                unsigned n_bo = VN.getOrInsertVN(BO, Top);
                VR.applyRange(n_bo, CR, Top, this);
                return;
              }
            }
            // fall-through
          case Instruction::Xor:
            if (Op0 == Zero) {
              add(BO, Op1, ICmpInst::ICMP_EQ, NewContext);
              return;
            } else if (Op1 == Zero) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
          case Instruction::And:
            if (Op0 == AllOnes) {
              add(BO, Op1, ICmpInst::ICMP_EQ, NewContext);
              return;
            } else if (Op1 == AllOnes) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            if (Op0 == Zero || Op1 == Zero) {
              add(BO, Zero, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
          case Instruction::Mul:
            if (Op0 == Zero || Op1 == Zero) {
              add(BO, Zero, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            if (Op0 == One) {
              add(BO, Op1, ICmpInst::ICMP_EQ, NewContext);
              return;
            } else if (Op1 == One) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
        }

        // "%x = add i32 %y, %z" and %x EQ %y then %z EQ 0
        // "%x = add i32 %y, %z" and %x EQ %z then %y EQ 0
        // "%x = shl i32 %y, %z" and %x EQ %y and %y NE 0 then %z EQ 0
        // "%x = udiv i32 %y, %z" and %x EQ %y and %y NE 0 then %z EQ 1

        Value *Known = Op0, *Unknown = Op1,
              *TheBO = VN.canonicalize(BO, Top);
        if (Known != TheBO) std::swap(Known, Unknown);
        if (Known == TheBO) {
          switch (Opcode) {
            default: break;
            case Instruction::LShr:
            case Instruction::AShr:
            case Instruction::Shl:
              if (!isRelatedBy(Known, Zero, ICmpInst::ICMP_NE)) break;
              // otherwise, fall-through.
            case Instruction::Sub:
              if (Unknown == Op0) break;
              // otherwise, fall-through.
            case Instruction::Xor:
            case Instruction::Add:
              add(Unknown, Zero, ICmpInst::ICMP_EQ, NewContext);
              break;
            case Instruction::UDiv:
            case Instruction::SDiv:
              if (Unknown == Op1) break;
              if (isRelatedBy(Known, Zero, ICmpInst::ICMP_NE))
                add(Unknown, One, ICmpInst::ICMP_EQ, NewContext);
              break;
          }
        }

        // TODO: "%a = add i32 %b, 1" and %b > %z then %a >= %z.

      } else if (ICmpInst *IC = dyn_cast<ICmpInst>(I)) {
        // "%a = icmp ult i32 %b, %c" and %b u<  %c then %a EQ true
        // "%a = icmp ult i32 %b, %c" and %b u>= %c then %a EQ false
        // etc.

        Value *Op0 = VN.canonicalize(IC->getOperand(0), Top);
        Value *Op1 = VN.canonicalize(IC->getOperand(1), Top);

        ICmpInst::Predicate Pred = IC->getPredicate();
        if (isRelatedBy(Op0, Op1, Pred))
          add(IC, ConstantInt::getTrue(), ICmpInst::ICMP_EQ, NewContext);
        else if (isRelatedBy(Op0, Op1, ICmpInst::getInversePredicate(Pred)))
          add(IC, ConstantInt::getFalse(), ICmpInst::ICMP_EQ, NewContext);

      } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        if (I->getType()->isFPOrFPVector()) return;

        // Given: "%a = select i1 %x, i32 %b, i32 %c"
        // %x EQ true  then %a EQ %b
        // %x EQ false then %a EQ %c
        // %b EQ %c then %a EQ %b

        Value *Canonical = VN.canonicalize(SI->getCondition(), Top);
        if (Canonical == ConstantInt::getTrue()) {
          add(SI, SI->getTrueValue(), ICmpInst::ICMP_EQ, NewContext);
        } else if (Canonical == ConstantInt::getFalse()) {
          add(SI, SI->getFalseValue(), ICmpInst::ICMP_EQ, NewContext);
        } else if (VN.canonicalize(SI->getTrueValue(), Top) ==
                   VN.canonicalize(SI->getFalseValue(), Top)) {
          add(SI, SI->getTrueValue(), ICmpInst::ICMP_EQ, NewContext);
        }
      } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
        const Type *DestTy = CI->getDestTy();
        if (DestTy->isFPOrFPVector()) return;

        Value *Op = VN.canonicalize(CI->getOperand(0), Top);
        Instruction::CastOps Opcode = CI->getOpcode();

        if (Constant *C = dyn_cast<Constant>(Op)) {
          add(CI, ConstantExpr::getCast(Opcode, C, DestTy),
              ICmpInst::ICMP_EQ, NewContext);
        }

        uint32_t W = VR.typeToWidth(DestTy);
        unsigned ci = VN.getOrInsertVN(CI, Top);
        ConstantRange CR = VR.range(VN.getOrInsertVN(Op, Top), Top);

        if (!CR.isFullSet()) {
          switch (Opcode) {
            default: break;
            case Instruction::ZExt:
              VR.applyRange(ci, CR.zeroExtend(W), Top, this);
              break;
            case Instruction::SExt:
              VR.applyRange(ci, CR.signExtend(W), Top, this);
              break;
            case Instruction::Trunc: {
              ConstantRange Result = CR.truncate(W);
              if (!Result.isFullSet())
                VR.applyRange(ci, Result, Top, this);
            } break;
            case Instruction::BitCast:
              VR.applyRange(ci, CR, Top, this);
              break;
            // TODO: other casts?
          }
        }
      } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
        for (GetElementPtrInst::op_iterator OI = GEPI->idx_begin(),
             OE = GEPI->idx_end(); OI != OE; ++OI) {
          ConstantInt *Op = dyn_cast<ConstantInt>(VN.canonicalize(*OI, Top));
          if (!Op || !Op->isZero()) return;
        }
        // TODO: The GEPI indices are all zero. Copy from operand to definition,
        // jumping the type plane as needed.
        Value *Ptr = GEPI->getPointerOperand();
        if (isRelatedBy(Ptr, Constant::getNullValue(Ptr->getType()),
                        ICmpInst::ICMP_NE)) {
          add(GEPI, Constant::getNullValue(GEPI->getType()), ICmpInst::ICMP_NE,
              NewContext);
        }
      }
    }

    /// solve - process the work queue
    void solve() {
      //DOUT << "WorkList entry, size: " << WorkList.size() << "\n";
      while (!WorkList.empty()) {
        //DOUT << "WorkList size: " << WorkList.size() << "\n";

        Operation &O = WorkList.front();
        TopInst = O.ContextInst;
        TopBB = O.ContextBB;
        Top = DTDFS->getNodeForBlock(TopBB); // XXX move this into Context

        O.LHS = VN.canonicalize(O.LHS, Top);
        O.RHS = VN.canonicalize(O.RHS, Top);

        assert(O.LHS == VN.canonicalize(O.LHS, Top) && "Canonicalize isn't.");
        assert(O.RHS == VN.canonicalize(O.RHS, Top) && "Canonicalize isn't.");

        DOUT << "solving " << *O.LHS << " " << O.Op << " " << *O.RHS;
        if (O.ContextInst) DOUT << " context inst: " << *O.ContextInst;
        else DOUT << " context block: " << O.ContextBB->getName();
        DOUT << "\n";

        DEBUG(VN.dump());
        DEBUG(IG.dump());
        DEBUG(VR.dump());

        // If they're both Constant, skip it. Check for contradiction and mark
        // the BB as unreachable if so.
        if (Constant *CI_L = dyn_cast<Constant>(O.LHS)) {
          if (Constant *CI_R = dyn_cast<Constant>(O.RHS)) {
            if (ConstantExpr::getCompare(O.Op, CI_L, CI_R) ==
                ConstantInt::getFalse())
              UB.mark(TopBB);

            WorkList.pop_front();
            continue;
          }
        }

        if (VN.compare(O.LHS, O.RHS)) {
          std::swap(O.LHS, O.RHS);
          O.Op = ICmpInst::getSwappedPredicate(O.Op);
        }

        if (O.Op == ICmpInst::ICMP_EQ) {
          if (!makeEqual(O.RHS, O.LHS))
            UB.mark(TopBB);
        } else {
          LatticeVal LV = cmpInstToLattice(O.Op);

          if ((LV & EQ_BIT) &&
              isRelatedBy(O.LHS, O.RHS, ICmpInst::getSwappedPredicate(O.Op))) {
            if (!makeEqual(O.RHS, O.LHS))
              UB.mark(TopBB);
          } else {
            if (isRelatedBy(O.LHS, O.RHS, ICmpInst::getInversePredicate(O.Op))){
              UB.mark(TopBB);
              WorkList.pop_front();
              continue;
            }

            unsigned n1 = VN.getOrInsertVN(O.LHS, Top);
            unsigned n2 = VN.getOrInsertVN(O.RHS, Top);

            if (n1 == n2) {
              if (O.Op != ICmpInst::ICMP_UGE && O.Op != ICmpInst::ICMP_ULE &&
                  O.Op != ICmpInst::ICMP_SGE && O.Op != ICmpInst::ICMP_SLE)
                UB.mark(TopBB);

              WorkList.pop_front();
              continue;
            }

            if (VR.isRelatedBy(n1, n2, Top, LV) ||
                IG.isRelatedBy(n1, n2, Top, LV)) {
              WorkList.pop_front();
              continue;
            }

            VR.addInequality(n1, n2, Top, LV, this);
            if ((!isa<ConstantInt>(O.RHS) && !isa<ConstantInt>(O.LHS)) ||
                LV == NE)
              IG.addInequality(n1, n2, Top, LV);

            if (Instruction *I1 = dyn_cast<Instruction>(O.LHS)) {
              if (aboveOrBelow(I1))
                defToOps(I1);
            }
            if (isa<Instruction>(O.LHS) || isa<Argument>(O.LHS)) {
              for (Value::use_iterator UI = O.LHS->use_begin(),
                   UE = O.LHS->use_end(); UI != UE;) {
                Use &TheUse = UI.getUse();
                ++UI;
                if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
                  if (aboveOrBelow(I))
                    opsToDef(I);
                }
              }
            }
            if (Instruction *I2 = dyn_cast<Instruction>(O.RHS)) {
              if (aboveOrBelow(I2))
              defToOps(I2);
            }
            if (isa<Instruction>(O.RHS) || isa<Argument>(O.RHS)) {
              for (Value::use_iterator UI = O.RHS->use_begin(),
                   UE = O.RHS->use_end(); UI != UE;) {
                Use &TheUse = UI.getUse();
                ++UI;
                if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
                  if (aboveOrBelow(I))
                    opsToDef(I);
                }
              }
            }
          }
        }
        WorkList.pop_front();
      }
    }
  };

  void ValueRanges::addToWorklist(Value *V, Constant *C,
                                  ICmpInst::Predicate Pred, VRPSolver *VRP) {
    VRP->add(V, C, Pred, VRP->TopInst);
  }

  void ValueRanges::markBlock(VRPSolver *VRP) {
    VRP->UB.mark(VRP->TopBB);
  }

  /// PredicateSimplifier - This class is a simplifier that replaces
  /// one equivalent variable with another. It also tracks what
  /// can't be equal and will solve setcc instructions when possible.
  /// @brief Root of the predicate simplifier optimization.
  class VISIBILITY_HIDDEN PredicateSimplifier : public FunctionPass {
    DomTreeDFS *DTDFS;
    bool modified;
    ValueNumbering *VN;
    InequalityGraph *IG;
    UnreachableBlocks UB;
    ValueRanges *VR;

    std::vector<DomTreeDFS::Node *> WorkList;

  public:
    static char ID; // Pass identification, replacement for typeid
    PredicateSimplifier() : FunctionPass(&ID) {}

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequired<DominatorTree>();
      AU.addRequired<TargetData>();
      AU.addPreserved<TargetData>();
    }

  private:
    /// Forwards - Adds new properties to VRPSolver and uses them to
    /// simplify instructions. Because new properties sometimes apply to
    /// a transition from one BasicBlock to another, this will use the
    /// PredicateSimplifier::proceedToSuccessor(s) interface to enter the
    /// basic block.
    /// @brief Performs abstract execution of the program.
    class VISIBILITY_HIDDEN Forwards : public InstVisitor<Forwards> {
      friend class InstVisitor<Forwards>;
      PredicateSimplifier *PS;
      DomTreeDFS::Node *DTNode;

    public:
      ValueNumbering &VN;
      InequalityGraph &IG;
      UnreachableBlocks &UB;
      ValueRanges &VR;

      Forwards(PredicateSimplifier *PS, DomTreeDFS::Node *DTNode)
        : PS(PS), DTNode(DTNode), VN(*PS->VN), IG(*PS->IG), UB(PS->UB),
          VR(*PS->VR) {}

      void visitTerminatorInst(TerminatorInst &TI);
      void visitBranchInst(BranchInst &BI);
      void visitSwitchInst(SwitchInst &SI);

      void visitAllocaInst(AllocaInst &AI);
      void visitLoadInst(LoadInst &LI);
      void visitStoreInst(StoreInst &SI);

      void visitSExtInst(SExtInst &SI);
      void visitZExtInst(ZExtInst &ZI);

      void visitBinaryOperator(BinaryOperator &BO);
      void visitICmpInst(ICmpInst &IC);
    };
  
    // Used by terminator instructions to proceed from the current basic
    // block to the next. Verifies that "current" dominates "next",
    // then calls visitBasicBlock.
    void proceedToSuccessors(DomTreeDFS::Node *Current) {
      for (DomTreeDFS::Node::iterator I = Current->begin(),
           E = Current->end(); I != E; ++I) {
        WorkList.push_back(*I);
      }
    }

    void proceedToSuccessor(DomTreeDFS::Node *Next) {
      WorkList.push_back(Next);
    }

    // Visits each instruction in the basic block.
    void visitBasicBlock(DomTreeDFS::Node *Node) {
      BasicBlock *BB = Node->getBlock();
      DOUT << "Entering Basic Block: " << BB->getName()
           << " (" << Node->getDFSNumIn() << ")\n";
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
        visitInstruction(I++, Node);
      }
    }

    // Tries to simplify each Instruction and add new properties.
    void visitInstruction(Instruction *I, DomTreeDFS::Node *DT) {
      DOUT << "Considering instruction " << *I << "\n";
      DEBUG(VN->dump());
      DEBUG(IG->dump());
      DEBUG(VR->dump());

      // Sometimes instructions are killed in earlier analysis.
      if (isInstructionTriviallyDead(I)) {
        ++NumSimple;
        modified = true;
        if (unsigned n = VN->valueNumber(I, DTDFS->getRootNode()))
          if (VN->value(n) == I) IG->remove(n);
        VN->remove(I);
        I->eraseFromParent();
        return;
      }

#ifndef NDEBUG
      // Try to replace the whole instruction.
      Value *V = VN->canonicalize(I, DT);
      assert(V == I && "Late instruction canonicalization.");
      if (V != I) {
        modified = true;
        ++NumInstruction;
        DOUT << "Removing " << *I << ", replacing with " << *V << "\n";
        if (unsigned n = VN->valueNumber(I, DTDFS->getRootNode()))
          if (VN->value(n) == I) IG->remove(n);
        VN->remove(I);
        I->replaceAllUsesWith(V);
        I->eraseFromParent();
        return;
      }

      // Try to substitute operands.
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        Value *Oper = I->getOperand(i);
        Value *V = VN->canonicalize(Oper, DT);
        assert(V == Oper && "Late operand canonicalization.");
        if (V != Oper) {
          modified = true;
          ++NumVarsReplaced;
          DOUT << "Resolving " << *I;
          I->setOperand(i, V);
          DOUT << " into " << *I;
        }
      }
#endif

      std::string name = I->getParent()->getName();
      DOUT << "push (%" << name << ")\n";
      Forwards visit(this, DT);
      visit.visit(*I);
      DOUT << "pop (%" << name << ")\n";
    }
  };

  bool PredicateSimplifier::runOnFunction(Function &F) {
    DominatorTree *DT = &getAnalysis<DominatorTree>();
    DTDFS = new DomTreeDFS(DT);
    TargetData *TD = &getAnalysis<TargetData>();

    DOUT << "Entering Function: " << F.getName() << "\n";

    modified = false;
    DomTreeDFS::Node *Root = DTDFS->getRootNode();
    VN = new ValueNumbering(DTDFS);
    IG = new InequalityGraph(*VN, Root);
    VR = new ValueRanges(*VN, TD);
    WorkList.push_back(Root);

    do {
      DomTreeDFS::Node *DTNode = WorkList.back();
      WorkList.pop_back();
      if (!UB.isDead(DTNode->getBlock())) visitBasicBlock(DTNode);
    } while (!WorkList.empty());

    delete DTDFS;
    delete VR;
    delete IG;
    delete VN;

    modified |= UB.kill();

    return modified;
  }

  void PredicateSimplifier::Forwards::visitTerminatorInst(TerminatorInst &TI) {
    PS->proceedToSuccessors(DTNode);
  }

  void PredicateSimplifier::Forwards::visitBranchInst(BranchInst &BI) {
    if (BI.isUnconditional()) {
      PS->proceedToSuccessors(DTNode);
      return;
    }

    Value *Condition = BI.getCondition();
    BasicBlock *TrueDest  = BI.getSuccessor(0);
    BasicBlock *FalseDest = BI.getSuccessor(1);

    if (isa<Constant>(Condition) || TrueDest == FalseDest) {
      PS->proceedToSuccessors(DTNode);
      return;
    }

    for (DomTreeDFS::Node::iterator I = DTNode->begin(), E = DTNode->end();
         I != E; ++I) {
      BasicBlock *Dest = (*I)->getBlock();
      DOUT << "Branch thinking about %" << Dest->getName()
           << "(" << PS->DTDFS->getNodeForBlock(Dest)->getDFSNumIn() << ")\n";

      if (Dest == TrueDest) {
        DOUT << "(" << DTNode->getBlock()->getName() << ") true set:\n";
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, Dest);
        VRP.add(ConstantInt::getTrue(), Condition, ICmpInst::ICMP_EQ);
        VRP.solve();
        DEBUG(VN.dump());
        DEBUG(IG.dump());
        DEBUG(VR.dump());
      } else if (Dest == FalseDest) {
        DOUT << "(" << DTNode->getBlock()->getName() << ") false set:\n";
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, Dest);
        VRP.add(ConstantInt::getFalse(), Condition, ICmpInst::ICMP_EQ);
        VRP.solve();
        DEBUG(VN.dump());
        DEBUG(IG.dump());
        DEBUG(VR.dump());
      }

      PS->proceedToSuccessor(*I);
    }
  }

  void PredicateSimplifier::Forwards::visitSwitchInst(SwitchInst &SI) {
    Value *Condition = SI.getCondition();

    // Set the EQProperty in each of the cases BBs, and the NEProperties
    // in the default BB.

    for (DomTreeDFS::Node::iterator I = DTNode->begin(), E = DTNode->end();
         I != E; ++I) {
      BasicBlock *BB = (*I)->getBlock();
      DOUT << "Switch thinking about BB %" << BB->getName()
           << "(" << PS->DTDFS->getNodeForBlock(BB)->getDFSNumIn() << ")\n";

      VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, BB);
      if (BB == SI.getDefaultDest()) {
        for (unsigned i = 1, e = SI.getNumCases(); i < e; ++i)
          if (SI.getSuccessor(i) != BB)
            VRP.add(Condition, SI.getCaseValue(i), ICmpInst::ICMP_NE);
        VRP.solve();
      } else if (ConstantInt *CI = SI.findCaseDest(BB)) {
        VRP.add(Condition, CI, ICmpInst::ICMP_EQ);
        VRP.solve();
      }
      PS->proceedToSuccessor(*I);
    }
  }

  void PredicateSimplifier::Forwards::visitAllocaInst(AllocaInst &AI) {
    VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &AI);
    VRP.add(Constant::getNullValue(AI.getType()), &AI, ICmpInst::ICMP_NE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitLoadInst(LoadInst &LI) {
    Value *Ptr = LI.getPointerOperand();
    // avoid "load i8* null" -> null NE null.
    if (isa<Constant>(Ptr)) return;

    VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &LI);
    VRP.add(Constant::getNullValue(Ptr->getType()), Ptr, ICmpInst::ICMP_NE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitStoreInst(StoreInst &SI) {
    Value *Ptr = SI.getPointerOperand();
    if (isa<Constant>(Ptr)) return;

    VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &SI);
    VRP.add(Constant::getNullValue(Ptr->getType()), Ptr, ICmpInst::ICMP_NE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitSExtInst(SExtInst &SI) {
    VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &SI);
    uint32_t SrcBitWidth = cast<IntegerType>(SI.getSrcTy())->getBitWidth();
    uint32_t DstBitWidth = cast<IntegerType>(SI.getDestTy())->getBitWidth();
    APInt Min(APInt::getHighBitsSet(DstBitWidth, DstBitWidth-SrcBitWidth+1));
    APInt Max(APInt::getLowBitsSet(DstBitWidth, SrcBitWidth-1));
    VRP.add(ConstantInt::get(Min), &SI, ICmpInst::ICMP_SLE);
    VRP.add(ConstantInt::get(Max), &SI, ICmpInst::ICMP_SGE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitZExtInst(ZExtInst &ZI) {
    VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &ZI);
    uint32_t SrcBitWidth = cast<IntegerType>(ZI.getSrcTy())->getBitWidth();
    uint32_t DstBitWidth = cast<IntegerType>(ZI.getDestTy())->getBitWidth();
    APInt Max(APInt::getLowBitsSet(DstBitWidth, SrcBitWidth));
    VRP.add(ConstantInt::get(Max), &ZI, ICmpInst::ICMP_UGE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitBinaryOperator(BinaryOperator &BO) {
    Instruction::BinaryOps ops = BO.getOpcode();

    switch (ops) {
    default: break;
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::UDiv:
      case Instruction::SDiv: {
        Value *Divisor = BO.getOperand(1);
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(Constant::getNullValue(Divisor->getType()), Divisor,
                ICmpInst::ICMP_NE);
        VRP.solve();
        break;
      }
    }

    switch (ops) {
      default: break;
      case Instruction::Shl: {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_UGE);
        VRP.solve();
      } break;
      case Instruction::AShr: {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_SLE);
        VRP.solve();
      } break;
      case Instruction::LShr:
      case Instruction::UDiv: {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_ULE);
        VRP.solve();
      } break;
      case Instruction::URem: {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(1), ICmpInst::ICMP_ULE);
        VRP.solve();
      } break;
      case Instruction::And: {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_ULE);
        VRP.add(&BO, BO.getOperand(1), ICmpInst::ICMP_ULE);
        VRP.solve();
      } break;
      case Instruction::Or: {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_UGE);
        VRP.add(&BO, BO.getOperand(1), ICmpInst::ICMP_UGE);
        VRP.solve();
      } break;
    }
  }

  void PredicateSimplifier::Forwards::visitICmpInst(ICmpInst &IC) {
    // If possible, squeeze the ICmp predicate into something simpler.
    // Eg., if x = [0, 4) and we're being asked icmp uge %x, 3 then change
    // the predicate to eq.

    // XXX: once we do full PHI handling, modifying the instruction in the
    // Forwards visitor will cause missed optimizations.

    ICmpInst::Predicate Pred = IC.getPredicate();

    switch (Pred) {
      default: break;
      case ICmpInst::ICMP_ULE: Pred = ICmpInst::ICMP_ULT; break;
      case ICmpInst::ICMP_UGE: Pred = ICmpInst::ICMP_UGT; break;
      case ICmpInst::ICMP_SLE: Pred = ICmpInst::ICMP_SLT; break;
      case ICmpInst::ICMP_SGE: Pred = ICmpInst::ICMP_SGT; break;
    }
    if (Pred != IC.getPredicate()) {
      VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &IC);
      if (VRP.isRelatedBy(IC.getOperand(1), IC.getOperand(0),
                          ICmpInst::ICMP_NE)) {
        ++NumSnuggle;
        PS->modified = true;
        IC.setPredicate(Pred);
      }
    }

    Pred = IC.getPredicate();

    if (ConstantInt *Op1 = dyn_cast<ConstantInt>(IC.getOperand(1))) {
      ConstantInt *NextVal = 0;
      switch (Pred) {
        default: break;
        case ICmpInst::ICMP_SLT:
        case ICmpInst::ICMP_ULT:
          if (Op1->getValue() != 0)
            NextVal = ConstantInt::get(Op1->getValue()-1);
         break;
        case ICmpInst::ICMP_SGT:
        case ICmpInst::ICMP_UGT:
          if (!Op1->getValue().isAllOnesValue())
            NextVal = ConstantInt::get(Op1->getValue()+1);
         break;
      }

      if (NextVal) {
        VRPSolver VRP(VN, IG, UB, VR, PS->DTDFS, PS->modified, &IC);
        if (VRP.isRelatedBy(IC.getOperand(0), NextVal,
                            ICmpInst::getInversePredicate(Pred))) {
          ICmpInst *NewIC = new ICmpInst(ICmpInst::ICMP_EQ, IC.getOperand(0),
                                         NextVal, "", &IC);
          NewIC->takeName(&IC);
          IC.replaceAllUsesWith(NewIC);

          // XXX: prove this isn't necessary
          if (unsigned n = VN.valueNumber(&IC, PS->DTDFS->getRootNode()))
            if (VN.value(n) == &IC) IG.remove(n);
          VN.remove(&IC);

          IC.eraseFromParent();
          ++NumSnuggle;
          PS->modified = true;
        }
      }
    }
  }
}

char PredicateSimplifier::ID = 0;
static RegisterPass<PredicateSimplifier>
X("predsimplify", "Predicate Simplifier");

FunctionPass *llvm::createPredicateSimplifierPass() {
  return new PredicateSimplifier();
}
