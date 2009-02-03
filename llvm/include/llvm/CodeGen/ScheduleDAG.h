//===------- llvm/CodeGen/ScheduleDAG.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAG class, which is used as the common
// base class for instruction schedulers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAG_H
#define LLVM_CODEGEN_SCHEDULEDAG_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  struct SUnit;
  class MachineConstantPool;
  class MachineFunction;
  class MachineModuleInfo;
  class MachineRegisterInfo;
  class MachineInstr;
  class TargetRegisterInfo;
  class ScheduleDAG;
  class SelectionDAG;
  class SDNode;
  class TargetInstrInfo;
  class TargetInstrDesc;
  class TargetLowering;
  class TargetMachine;
  class TargetRegisterClass;
  template<class Graph> class GraphWriter;

  /// SDep - Scheduling dependency. This represents one direction of an
  /// edge in the scheduling DAG.
  class SDep {
  public:
    /// Kind - These are the different kinds of scheduling dependencies.
    enum Kind {
      Data,        ///< Regular data dependence (aka true-dependence).
      Anti,        ///< A register anti-dependedence (aka WAR).
      Output,      ///< A register output-dependence (aka WAW).
      Order        ///< Any other ordering dependency.
    };

  private:
    /// Dep - A pointer to the depending/depended-on SUnit, and an enum
    /// indicating the kind of the dependency.
    PointerIntPair<SUnit *, 2, Kind> Dep;

    /// Contents - A union discriminated by the dependence kind.
    union {
      /// Reg - For Data, Anti, and Output dependencies, the associated
      /// register. For Data dependencies that don't currently have a register
      /// assigned, this is set to zero.
      unsigned Reg;

      /// Order - Additional information about Order dependencies.
      struct {
        /// isNormalMemory - True if both sides of the dependence
        /// access memory in non-volatile and fully modeled ways.
        bool isNormalMemory : 1;

        /// isMustAlias - True if both sides of the dependence are known to
        /// access the same memory.
        bool isMustAlias : 1;

        /// isArtificial - True if this is an artificial dependency, meaning
        /// it is not necessary for program correctness, and may be safely
        /// deleted if necessary.
        bool isArtificial : 1;
      } Order;
    } Contents;

    /// Latency - The time associated with this edge. Often this is just
    /// the value of the Latency field of the predecessor, however advanced
    /// models may provide additional information about specific edges.
    unsigned Latency;

  public:
    /// SDep - Construct a null SDep. This is only for use by container
    /// classes which require default constructors. SUnits may not
    /// have null SDep edges.
    SDep() : Dep(0, Data) {}

    /// SDep - Construct an SDep with the specified values.
    SDep(SUnit *S, Kind kind, unsigned latency = 1, unsigned Reg = 0,
         bool isNormalMemory = false, bool isMustAlias = false,
         bool isArtificial = false)
      : Dep(S, kind), Contents(), Latency(latency) {
      switch (kind) {
      case Anti:
      case Output:
        assert(Reg != 0 &&
               "SDep::Anti and SDep::Output must use a non-zero Reg!");
        // fall through
      case Data:
        assert(!isMustAlias && "isMustAlias only applies with SDep::Order!");
        assert(!isArtificial && "isArtificial only applies with SDep::Order!");
        Contents.Reg = Reg;
        break;
      case Order:
        assert(Reg == 0 && "Reg given for non-register dependence!");
        Contents.Order.isNormalMemory = isNormalMemory;
        Contents.Order.isMustAlias = isMustAlias;
        Contents.Order.isArtificial = isArtificial;
        break;
      }
    }

    bool operator==(const SDep &Other) const {
      if (Dep != Other.Dep || Latency != Other.Latency) return false;
      switch (Dep.getInt()) {
      case Data:
      case Anti:
      case Output:
        return Contents.Reg == Other.Contents.Reg;
      case Order:
        return Contents.Order.isNormalMemory ==
                 Other.Contents.Order.isNormalMemory &&
               Contents.Order.isMustAlias == Other.Contents.Order.isMustAlias &&
               Contents.Order.isArtificial == Other.Contents.Order.isArtificial;
      }
      assert(0 && "Invalid dependency kind!");
      return false;
    }

    bool operator!=(const SDep &Other) const {
      return !operator==(Other);
    }

    /// getLatency - Return the latency value for this edge, which roughly
    /// means the minimum number of cycles that must elapse between the
    /// predecessor and the successor, given that they have this edge
    /// between them.
    unsigned getLatency() const {
      return Latency;
    }

    //// getSUnit - Return the SUnit to which this edge points.
    SUnit *getSUnit() const {
      return Dep.getPointer();
    }

    //// setSUnit - Assign the SUnit to which this edge points.
    void setSUnit(SUnit *SU) {
      Dep.setPointer(SU);
    }

    /// getKind - Return an enum value representing the kind of the dependence.
    Kind getKind() const {
      return Dep.getInt();
    }

    /// isCtrl - Shorthand for getKind() != SDep::Data.
    bool isCtrl() const {
      return getKind() != Data;
    }

    /// isNormalMemory - Test if this is an Order dependence between two
    /// memory accesses where both sides of the dependence access memory
    /// in non-volatile and fully modeled ways.
    bool isNormalMemory() const {
      return getKind() == Order && Contents.Order.isNormalMemory;
    }

    /// isMustAlias - Test if this is an Order dependence that is marked
    /// as "must alias", meaning that the SUnits at either end of the edge
    /// have a memory dependence on a known memory location.
    bool isMustAlias() const {
      return getKind() == Order && Contents.Order.isMustAlias;
    }

    /// isArtificial - Test if this is an Order dependence that is marked
    /// as "artificial", meaning it isn't necessary for correctness.
    bool isArtificial() const {
      return getKind() == Order && Contents.Order.isArtificial;
    }

    /// isAssignedRegDep - Test if this is a Data dependence that is
    /// associated with a register.
    bool isAssignedRegDep() const {
      return getKind() == Data && Contents.Reg != 0;
    }

    /// getReg - Return the register associated with this edge. This is
    /// only valid on Data, Anti, and Output edges. On Data edges, this
    /// value may be zero, meaning there is no associated register.
    unsigned getReg() const {
      assert((getKind() == Data || getKind() == Anti || getKind() == Output) &&
             "getReg called on non-register dependence edge!");
      return Contents.Reg;
    }

    /// setReg - Assign the associated register for this edge. This is
    /// only valid on Data, Anti, and Output edges. On Anti and Output
    /// edges, this value must not be zero. On Data edges, the value may
    /// be zero, which would mean that no specific register is associated
    /// with this edge.
    void setReg(unsigned Reg) {
      assert((getKind() == Data || getKind() == Anti || getKind() == Output) &&
             "setReg called on non-register dependence edge!");
      assert((getKind() != Anti || Reg != 0) &&
             "SDep::Anti edge cannot use the zero register!");
      assert((getKind() != Output || Reg != 0) &&
             "SDep::Output edge cannot use the zero register!");
      Contents.Reg = Reg;
    }
  };

  /// SUnit - Scheduling unit. This is a node in the scheduling DAG.
  struct SUnit {
  private:
    SDNode *Node;                       // Representative node.
    MachineInstr *Instr;                // Alternatively, a MachineInstr.
  public:
    SUnit *OrigNode;                    // If not this, the node from which
                                        // this node was cloned.
    
    // Preds/Succs - The SUnits before/after us in the graph.  The boolean value
    // is true if the edge is a token chain edge, false if it is a value edge. 
    SmallVector<SDep, 4> Preds;  // All sunit predecessors.
    SmallVector<SDep, 4> Succs;  // All sunit successors.

    typedef SmallVector<SDep, 4>::iterator pred_iterator;
    typedef SmallVector<SDep, 4>::iterator succ_iterator;
    typedef SmallVector<SDep, 4>::const_iterator const_pred_iterator;
    typedef SmallVector<SDep, 4>::const_iterator const_succ_iterator;
    
    unsigned NodeNum;                   // Entry # of node in the node vector.
    unsigned NodeQueueId;               // Queue id of node.
    unsigned short Latency;             // Node latency.
    short NumPreds;                     // # of SDep::Data preds.
    short NumSuccs;                     // # of SDep::Data sucss.
    short NumPredsLeft;                 // # of preds not scheduled.
    short NumSuccsLeft;                 // # of succs not scheduled.
    bool isTwoAddress     : 1;          // Is a two-address instruction.
    bool isCommutable     : 1;          // Is a commutable instruction.
    bool hasPhysRegDefs   : 1;          // Has physreg defs that are being used.
    bool isPending        : 1;          // True once pending.
    bool isAvailable      : 1;          // True once available.
    bool isScheduled      : 1;          // True once scheduled.
    bool isScheduleHigh   : 1;          // True if preferable to schedule high.
    bool isCloned         : 1;          // True if this node has been cloned.
  private:
    bool isDepthCurrent   : 1;          // True if Depth is current.
    bool isHeightCurrent  : 1;          // True if Height is current.
    unsigned Depth;                     // Node depth.
    unsigned Height;                    // Node height.
  public:
    const TargetRegisterClass *CopyDstRC; // Is a special copy node if not null.
    const TargetRegisterClass *CopySrcRC;
    
    /// SUnit - Construct an SUnit for pre-regalloc scheduling to represent
    /// an SDNode and any nodes flagged to it.
    SUnit(SDNode *node, unsigned nodenum)
      : Node(node), Instr(0), OrigNode(0), NodeNum(nodenum), NodeQueueId(0),
        Latency(0), NumPreds(0), NumSuccs(0), NumPredsLeft(0), NumSuccsLeft(0),
        isTwoAddress(false), isCommutable(false), hasPhysRegDefs(false),
        isPending(false), isAvailable(false), isScheduled(false),
        isScheduleHigh(false), isCloned(false),
        isDepthCurrent(false), isHeightCurrent(false), Depth(0), Height(0),
        CopyDstRC(NULL), CopySrcRC(NULL) {}

    /// SUnit - Construct an SUnit for post-regalloc scheduling to represent
    /// a MachineInstr.
    SUnit(MachineInstr *instr, unsigned nodenum)
      : Node(0), Instr(instr), OrigNode(0), NodeNum(nodenum), NodeQueueId(0),
        Latency(0), NumPreds(0), NumSuccs(0), NumPredsLeft(0), NumSuccsLeft(0),
        isTwoAddress(false), isCommutable(false), hasPhysRegDefs(false),
        isPending(false), isAvailable(false), isScheduled(false),
        isScheduleHigh(false), isCloned(false),
        isDepthCurrent(false), isHeightCurrent(false), Depth(0), Height(0),
        CopyDstRC(NULL), CopySrcRC(NULL) {}

    /// setNode - Assign the representative SDNode for this SUnit.
    /// This may be used during pre-regalloc scheduling.
    void setNode(SDNode *N) {
      assert(!Instr && "Setting SDNode of SUnit with MachineInstr!");
      Node = N;
    }

    /// getNode - Return the representative SDNode for this SUnit.
    /// This may be used during pre-regalloc scheduling.
    SDNode *getNode() const {
      assert(!Instr && "Reading SDNode of SUnit with MachineInstr!");
      return Node;
    }

    /// setInstr - Assign the instruction for the SUnit.
    /// This may be used during post-regalloc scheduling.
    void setInstr(MachineInstr *MI) {
      assert(!Node && "Setting MachineInstr of SUnit with SDNode!");
      Instr = MI;
    }

    /// getInstr - Return the representative MachineInstr for this SUnit.
    /// This may be used during post-regalloc scheduling.
    MachineInstr *getInstr() const {
      assert(!Node && "Reading MachineInstr of SUnit with SDNode!");
      return Instr;
    }

    /// addPred - This adds the specified edge as a pred of the current node if
    /// not already.  It also adds the current node as a successor of the
    /// specified node.
    void addPred(const SDep &D);

    /// removePred - This removes the specified edge as a pred of the current
    /// node if it exists.  It also removes the current node as a successor of
    /// the specified node.
    void removePred(const SDep &D);

    /// getDepth - Return the depth of this node, which is the length of the
    /// maximum path up to any node with has no predecessors.
    unsigned getDepth() const {
      if (!isDepthCurrent) const_cast<SUnit *>(this)->ComputeDepth();
      return Depth;
    }

    /// getHeight - Return the height of this node, which is the length of the
    /// maximum path down to any node with has no successors.
    unsigned getHeight() const {
      if (!isHeightCurrent) const_cast<SUnit *>(this)->ComputeHeight();
      return Height;
    }

    /// setDepthToAtLeast - If NewDepth is greater than this node's depth
    /// value, set it to be the new depth value. This also recursively
    /// marks successor nodes dirty.
    void setDepthToAtLeast(unsigned NewDepth);

    /// setDepthToAtLeast - If NewDepth is greater than this node's depth
    /// value, set it to be the new height value. This also recursively
    /// marks predecessor nodes dirty.
    void setHeightToAtLeast(unsigned NewHeight);

    /// setDepthDirty - Set a flag in this node to indicate that its
    /// stored Depth value will require recomputation the next time
    /// getDepth() is called.
    void setDepthDirty();

    /// setHeightDirty - Set a flag in this node to indicate that its
    /// stored Height value will require recomputation the next time
    /// getHeight() is called.
    void setHeightDirty();

    /// isPred - Test if node N is a predecessor of this node.
    bool isPred(SUnit *N) {
      for (unsigned i = 0, e = (unsigned)Preds.size(); i != e; ++i)
        if (Preds[i].getSUnit() == N)
          return true;
      return false;
    }
    
    /// isSucc - Test if node N is a successor of this node.
    bool isSucc(SUnit *N) {
      for (unsigned i = 0, e = (unsigned)Succs.size(); i != e; ++i)
        if (Succs[i].getSUnit() == N)
          return true;
      return false;
    }
    
    void dump(const ScheduleDAG *G) const;
    void dumpAll(const ScheduleDAG *G) const;
    void print(raw_ostream &O, const ScheduleDAG *G) const;

  private:
    void ComputeDepth();
    void ComputeHeight();
  };

  //===--------------------------------------------------------------------===//
  /// SchedulingPriorityQueue - This interface is used to plug different
  /// priorities computation algorithms into the list scheduler. It implements
  /// the interface of a standard priority queue, where nodes are inserted in 
  /// arbitrary order and returned in priority order.  The computation of the
  /// priority and the representation of the queue are totally up to the
  /// implementation to decide.
  /// 
  class SchedulingPriorityQueue {
  public:
    virtual ~SchedulingPriorityQueue() {}
  
    virtual void initNodes(std::vector<SUnit> &SUnits) = 0;
    virtual void addNode(const SUnit *SU) = 0;
    virtual void updateNode(const SUnit *SU) = 0;
    virtual void releaseState() = 0;

    virtual unsigned size() const = 0;
    virtual bool empty() const = 0;
    virtual void push(SUnit *U) = 0;
  
    virtual void push_all(const std::vector<SUnit *> &Nodes) = 0;
    virtual SUnit *pop() = 0;

    virtual void remove(SUnit *SU) = 0;

    /// ScheduledNode - As each node is scheduled, this method is invoked.  This
    /// allows the priority function to adjust the priority of related
    /// unscheduled nodes, for example.
    ///
    virtual void ScheduledNode(SUnit *) {}

    virtual void UnscheduledNode(SUnit *) {}
  };

  class ScheduleDAG {
  public:
    SelectionDAG *DAG;                    // DAG of the current basic block
    MachineBasicBlock *BB;                // Current basic block
    MachineBasicBlock::iterator Begin;    // The beginning of the range to be scheduled.
    MachineBasicBlock::iterator End;      // The end of the range to be scheduled.
    const TargetMachine &TM;              // Target processor
    const TargetInstrInfo *TII;           // Target instruction information
    const TargetRegisterInfo *TRI;        // Target processor register info
    TargetLowering *TLI;                  // Target lowering info
    MachineFunction &MF;                  // Machine function
    MachineRegisterInfo &MRI;             // Virtual/real register map
    MachineConstantPool *ConstPool;       // Target constant pool
    std::vector<SUnit*> Sequence;         // The schedule. Null SUnit*'s
                                          // represent noop instructions.
    std::vector<SUnit> SUnits;            // The scheduling units.

    explicit ScheduleDAG(MachineFunction &mf);

    virtual ~ScheduleDAG();

    /// viewGraph - Pop up a GraphViz/gv window with the ScheduleDAG rendered
    /// using 'dot'.
    ///
    void viewGraph();
  
    /// Run - perform scheduling.
    ///
    void Run(SelectionDAG *DAG, MachineBasicBlock *MBB,
             MachineBasicBlock::iterator Begin,
             MachineBasicBlock::iterator End);

    /// BuildSchedGraph - Build SUnits and set up their Preds and Succs
    /// to form the scheduling dependency graph.
    ///
    virtual void BuildSchedGraph() = 0;

    /// ComputeLatency - Compute node latency.
    ///
    virtual void ComputeLatency(SUnit *SU) = 0;

  protected:
    /// EmitNoop - Emit a noop instruction.
    ///
    void EmitNoop();

  public:
    virtual MachineBasicBlock *EmitSchedule() = 0;

    void dumpSchedule() const;

    /// Schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    virtual void Schedule() = 0;

    virtual void dumpNode(const SUnit *SU) const = 0;

    /// getGraphNodeLabel - Return a label for an SUnit node in a visualization
    /// of the ScheduleDAG.
    virtual std::string getGraphNodeLabel(const SUnit *SU) const = 0;

    /// addCustomGraphFeatures - Add custom features for a visualization of
    /// the ScheduleDAG.
    virtual void addCustomGraphFeatures(GraphWriter<ScheduleDAG*> &) const {}

#ifndef NDEBUG
    /// VerifySchedule - Verify that all SUnits were scheduled and that
    /// their state is consistent.
    void VerifySchedule(bool isBottomUp);
#endif

  protected:
    void AddMemOperand(MachineInstr *MI, const MachineMemOperand &MO);

    void EmitPhysRegCopy(SUnit *SU, DenseMap<SUnit*, unsigned> &VRBaseMap);

    /// ForceUnitLatencies - Return true if all scheduling edges should be given a
    /// latency value of one.  The default is to return false; schedulers may
    /// override this as needed.
    virtual bool ForceUnitLatencies() const { return false; }

  private:
    /// EmitLiveInCopy - Emit a copy for a live in physical register. If the
    /// physical register has only a single copy use, then coalesced the copy
    /// if possible.
    void EmitLiveInCopy(MachineBasicBlock *MBB,
                        MachineBasicBlock::iterator &InsertPos,
                        unsigned VirtReg, unsigned PhysReg,
                        const TargetRegisterClass *RC,
                        DenseMap<MachineInstr*, unsigned> &CopyRegMap);

    /// EmitLiveInCopies - If this is the first basic block in the function,
    /// and if it has live ins that need to be copied into vregs, emit the
    /// copies into the top of the block.
    void EmitLiveInCopies(MachineBasicBlock *MBB);
  };

  class SUnitIterator : public forward_iterator<SUnit, ptrdiff_t> {
    SUnit *Node;
    unsigned Operand;

    SUnitIterator(SUnit *N, unsigned Op) : Node(N), Operand(Op) {}
  public:
    bool operator==(const SUnitIterator& x) const {
      return Operand == x.Operand;
    }
    bool operator!=(const SUnitIterator& x) const { return !operator==(x); }

    const SUnitIterator &operator=(const SUnitIterator &I) {
      assert(I.Node == Node && "Cannot assign iterators to two different nodes!");
      Operand = I.Operand;
      return *this;
    }

    pointer operator*() const {
      return Node->Preds[Operand].getSUnit();
    }
    pointer operator->() const { return operator*(); }

    SUnitIterator& operator++() {                // Preincrement
      ++Operand;
      return *this;
    }
    SUnitIterator operator++(int) { // Postincrement
      SUnitIterator tmp = *this; ++*this; return tmp;
    }

    static SUnitIterator begin(SUnit *N) { return SUnitIterator(N, 0); }
    static SUnitIterator end  (SUnit *N) {
      return SUnitIterator(N, (unsigned)N->Preds.size());
    }

    unsigned getOperand() const { return Operand; }
    const SUnit *getNode() const { return Node; }
    /// isCtrlDep - Test if this is not an SDep::Data dependence.
    bool isCtrlDep() const {
      return getSDep().isCtrl();
    }
    bool isArtificialDep() const {
      return getSDep().isArtificial();
    }
    const SDep &getSDep() const {
      return Node->Preds[Operand];
    }
  };

  template <> struct GraphTraits<SUnit*> {
    typedef SUnit NodeType;
    typedef SUnitIterator ChildIteratorType;
    static inline NodeType *getEntryNode(SUnit *N) { return N; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      return SUnitIterator::begin(N);
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return SUnitIterator::end(N);
    }
  };

  template <> struct GraphTraits<ScheduleDAG*> : public GraphTraits<SUnit*> {
    typedef std::vector<SUnit>::iterator nodes_iterator;
    static nodes_iterator nodes_begin(ScheduleDAG *G) {
      return G->SUnits.begin();
    }
    static nodes_iterator nodes_end(ScheduleDAG *G) {
      return G->SUnits.end();
    }
  };

  /// ScheduleDAGTopologicalSort is a class that computes a topological
  /// ordering for SUnits and provides methods for dynamically updating
  /// the ordering as new edges are added.
  ///
  /// This allows a very fast implementation of IsReachable, for example.
  ///
  class ScheduleDAGTopologicalSort {
    /// SUnits - A reference to the ScheduleDAG's SUnits.
    std::vector<SUnit> &SUnits;

    /// Index2Node - Maps topological index to the node number.
    std::vector<int> Index2Node;
    /// Node2Index - Maps the node number to its topological index.
    std::vector<int> Node2Index;
    /// Visited - a set of nodes visited during a DFS traversal.
    BitVector Visited;

    /// DFS - make a DFS traversal and mark all nodes affected by the 
    /// edge insertion. These nodes will later get new topological indexes
    /// by means of the Shift method.
    void DFS(const SUnit *SU, int UpperBound, bool& HasLoop);

    /// Shift - reassign topological indexes for the nodes in the DAG
    /// to preserve the topological ordering.
    void Shift(BitVector& Visited, int LowerBound, int UpperBound);

    /// Allocate - assign the topological index to the node n.
    void Allocate(int n, int index);

  public:
    explicit ScheduleDAGTopologicalSort(std::vector<SUnit> &SUnits);

    /// InitDAGTopologicalSorting - create the initial topological 
    /// ordering from the DAG to be scheduled.
    void InitDAGTopologicalSorting();

    /// IsReachable - Checks if SU is reachable from TargetSU.
    bool IsReachable(const SUnit *SU, const SUnit *TargetSU);

    /// WillCreateCycle - Returns true if adding an edge from SU to TargetSU
    /// will create a cycle.
    bool WillCreateCycle(SUnit *SU, SUnit *TargetSU);

    /// AddPred - Updates the topological ordering to accomodate an edge
    /// to be added from SUnit X to SUnit Y.
    void AddPred(SUnit *Y, SUnit *X);

    /// RemovePred - Updates the topological ordering to accomodate an
    /// an edge to be removed from the specified node N from the predecessors
    /// of the current node M.
    void RemovePred(SUnit *M, SUnit *N);

    typedef std::vector<int>::iterator iterator;
    typedef std::vector<int>::const_iterator const_iterator;
    iterator begin() { return Index2Node.begin(); }
    const_iterator begin() const { return Index2Node.begin(); }
    iterator end() { return Index2Node.end(); }
    const_iterator end() const { return Index2Node.end(); }

    typedef std::vector<int>::reverse_iterator reverse_iterator;
    typedef std::vector<int>::const_reverse_iterator const_reverse_iterator;
    reverse_iterator rbegin() { return Index2Node.rbegin(); }
    const_reverse_iterator rbegin() const { return Index2Node.rbegin(); }
    reverse_iterator rend() { return Index2Node.rend(); }
    const_reverse_iterator rend() const { return Index2Node.rend(); }
  };
}

#endif
