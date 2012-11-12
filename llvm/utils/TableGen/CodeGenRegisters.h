//===- CodeGenRegisters.h - Register and RegisterClass Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_REGISTERS_H
#define CODEGEN_REGISTERS_H

#include "SetTheory.h"
#include "llvm/TableGen/Record.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdlib>
#include <map>
#include <string>
#include <set>
#include <vector>

namespace llvm {
  class CodeGenRegBank;

  /// CodeGenSubRegIndex - Represents a sub-register index.
  class CodeGenSubRegIndex {
    Record *const TheDef;
    std::string Name;
    std::string Namespace;

  public:
    const unsigned EnumValue;
    unsigned LaneMask;

    CodeGenSubRegIndex(Record *R, unsigned Enum);
    CodeGenSubRegIndex(StringRef N, StringRef Nspace, unsigned Enum);

    const std::string &getName() const { return Name; }
    const std::string &getNamespace() const { return Namespace; }
    std::string getQualifiedName() const;

    // Order CodeGenSubRegIndex pointers by EnumValue.
    struct Less {
      bool operator()(const CodeGenSubRegIndex *A,
                      const CodeGenSubRegIndex *B) const {
        assert(A && B);
        return A->EnumValue < B->EnumValue;
      }
    };

    // Map of composite subreg indices.
    typedef std::map<CodeGenSubRegIndex*, CodeGenSubRegIndex*, Less> CompMap;

    // Returns the subreg index that results from composing this with Idx.
    // Returns NULL if this and Idx don't compose.
    CodeGenSubRegIndex *compose(CodeGenSubRegIndex *Idx) const {
      CompMap::const_iterator I = Composed.find(Idx);
      return I == Composed.end() ? 0 : I->second;
    }

    // Add a composite subreg index: this+A = B.
    // Return a conflicting composite, or NULL
    CodeGenSubRegIndex *addComposite(CodeGenSubRegIndex *A,
                                     CodeGenSubRegIndex *B) {
      assert(A && B);
      std::pair<CompMap::iterator, bool> Ins =
        Composed.insert(std::make_pair(A, B));
      return (Ins.second || Ins.first->second == B) ? 0 : Ins.first->second;
    }

    // Update the composite maps of components specified in 'ComposedOf'.
    void updateComponents(CodeGenRegBank&);

    // Return the map of composites.
    const CompMap &getComposites() const { return Composed; }

    // Compute LaneMask from Composed. Return LaneMask.
    unsigned computeLaneMask();

  private:
    CompMap Composed;
  };

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    unsigned EnumValue;
    unsigned CostPerUse;
    bool CoveredBySubRegs;

    // Map SubRegIndex -> Register.
    typedef std::map<CodeGenSubRegIndex*, CodeGenRegister*,
                     CodeGenSubRegIndex::Less> SubRegMap;

    CodeGenRegister(Record *R, unsigned Enum);

    const std::string &getName() const;

    // Extract more information from TheDef. This is used to build an object
    // graph after all CodeGenRegister objects have been created.
    void buildObjectGraph(CodeGenRegBank&);

    // Lazily compute a map of all sub-registers.
    // This includes unique entries for all sub-sub-registers.
    const SubRegMap &computeSubRegs(CodeGenRegBank&);

    // Compute extra sub-registers by combining the existing sub-registers.
    void computeSecondarySubRegs(CodeGenRegBank&);

    // Add this as a super-register to all sub-registers after the sub-register
    // graph has been built.
    void computeSuperRegs(CodeGenRegBank&);

    const SubRegMap &getSubRegs() const {
      assert(SubRegsComplete && "Must precompute sub-registers");
      return SubRegs;
    }

    // Add sub-registers to OSet following a pre-order defined by the .td file.
    void addSubRegsPreOrder(SetVector<const CodeGenRegister*> &OSet,
                            CodeGenRegBank&) const;

    // Return the sub-register index naming Reg as a sub-register of this
    // register. Returns NULL if Reg is not a sub-register.
    CodeGenSubRegIndex *getSubRegIndex(const CodeGenRegister *Reg) const {
      return SubReg2Idx.lookup(Reg);
    }

    typedef std::vector<const CodeGenRegister*> SuperRegList;

    // Get the list of super-registers in topological order, small to large.
    // This is valid after computeSubRegs visits all registers during RegBank
    // construction.
    const SuperRegList &getSuperRegs() const {
      assert(SubRegsComplete && "Must precompute sub-registers");
      return SuperRegs;
    }

    // Get the list of ad hoc aliases. The graph is symmetric, so the list
    // contains all registers in 'Aliases', and all registers that mention this
    // register in 'Aliases'.
    ArrayRef<CodeGenRegister*> getExplicitAliases() const {
      return ExplicitAliases;
    }

    // Get the topological signature of this register. This is a small integer
    // less than RegBank.getNumTopoSigs(). Registers with the same TopoSig have
    // identical sub-register structure. That is, they support the same set of
    // sub-register indices mapping to the same kind of sub-registers
    // (TopoSig-wise).
    unsigned getTopoSig() const {
      assert(SuperRegsComplete && "TopoSigs haven't been computed yet.");
      return TopoSig;
    }

    // List of register units in ascending order.
    typedef SmallVector<unsigned, 16> RegUnitList;

    // How many entries in RegUnitList are native?
    unsigned NumNativeRegUnits;

    // Get the list of register units.
    // This is only valid after computeSubRegs() completes.
    const RegUnitList &getRegUnits() const { return RegUnits; }

    // Get the native register units. This is a prefix of getRegUnits().
    ArrayRef<unsigned> getNativeRegUnits() const {
      return makeArrayRef(RegUnits).slice(0, NumNativeRegUnits);
    }

    // Inherit register units from subregisters.
    // Return true if the RegUnits changed.
    bool inheritRegUnits(CodeGenRegBank &RegBank);

    // Adopt a register unit for pressure tracking.
    // A unit is adopted iff its unit number is >= NumNativeRegUnits.
    void adoptRegUnit(unsigned RUID) { RegUnits.push_back(RUID); }

    // Get the sum of this register's register unit weights.
    unsigned getWeight(const CodeGenRegBank &RegBank) const;

    // Order CodeGenRegister pointers by EnumValue.
    struct Less {
      bool operator()(const CodeGenRegister *A,
                      const CodeGenRegister *B) const {
        assert(A && B);
        return A->EnumValue < B->EnumValue;
      }
    };

    // Canonically ordered set.
    typedef std::set<const CodeGenRegister*, Less> Set;

    // Compute the set of registers overlapping this.
    void computeOverlaps(Set &Overlaps, const CodeGenRegBank&) const;

  private:
    bool SubRegsComplete;
    bool SuperRegsComplete;
    unsigned TopoSig;

    // The sub-registers explicit in the .td file form a tree.
    SmallVector<CodeGenSubRegIndex*, 8> ExplicitSubRegIndices;
    SmallVector<CodeGenRegister*, 8> ExplicitSubRegs;

    // Explicit ad hoc aliases, symmetrized to form an undirected graph.
    SmallVector<CodeGenRegister*, 8> ExplicitAliases;

    // Super-registers where this is the first explicit sub-register.
    SuperRegList LeadingSuperRegs;

    SubRegMap SubRegs;
    SuperRegList SuperRegs;
    DenseMap<const CodeGenRegister*, CodeGenSubRegIndex*> SubReg2Idx;
    RegUnitList RegUnits;
  };


  class CodeGenRegisterClass {
    CodeGenRegister::Set Members;
    // Allocation orders. Order[0] always contains all registers in Members.
    std::vector<SmallVector<Record*, 16> > Orders;
    // Bit mask of sub-classes including this, indexed by their EnumValue.
    BitVector SubClasses;
    // List of super-classes, topologocally ordered to have the larger classes
    // first.  This is the same as sorting by EnumValue.
    SmallVector<CodeGenRegisterClass*, 4> SuperClasses;
    Record *TheDef;
    std::string Name;

    // For a synthesized class, inherit missing properties from the nearest
    // super-class.
    void inheritProperties(CodeGenRegBank&);

    // Map SubRegIndex -> sub-class.  This is the largest sub-class where all
    // registers have a SubRegIndex sub-register.
    DenseMap<CodeGenSubRegIndex*, CodeGenRegisterClass*> SubClassWithSubReg;

    // Map SubRegIndex -> set of super-reg classes.  This is all register
    // classes SuperRC such that:
    //
    //   R:SubRegIndex in this RC for all R in SuperRC.
    //
    DenseMap<CodeGenSubRegIndex*,
             SmallPtrSet<CodeGenRegisterClass*, 8> > SuperRegClasses;

    // Bit vector of TopoSigs for the registers in this class. This will be
    // very sparse on regular architectures.
    BitVector TopoSigs;

  public:
    unsigned EnumValue;
    std::string Namespace;
    std::vector<MVT::SimpleValueType> VTs;
    unsigned SpillSize;
    unsigned SpillAlignment;
    int CopyCost;
    bool Allocatable;
    std::string AltOrderSelect;

    // Return the Record that defined this class, or NULL if the class was
    // created by TableGen.
    Record *getDef() const { return TheDef; }

    const std::string &getName() const { return Name; }
    std::string getQualifiedName() const;
    const std::vector<MVT::SimpleValueType> &getValueTypes() const {return VTs;}
    unsigned getNumValueTypes() const { return VTs.size(); }

    MVT::SimpleValueType getValueTypeNum(unsigned VTNum) const {
      if (VTNum < VTs.size())
        return VTs[VTNum];
      llvm_unreachable("VTNum greater than number of ValueTypes in RegClass!");
    }

    // Return true if this this class contains the register.
    bool contains(const CodeGenRegister*) const;

    // Returns true if RC is a subclass.
    // RC is a sub-class of this class if it is a valid replacement for any
    // instruction operand where a register of this classis required. It must
    // satisfy these conditions:
    //
    // 1. All RC registers are also in this.
    // 2. The RC spill size must not be smaller than our spill size.
    // 3. RC spill alignment must be compatible with ours.
    //
    bool hasSubClass(const CodeGenRegisterClass *RC) const {
      return SubClasses.test(RC->EnumValue);
    }

    // getSubClassWithSubReg - Returns the largest sub-class where all
    // registers have a SubIdx sub-register.
    CodeGenRegisterClass*
    getSubClassWithSubReg(CodeGenSubRegIndex *SubIdx) const {
      return SubClassWithSubReg.lookup(SubIdx);
    }

    void setSubClassWithSubReg(CodeGenSubRegIndex *SubIdx,
                               CodeGenRegisterClass *SubRC) {
      SubClassWithSubReg[SubIdx] = SubRC;
    }

    // getSuperRegClasses - Returns a bit vector of all register classes
    // containing only SubIdx super-registers of this class.
    void getSuperRegClasses(CodeGenSubRegIndex *SubIdx, BitVector &Out) const;

    // addSuperRegClass - Add a class containing only SudIdx super-registers.
    void addSuperRegClass(CodeGenSubRegIndex *SubIdx,
                          CodeGenRegisterClass *SuperRC) {
      SuperRegClasses[SubIdx].insert(SuperRC);
    }

    // getSubClasses - Returns a constant BitVector of subclasses indexed by
    // EnumValue.
    // The SubClasses vector includs an entry for this class.
    const BitVector &getSubClasses() const { return SubClasses; }

    // getSuperClasses - Returns a list of super classes ordered by EnumValue.
    // The array does not include an entry for this class.
    ArrayRef<CodeGenRegisterClass*> getSuperClasses() const {
      return SuperClasses;
    }

    // Returns an ordered list of class members.
    // The order of registers is the same as in the .td file.
    // No = 0 is the default allocation order, No = 1 is the first alternative.
    ArrayRef<Record*> getOrder(unsigned No = 0) const {
        return Orders[No];
    }

    // Return the total number of allocation orders available.
    unsigned getNumOrders() const { return Orders.size(); }

    // Get the set of registers.  This set contains the same registers as
    // getOrder(0).
    const CodeGenRegister::Set &getMembers() const { return Members; }

    // Get a bit vector of TopoSigs present in this register class.
    const BitVector &getTopoSigs() const { return TopoSigs; }

    // Populate a unique sorted list of units from a register set.
    void buildRegUnitSet(std::vector<unsigned> &RegUnits) const;

    CodeGenRegisterClass(CodeGenRegBank&, Record *R);

    // A key representing the parts of a register class used for forming
    // sub-classes.  Note the ordering provided by this key is not the same as
    // the topological order used for the EnumValues.
    struct Key {
      const CodeGenRegister::Set *Members;
      unsigned SpillSize;
      unsigned SpillAlignment;

      Key(const Key &O)
        : Members(O.Members),
          SpillSize(O.SpillSize),
          SpillAlignment(O.SpillAlignment) {}

      Key(const CodeGenRegister::Set *M, unsigned S = 0, unsigned A = 0)
        : Members(M), SpillSize(S), SpillAlignment(A) {}

      Key(const CodeGenRegisterClass &RC)
        : Members(&RC.getMembers()),
          SpillSize(RC.SpillSize),
          SpillAlignment(RC.SpillAlignment) {}

      // Lexicographical order of (Members, SpillSize, SpillAlignment).
      bool operator<(const Key&) const;
    };

    // Create a non-user defined register class.
    CodeGenRegisterClass(CodeGenRegBank&, StringRef Name, Key Props);

    // Called by CodeGenRegBank::CodeGenRegBank().
    static void computeSubClasses(CodeGenRegBank&);
  };

  // Register units are used to model interference and register pressure.
  // Every register is assigned one or more register units such that two
  // registers overlap if and only if they have a register unit in common.
  //
  // Normally, one register unit is created per leaf register. Non-leaf
  // registers inherit the units of their sub-registers.
  struct RegUnit {
    // Weight assigned to this RegUnit for estimating register pressure.
    // This is useful when equalizing weights in register classes with mixed
    // register topologies.
    unsigned Weight;

    // Each native RegUnit corresponds to one or two root registers. The full
    // set of registers containing this unit can be computed as the union of
    // these two registers and their super-registers.
    const CodeGenRegister *Roots[2];

    RegUnit() : Weight(0) { Roots[0] = Roots[1] = 0; }

    ArrayRef<const CodeGenRegister*> getRoots() const {
      assert(!(Roots[1] && !Roots[0]) && "Invalid roots array");
      return makeArrayRef(Roots, !!Roots[0] + !!Roots[1]);
    }
  };

  // Each RegUnitSet is a sorted vector with a name.
  struct RegUnitSet {
    typedef std::vector<unsigned>::const_iterator iterator;

    std::string Name;
    std::vector<unsigned> Units;
  };

  // Base vector for identifying TopoSigs. The contents uniquely identify a
  // TopoSig, only computeSuperRegs needs to know how.
  typedef SmallVector<unsigned, 16> TopoSigId;

  // CodeGenRegBank - Represent a target's registers and the relations between
  // them.
  class CodeGenRegBank {
    SetTheory Sets;

    // SubRegIndices.
    std::vector<CodeGenSubRegIndex*> SubRegIndices;
    DenseMap<Record*, CodeGenSubRegIndex*> Def2SubRegIdx;

    CodeGenSubRegIndex *createSubRegIndex(StringRef Name, StringRef NameSpace);

    typedef std::map<SmallVector<CodeGenSubRegIndex*, 8>,
                     CodeGenSubRegIndex*> ConcatIdxMap;
    ConcatIdxMap ConcatIdx;

    // Registers.
    std::vector<CodeGenRegister*> Registers;
    StringMap<CodeGenRegister*> RegistersByName;
    DenseMap<Record*, CodeGenRegister*> Def2Reg;
    unsigned NumNativeRegUnits;

    std::map<TopoSigId, unsigned> TopoSigs;

    // Includes native (0..NumNativeRegUnits-1) and adopted register units.
    SmallVector<RegUnit, 8> RegUnits;

    // Register classes.
    std::vector<CodeGenRegisterClass*> RegClasses;
    DenseMap<Record*, CodeGenRegisterClass*> Def2RC;
    typedef std::map<CodeGenRegisterClass::Key, CodeGenRegisterClass*> RCKeyMap;
    RCKeyMap Key2RC;

    // Remember each unique set of register units. Initially, this contains a
    // unique set for each register class. Simliar sets are coalesced with
    // pruneUnitSets and new supersets are inferred during computeRegUnitSets.
    std::vector<RegUnitSet> RegUnitSets;

    // Map RegisterClass index to the index of the RegUnitSet that contains the
    // class's units and any inferred RegUnit supersets.
    std::vector<std::vector<unsigned> > RegClassUnitSets;

    // Add RC to *2RC maps.
    void addToMaps(CodeGenRegisterClass*);

    // Create a synthetic sub-class if it is missing.
    CodeGenRegisterClass *getOrCreateSubClass(const CodeGenRegisterClass *RC,
                                              const CodeGenRegister::Set *Membs,
                                              StringRef Name);

    // Infer missing register classes.
    void computeInferredRegisterClasses();
    void inferCommonSubClass(CodeGenRegisterClass *RC);
    void inferSubClassWithSubReg(CodeGenRegisterClass *RC);
    void inferMatchingSuperRegClass(CodeGenRegisterClass *RC,
                                    unsigned FirstSubRegRC = 0);

    // Iteratively prune unit sets.
    void pruneUnitSets();

    // Compute a weight for each register unit created during getSubRegs.
    void computeRegUnitWeights();

    // Create a RegUnitSet for each RegClass and infer superclasses.
    void computeRegUnitSets();

    // Populate the Composite map from sub-register relationships.
    void computeComposites();

    // Compute a lane mask for each sub-register index.
    void computeSubRegIndexLaneMasks();

  public:
    CodeGenRegBank(RecordKeeper&);

    SetTheory &getSets() { return Sets; }

    // Sub-register indices. The first NumNamedIndices are defined by the user
    // in the .td files. The rest are synthesized such that all sub-registers
    // have a unique name.
    ArrayRef<CodeGenSubRegIndex*> getSubRegIndices() { return SubRegIndices; }

    // Find a SubRegIndex form its Record def.
    CodeGenSubRegIndex *getSubRegIdx(Record*);

    // Find or create a sub-register index representing the A+B composition.
    CodeGenSubRegIndex *getCompositeSubRegIndex(CodeGenSubRegIndex *A,
                                                CodeGenSubRegIndex *B);

    // Find or create a sub-register index representing the concatenation of
    // non-overlapping sibling indices.
    CodeGenSubRegIndex *
      getConcatSubRegIndex(const SmallVector<CodeGenSubRegIndex*, 8>&);

    void
    addConcatSubRegIndex(const SmallVector<CodeGenSubRegIndex*, 8> &Parts,
                         CodeGenSubRegIndex *Idx) {
      ConcatIdx.insert(std::make_pair(Parts, Idx));
    }

    const std::vector<CodeGenRegister*> &getRegisters() { return Registers; }
    const StringMap<CodeGenRegister*> &getRegistersByName() {
      return RegistersByName;
    }

    // Find a register from its Record def.
    CodeGenRegister *getReg(Record*);

    // Get a Register's index into the Registers array.
    unsigned getRegIndex(const CodeGenRegister *Reg) const {
      return Reg->EnumValue - 1;
    }

    // Return the number of allocated TopoSigs. The first TopoSig representing
    // leaf registers is allocated number 0.
    unsigned getNumTopoSigs() const {
      return TopoSigs.size();
    }

    // Find or create a TopoSig for the given TopoSigId.
    // This function is only for use by CodeGenRegister::computeSuperRegs().
    // Others should simply use Reg->getTopoSig().
    unsigned getTopoSig(const TopoSigId &Id) {
      return TopoSigs.insert(std::make_pair(Id, TopoSigs.size())).first->second;
    }

    // Create a native register unit that is associated with one or two root
    // registers.
    unsigned newRegUnit(CodeGenRegister *R0, CodeGenRegister *R1 = 0) {
      RegUnits.resize(RegUnits.size() + 1);
      RegUnits.back().Roots[0] = R0;
      RegUnits.back().Roots[1] = R1;
      return RegUnits.size() - 1;
    }

    // Create a new non-native register unit that can be adopted by a register
    // to increase its pressure. Note that NumNativeRegUnits is not increased.
    unsigned newRegUnit(unsigned Weight) {
      RegUnits.resize(RegUnits.size() + 1);
      RegUnits.back().Weight = Weight;
      return RegUnits.size() - 1;
    }

    // Native units are the singular unit of a leaf register. Register aliasing
    // is completely characterized by native units. Adopted units exist to give
    // register additional weight but don't affect aliasing.
    bool isNativeUnit(unsigned RUID) {
      return RUID < NumNativeRegUnits;
    }

    unsigned getNumNativeRegUnits() const {
      return NumNativeRegUnits;
    }

    RegUnit &getRegUnit(unsigned RUID) { return RegUnits[RUID]; }
    const RegUnit &getRegUnit(unsigned RUID) const { return RegUnits[RUID]; }

    ArrayRef<CodeGenRegisterClass*> getRegClasses() const {
      return RegClasses;
    }

    // Find a register class from its def.
    CodeGenRegisterClass *getRegClass(Record*);

    /// getRegisterClassForRegister - Find the register class that contains the
    /// specified physical register.  If the register is not in a register
    /// class, return null. If the register is in multiple classes, and the
    /// classes have a superset-subset relationship and the same set of types,
    /// return the superclass.  Otherwise return null.
    const CodeGenRegisterClass* getRegClassForRegister(Record *R);

    // Get the sum of unit weights.
    unsigned getRegUnitSetWeight(const std::vector<unsigned> &Units) const {
      unsigned Weight = 0;
      for (std::vector<unsigned>::const_iterator
             I = Units.begin(), E = Units.end(); I != E; ++I)
        Weight += getRegUnit(*I).Weight;
      return Weight;
    }

    // Increase a RegUnitWeight.
    void increaseRegUnitWeight(unsigned RUID, unsigned Inc) {
      getRegUnit(RUID).Weight += Inc;
    }

    // Get the number of register pressure dimensions.
    unsigned getNumRegPressureSets() const { return RegUnitSets.size(); }

    // Get a set of register unit IDs for a given dimension of pressure.
    RegUnitSet getRegPressureSet(unsigned Idx) const {
      return RegUnitSets[Idx];
    }

    // Get a list of pressure set IDs for a register class. Liveness of a
    // register in this class impacts each pressure set in this list by the
    // weight of the register. An exact solution requires all registers in a
    // class to have the same class, but it is not strictly guaranteed.
    ArrayRef<unsigned> getRCPressureSetIDs(unsigned RCIdx) const {
      return RegClassUnitSets[RCIdx];
    }

    // Computed derived records such as missing sub-register indices.
    void computeDerivedInfo();

    // Compute the set of registers completely covered by the registers in Regs.
    // The returned BitVector will have a bit set for each register in Regs,
    // all sub-registers, and all super-registers that are covered by the
    // registers in Regs.
    //
    // This is used to compute the mask of call-preserved registers from a list
    // of callee-saves.
    BitVector computeCoveredRegisters(ArrayRef<Record*> Regs);
  };
}

#endif
