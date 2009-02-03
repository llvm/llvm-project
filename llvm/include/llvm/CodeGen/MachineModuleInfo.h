//===-- llvm/CodeGen/MachineModuleInfo.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect meta information for a module.  This information should be in a
// neutral form that can be used by different debugging and exception handling
// schemes.
//
// The organization of information is primarily clustered around the source
// compile units.  The main exception is source line correspondence where
// inlining may interleave code from various compile units.
//
// The following information can be retrieved from the MachineModuleInfo.
//
//  -- Source directories - Directories are uniqued based on their canonical
//     string and assigned a sequential numeric ID (base 1.)
//  -- Source files - Files are also uniqued based on their name and directory
//     ID.  A file ID is sequential number (base 1.)
//  -- Source line correspondence - A vector of file ID, line#, column# triples.
//     A DEBUG_LOCATION instruction is generated  by the DAG Legalizer
//     corresponding to each entry in the source line list.  This allows a debug
//     emitter to generate labels referenced by debug information tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFO_H
#define LLVM_CODEGEN_MACHINEMODULEINFO_H

#include "llvm/Support/Dwarf.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/GlobalValue.h"
#include "llvm/Pass.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
class Constant;
class GlobalVariable;
class MachineBasicBlock;
class MachineFunction;
class Module;
class PointerType;
class StructType;

//===----------------------------------------------------------------------===//
/// SourceLineInfo - This class is used to record source line correspondence.
///
class SourceLineInfo {
  unsigned Line;                        // Source line number.
  unsigned Column;                      // Source column.
  unsigned SourceID;                    // Source ID number.
  unsigned LabelID;                     // Label in code ID number.
public:
  SourceLineInfo(unsigned L, unsigned C, unsigned S, unsigned I)
  : Line(L), Column(C), SourceID(S), LabelID(I) {}
  
  // Accessors
  unsigned getLine()     const { return Line; }
  unsigned getColumn()   const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  unsigned getLabelID()  const { return LabelID; }
};

//===----------------------------------------------------------------------===//
/// LandingPadInfo - This structure is used to retain landing pad info for
/// the current function.
///
struct LandingPadInfo {
  MachineBasicBlock *LandingPadBlock;   // Landing pad block.
  SmallVector<unsigned, 1> BeginLabels; // Labels prior to invoke.
  SmallVector<unsigned, 1> EndLabels;   // Labels after invoke.
  unsigned LandingPadLabel;             // Label at beginning of landing pad.
  Function *Personality;                // Personality function.
  std::vector<int> TypeIds;             // List of type ids (filters negative)

  explicit LandingPadInfo(MachineBasicBlock *MBB)
  : LandingPadBlock(MBB)
  , LandingPadLabel(0)
  , Personality(NULL)  
  {}
};

//===----------------------------------------------------------------------===//
/// MachineModuleInfo - This class contains meta information specific to a
/// module.  Queries can be made by different debugging and exception handling 
/// schemes and reformated for specific use.
///
class MachineModuleInfo : public ImmutablePass {
private:
  // Lines - List of of source line correspondence.
  std::vector<SourceLineInfo> Lines;
  
  // LabelIDList - One entry per assigned label.  Normally the entry is equal to
  // the list index(+1).  If the entry is zero then the label has been deleted.
  // Any other value indicates the label has been deleted by is mapped to
  // another label.
  std::vector<unsigned> LabelIDList;
  
  // FrameMoves - List of moves done by a function's prolog.  Used to construct
  // frame maps by debug and exception handling consumers.
  std::vector<MachineMove> FrameMoves;
  
  // LandingPads - List of LandingPadInfo describing the landing pad information
  // in the current function.
  std::vector<LandingPadInfo> LandingPads;
  
  // TypeInfos - List of C++ TypeInfo used in the current function.
  //
  std::vector<GlobalVariable *> TypeInfos;

  // FilterIds - List of typeids encoding filters used in the current function.
  //
  std::vector<unsigned> FilterIds;

  // FilterEnds - List of the indices in FilterIds corresponding to filter
  // terminators.
  //
  std::vector<unsigned> FilterEnds;

  // Personalities - Vector of all personality functions ever seen. Used to emit
  // common EH frames.
  std::vector<Function *> Personalities;

  // UsedFunctions - the functions in the llvm.used list in a more easily
  // searchable format.
  SmallPtrSet<const Function *, 32> UsedFunctions;

  bool CallsEHReturn;
  bool CallsUnwindInit;
 
  /// DbgInfoAvailable - True if debugging information is available
  /// in this module.
  bool DbgInfoAvailable;
public:
  static char ID; // Pass identification, replacement for typeid

  MachineModuleInfo();
  ~MachineModuleInfo();
  
  /// doInitialization - Initialize the state for a new module.
  ///
  bool doInitialization();
  
  /// doFinalization - Tear down the state after completion of a module.
  ///
  bool doFinalization();
  
  /// BeginFunction - Begin gathering function meta information.
  ///
  void BeginFunction(MachineFunction *MF);
  
  /// EndFunction - Discard function meta information.
  ///
  void EndFunction();

  /// AnalyzeModule - Scan the module for global debug information.
  ///
  void AnalyzeModule(Module &M);
  
  /// hasDebugInfo - Returns true if valid debug info is present.
  ///
  bool hasDebugInfo() const { return DbgInfoAvailable; }
  void setDebugInfoAvailability(bool avail) { DbgInfoAvailable = true; }

  bool callsEHReturn() const { return CallsEHReturn; }
  void setCallsEHReturn(bool b) { CallsEHReturn = b; }

  bool callsUnwindInit() const { return CallsUnwindInit; }
  void setCallsUnwindInit(bool b) { CallsUnwindInit = b; }
  
  /// NextLabelID - Return the next unique label id.
  ///
  unsigned NextLabelID() {
    unsigned ID = (unsigned)LabelIDList.size() + 1;
    LabelIDList.push_back(ID);
    return ID;
  }
  
  /// RecordSourceLine - Records location information and associates it with a
  /// label.  Returns a unique label ID used to generate a label and 
  /// provide correspondence to the source line list.
  unsigned RecordSourceLine(unsigned Line, unsigned Column, unsigned Source);
  
  /// InvalidateLabel - Inhibit use of the specified label # from
  /// MachineModuleInfo, for example because the code was deleted.
  void InvalidateLabel(unsigned LabelID) {
    // Remap to zero to indicate deletion.
    RemapLabel(LabelID, 0);
  }

  /// RemapLabel - Indicate that a label has been merged into another.
  ///
  void RemapLabel(unsigned OldLabelID, unsigned NewLabelID) {
    assert(0 < OldLabelID && OldLabelID <= LabelIDList.size() &&
          "Old label ID out of range.");
    assert(NewLabelID <= LabelIDList.size() &&
          "New label ID out of range.");
    LabelIDList[OldLabelID - 1] = NewLabelID;
  }
  
  /// MappedLabel - Find out the label's final ID.  Zero indicates deletion.
  /// ID != Mapped ID indicates that the label was folded into another label.
  unsigned MappedLabel(unsigned LabelID) const {
    assert(LabelID <= LabelIDList.size() && "Debug label ID out of range.");
    return LabelID ? LabelIDList[LabelID - 1] : 0;
  }

  /// getSourceLines - Return a vector of source lines.
  ///
  const std::vector<SourceLineInfo> &getSourceLines() const {
    return Lines;
  }
  
  /// getFrameMoves - Returns a reference to a list of moves done in the current
  /// function's prologue.  Used to construct frame maps for debug and exception
  /// handling comsumers.
  std::vector<MachineMove> &getFrameMoves() { return FrameMoves; }
  
  //===-EH-----------------------------------------------------------------===//

  /// getOrCreateLandingPadInfo - Find or create an LandingPadInfo for the
  /// specified MachineBasicBlock.
  LandingPadInfo &getOrCreateLandingPadInfo(MachineBasicBlock *LandingPad);

  /// addInvoke - Provide the begin and end labels of an invoke style call and
  /// associate it with a try landing pad block.
  void addInvoke(MachineBasicBlock *LandingPad, unsigned BeginLabel,
                                                unsigned EndLabel);
  
  /// addLandingPad - Add a new panding pad.  Returns the label ID for the 
  /// landing pad entry.
  unsigned addLandingPad(MachineBasicBlock *LandingPad);
  
  /// addPersonality - Provide the personality function for the exception
  /// information.
  void addPersonality(MachineBasicBlock *LandingPad, Function *Personality);

  /// getPersonalityIndex - Get index of the current personality function inside
  /// Personalitites array
  unsigned getPersonalityIndex() const;

  /// getPersonalities - Return array of personality functions ever seen.
  const std::vector<Function *>& getPersonalities() const {
    return Personalities;
  }

  // UsedFunctions - Return set of the functions in the llvm.used list.
  const SmallPtrSet<const Function *, 32>& getUsedFunctions() const {
    return UsedFunctions;
  }

  /// addCatchTypeInfo - Provide the catch typeinfo for a landing pad.
  ///
  void addCatchTypeInfo(MachineBasicBlock *LandingPad,
                        std::vector<GlobalVariable *> &TyInfo);

  /// addFilterTypeInfo - Provide the filter typeinfo for a landing pad.
  ///
  void addFilterTypeInfo(MachineBasicBlock *LandingPad,
                         std::vector<GlobalVariable *> &TyInfo);

  /// addCleanup - Add a cleanup action for a landing pad.
  ///
  void addCleanup(MachineBasicBlock *LandingPad);

  /// getTypeIDFor - Return the type id for the specified typeinfo.  This is 
  /// function wide.
  unsigned getTypeIDFor(GlobalVariable *TI);

  /// getFilterIDFor - Return the id of the filter encoded by TyIds.  This is
  /// function wide.
  int getFilterIDFor(std::vector<unsigned> &TyIds);

  /// TidyLandingPads - Remap landing pad labels and remove any deleted landing
  /// pads.
  void TidyLandingPads();
                        
  /// getLandingPads - Return a reference to the landing pad info for the
  /// current function.
  const std::vector<LandingPadInfo> &getLandingPads() const {
    return LandingPads;
  }
  
  /// getTypeInfos - Return a reference to the C++ typeinfo for the current
  /// function.
  const std::vector<GlobalVariable *> &getTypeInfos() const {
    return TypeInfos;
  }

  /// getFilterIds - Return a reference to the typeids encoding filters used in
  /// the current function.
  const std::vector<unsigned> &getFilterIds() const {
    return FilterIds;
  }

  /// getPersonality - Return a personality function if available.  The presence
  /// of one is required to emit exception handling info.
  Function *getPersonality() const;

}; // End class MachineModuleInfo

} // End llvm namespace

#endif
