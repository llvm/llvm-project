//===-- CodeGen/AsmPrinter/DwarfException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

static TimerGroup &getDwarfTimerGroup() {
  static TimerGroup DwarfTimerGroup("Dwarf Exception");
  return DwarfTimerGroup;
}

DwarfException::DwarfException(raw_ostream &OS, AsmPrinter *A,
                               const TargetAsmInfo *T)
  : Dwarf(OS, A, T, "eh"), shouldEmitTable(false), shouldEmitMoves(false),
    shouldEmitTableModule(false), shouldEmitMovesModule(false),
    ExceptionTimer(0) {
  if (TimePassesIsEnabled) 
    ExceptionTimer = new Timer("Dwarf Exception Writer",
                               getDwarfTimerGroup());
}

DwarfException::~DwarfException() {
  delete ExceptionTimer;
}

void DwarfException::EmitCommonEHFrame(const Function *Personality,
                                       unsigned Index) {
  // Size and sign of stack growth.
  int stackGrowth =
    Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
    TargetFrameInfo::StackGrowsUp ?
    TD->getPointerSize() : -TD->getPointerSize();

  // Begin eh frame section.
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getEHFrameSection());

  if (TAI->is_EHSymbolPrivate())
    O << TAI->getPrivateGlobalPrefix();

  O << "EH_frame" << Index << ":\n";
  EmitLabel("section_eh_frame", Index);

  // Define base labels.
  EmitLabel("eh_frame_common", Index);

  // Define the eh frame length.
  EmitDifference("eh_frame_common_end", Index,
                 "eh_frame_common_begin", Index, true);
  Asm->EOL("Length of Common Information Entry");

  // EH frame header.
  EmitLabel("eh_frame_common_begin", Index);
  Asm->EmitInt32((int)0);
  Asm->EOL("CIE Identifier Tag");
  Asm->EmitInt8(dwarf::DW_CIE_VERSION);
  Asm->EOL("CIE Version");

  // The personality presence indicates that language specific information will
  // show up in the eh frame.
  Asm->EmitString(Personality ? "zPLR" : "zR");
  Asm->EOL("CIE Augmentation");

  // Round out reader.
  Asm->EmitULEB128Bytes(1);
  Asm->EOL("CIE Code Alignment Factor");
  Asm->EmitSLEB128Bytes(stackGrowth);
  Asm->EOL("CIE Data Alignment Factor");
  Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), true));
  Asm->EOL("CIE Return Address Column");

  // If there is a personality, we need to indicate the functions location.
  if (Personality) {
    Asm->EmitULEB128Bytes(7);
    Asm->EOL("Augmentation Size");

    if (TAI->getNeedsIndirectEncoding()) {
      Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4 |
                    dwarf::DW_EH_PE_indirect);
      Asm->EOL("Personality (pcrel sdata4 indirect)");
    } else {
      Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
      Asm->EOL("Personality (pcrel sdata4)");
    }

    PrintRelDirective(true);
    O << TAI->getPersonalityPrefix();
    Asm->EmitExternalGlobal((const GlobalVariable *)(Personality));
    O << TAI->getPersonalitySuffix();
    if (strcmp(TAI->getPersonalitySuffix(), "+4@GOTPCREL"))
      O << "-" << TAI->getPCSymbol();
    Asm->EOL("Personality");

    Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    Asm->EOL("LSDA Encoding (pcrel sdata4)");

    Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    Asm->EOL("FDE Encoding (pcrel sdata4)");
  } else {
    Asm->EmitULEB128Bytes(1);
    Asm->EOL("Augmentation Size");

    Asm->EmitInt8(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    Asm->EOL("FDE Encoding (pcrel sdata4)");
  }

  // Indicate locations of general callee saved registers in frame.
  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);
  EmitFrameMoves(NULL, 0, Moves, true);

  // On Darwin the linker honors the alignment of eh_frame, which means it must
  // be 8-byte on 64-bit targets to match what gcc does.  Otherwise you get
  // holes which confuse readers of eh_frame.
  Asm->EmitAlignment(TD->getPointerSize() == sizeof(int32_t) ? 2 : 3,
                     0, 0, false);
  EmitLabel("eh_frame_common_end", Index);

  Asm->EOL();
}

/// EmitEHFrame - Emit function exception frame information.
///
void DwarfException::EmitEHFrame(const FunctionEHFrameInfo &EHFrameInfo) {
  assert(!EHFrameInfo.function->hasAvailableExternallyLinkage() && 
         "Should not emit 'available externally' functions at all");

  const Function *TheFunc = EHFrameInfo.function;
  
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getEHFrameSection());
  
  // Externally visible entry into the functions eh frame info. If the
  // corresponding function is static, this should not be externally visible.
  if (!TheFunc->hasLocalLinkage())
    if (const char *GlobalEHDirective = TAI->getGlobalEHDirective())
      O << GlobalEHDirective << EHFrameInfo.FnName << "\n";

  // If corresponding function is weak definition, this should be too.
  if (TheFunc->isWeakForLinker() && TAI->getWeakDefDirective())
    O << TAI->getWeakDefDirective() << EHFrameInfo.FnName << "\n";

  // If there are no calls then you can't unwind.  This may mean we can omit the
  // EH Frame, but some environments do not handle weak absolute symbols. If
  // UnwindTablesMandatory is set we cannot do this optimization; the unwind
  // info is to be available for non-EH uses.
  if (!EHFrameInfo.hasCalls && !UnwindTablesMandatory &&
      (!TheFunc->isWeakForLinker() ||
       !TAI->getWeakDefDirective() ||
       TAI->getSupportsWeakOmittedEHFrame())) {
    O << EHFrameInfo.FnName << " = 0\n";
    // This name has no connection to the function, so it might get
    // dead-stripped when the function is not, erroneously.  Prohibit
    // dead-stripping unconditionally.
    if (const char *UsedDirective = TAI->getUsedDirective())
      O << UsedDirective << EHFrameInfo.FnName << "\n\n";
  } else {
    O << EHFrameInfo.FnName << ":\n";

    // EH frame header.
    EmitDifference("eh_frame_end", EHFrameInfo.Number,
                   "eh_frame_begin", EHFrameInfo.Number, true);
    Asm->EOL("Length of Frame Information Entry");

    EmitLabel("eh_frame_begin", EHFrameInfo.Number);

    EmitSectionOffset("eh_frame_begin", "eh_frame_common",
                      EHFrameInfo.Number, EHFrameInfo.PersonalityIndex,
                      true, true, false);

    Asm->EOL("FDE CIE offset");

    EmitReference("eh_func_begin", EHFrameInfo.Number, true, true);
    Asm->EOL("FDE initial location");
    EmitDifference("eh_func_end", EHFrameInfo.Number,
                   "eh_func_begin", EHFrameInfo.Number, true);
    Asm->EOL("FDE address range");

    // If there is a personality and landing pads then point to the language
    // specific data area in the exception table.
    if (EHFrameInfo.PersonalityIndex) {
      bool is4Byte = TD->getPointerSize() == sizeof(int32_t);

      Asm->EmitULEB128Bytes(is4Byte ? 4 : 8);
      Asm->EOL("Augmentation size");

      if (EHFrameInfo.hasLandingPads)
        EmitReference("exception", EHFrameInfo.Number, true, false);
      else {
	if (is4Byte)
	  Asm->EmitInt32((int)0);
	else
	  Asm->EmitInt64((int)0);
      }
      Asm->EOL("Language Specific Data Area");
    } else {
      Asm->EmitULEB128Bytes(0);
      Asm->EOL("Augmentation size");
    }

    // Indicate locations of function specific callee saved registers in frame.
    EmitFrameMoves("eh_func_begin", EHFrameInfo.Number, EHFrameInfo.Moves, 
                   true);

    // On Darwin the linker honors the alignment of eh_frame, which means it
    // must be 8-byte on 64-bit targets to match what gcc does.  Otherwise you
    // get holes which confuse readers of eh_frame.
    Asm->EmitAlignment(TD->getPointerSize() == sizeof(int32_t) ? 2 : 3,
                       0, 0, false);
    EmitLabel("eh_frame_end", EHFrameInfo.Number);

    // If the function is marked used, this table should be also.  We cannot
    // make the mark unconditional in this case, since retaining the table also
    // retains the function in this case, and there is code around that depends
    // on unused functions (calling undefined externals) being dead-stripped to
    // link correctly.  Yes, there really is.
    if (MMI->isUsedFunction(EHFrameInfo.function))
      if (const char *UsedDirective = TAI->getUsedDirective())
        O << UsedDirective << EHFrameInfo.FnName << "\n\n";
  }
}

/// SharedTypeIds - How many leading type ids two landing pads have in common.
unsigned DwarfException::SharedTypeIds(const LandingPadInfo *L,
                                       const LandingPadInfo *R) {
  const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
  unsigned LSize = LIds.size(), RSize = RIds.size();
  unsigned MinSize = LSize < RSize ? LSize : RSize;
  unsigned Count = 0;

  for (; Count != MinSize; ++Count)
    if (LIds[Count] != RIds[Count])
      return Count;

  return Count;
}

/// PadLT - Order landing pads lexicographically by type id.
bool DwarfException::PadLT(const LandingPadInfo *L, const LandingPadInfo *R) {
  const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
  unsigned LSize = LIds.size(), RSize = RIds.size();
  unsigned MinSize = LSize < RSize ? LSize : RSize;

  for (unsigned i = 0; i != MinSize; ++i)
    if (LIds[i] != RIds[i])
      return LIds[i] < RIds[i];

  return LSize < RSize;
}

/// ComputeActionsTable - Compute the actions table and gather the first action
/// index for each landing pad site.
unsigned DwarfException::
ComputeActionsTable(const SmallVectorImpl<const LandingPadInfo*> &LandingPads,
                    SmallVectorImpl<ActionEntry> &Actions,
                    SmallVectorImpl<unsigned> &FirstActions) {

  // The action table follows the call-site table in the LSDA. The individual
  // records are of two types:
  //
  //   * Catch clause
  //   * Exception specification
  //
  // The two record kinds have the same format, with only small differences.
  // They are distinguished by the "switch value" field: Catch clauses
  // (TypeInfos) have strictly positive switch values, and exception
  // specifications (FilterIds) have strictly negative switch values. Value 0
  // indicates a catch-all clause.
  //
  // Negative type IDs index into FilterIds. Positive type IDs index into
  // TypeInfos.  The value written for a positive type ID is just the type ID
  // itself.  For a negative type ID, however, the value written is the
  // (negative) byte offset of the corresponding FilterIds entry.  The byte
  // offset is usually equal to the type ID (because the FilterIds entries are
  // written using a variable width encoding, which outputs one byte per entry
  // as long as the value written is not too large) but can differ.  This kind
  // of complication does not occur for positive type IDs because type infos are
  // output using a fixed width encoding.  FilterOffsets[i] holds the byte
  // offset corresponding to FilterIds[i].

  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  SmallVector<int, 16> FilterOffsets;
  FilterOffsets.reserve(FilterIds.size());
  int Offset = -1;

  for (std::vector<unsigned>::const_iterator
         I = FilterIds.begin(), E = FilterIds.end(); I != E; ++I) {
    FilterOffsets.push_back(Offset);
    Offset -= TargetAsmInfo::getULEB128Size(*I);
  }

  FirstActions.reserve(LandingPads.size());

  int FirstAction = 0;
  unsigned SizeActions = 0;
  const LandingPadInfo *PrevLPI = 0;

  for (SmallVectorImpl<const LandingPadInfo *>::const_iterator
         I = LandingPads.begin(), E = LandingPads.end(); I != E; ++I) {
    const LandingPadInfo *LPI = *I;
    const std::vector<int> &TypeIds = LPI->TypeIds;
    const unsigned NumShared = PrevLPI ? SharedTypeIds(LPI, PrevLPI) : 0;
    unsigned SizeSiteActions = 0;

    if (NumShared < TypeIds.size()) {
      unsigned SizeAction = 0;
      ActionEntry *PrevAction = 0;

      if (NumShared) {
        const unsigned SizePrevIds = PrevLPI->TypeIds.size();
        assert(Actions.size());
        PrevAction = &Actions.back();
        SizeAction = TargetAsmInfo::getSLEB128Size(PrevAction->NextAction) +
          TargetAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);

        for (unsigned j = NumShared; j != SizePrevIds; ++j) {
          SizeAction -=
            TargetAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
          SizeAction += -PrevAction->NextAction;
          PrevAction = PrevAction->Previous;
        }
      }

      // Compute the actions.
      for (unsigned J = NumShared, M = TypeIds.size(); J != M; ++J) {
        int TypeID = TypeIds[J];
        assert(-1 - TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
        int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
        unsigned SizeTypeID = TargetAsmInfo::getSLEB128Size(ValueForTypeID);

        int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
        SizeAction = SizeTypeID + TargetAsmInfo::getSLEB128Size(NextAction);
        SizeSiteActions += SizeAction;

        ActionEntry Action = { ValueForTypeID, NextAction, PrevAction };
        Actions.push_back(Action);
        PrevAction = &Actions.back();
      }

      // Record the first action of the landing pad site.
      FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
    } // else identical - re-use previous FirstAction

    // Information used when created the call-site table. The action record
    // field of the call site record is the offset of the first associated
    // action record, relative to the start of the actions table. This value is
    // biased by 1 (1 in dicating the start of the actions table), and 0
    // indicates that there are no actions.
    FirstActions.push_back(FirstAction);

    // Compute this sites contribution to size.
    SizeActions += SizeSiteActions;

    PrevLPI = LPI;
  }

  return SizeActions;
}

/// ComputeCallSiteTable - Compute the call-site table.  The entry for an invoke
/// has a try-range containing the call, a non-zero landing pad, and an
/// appropriate action.  The entry for an ordinary call has a try-range
/// containing the call and zero for the landing pad and the action.  Calls
/// marked 'nounwind' have no entry and must not be contained in the try-range
/// of any entry - they form gaps in the table.  Entries must be ordered by
/// try-range address.
void DwarfException::
ComputeCallSiteTable(SmallVectorImpl<CallSiteEntry> &CallSites,
                     const RangeMapType &PadMap,
                     const SmallVectorImpl<const LandingPadInfo *> &LandingPads,
                     const SmallVectorImpl<unsigned> &FirstActions) {
  // The end label of the previous invoke or nounwind try-range.
  unsigned LastLabel = 0;

  // Whether there is a potentially throwing instruction (currently this means
  // an ordinary call) between the end of the previous try-range and now.
  bool SawPotentiallyThrowing = false;

  // Whether the last CallSite entry was for an invoke.
  bool PreviousIsInvoke = false;

  // Visit all instructions in order of address.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
         MI != E; ++MI) {
      if (!MI->isLabel()) {
        SawPotentiallyThrowing |= MI->getDesc().isCall();
        continue;
      }

      unsigned BeginLabel = MI->getOperand(0).getImm();
      assert(BeginLabel && "Invalid label!");

      // End of the previous try-range?
      if (BeginLabel == LastLabel)
        SawPotentiallyThrowing = false;

      // Beginning of a new try-range?
      RangeMapType::iterator L = PadMap.find(BeginLabel);
      if (L == PadMap.end())
        // Nope, it was just some random label.
        continue;

      const PadRange &P = L->second;
      const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];
      assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
             "Inconsistent landing pad map!");

      // For Dwarf exception handling (SjLj handling doesn't use this). If some
      // instruction between the previous try-range and this one may throw,
      // create a call-site entry with no landing pad for the region between the
      // try-ranges.
      if (SawPotentiallyThrowing &&
          TAI->getExceptionHandlingType() == ExceptionHandling::Dwarf) {
        CallSiteEntry Site = { LastLabel, BeginLabel, 0, 0 };
        CallSites.push_back(Site);
        PreviousIsInvoke = false;
      }

      LastLabel = LandingPad->EndLabels[P.RangeIndex];
      assert(BeginLabel && LastLabel && "Invalid landing pad!");

      if (LandingPad->LandingPadLabel) {
        // This try-range is for an invoke.
        CallSiteEntry Site = {
          BeginLabel,
          LastLabel,
          LandingPad->LandingPadLabel,
          FirstActions[P.PadIndex]
        };

        // Try to merge with the previous call-site.
        if (PreviousIsInvoke) {
          CallSiteEntry &Prev = CallSites.back();
          if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
            // Extend the range of the previous entry.
            Prev.EndLabel = Site.EndLabel;
            continue;
          }
        }

        // Otherwise, create a new call-site.
        CallSites.push_back(Site);
        PreviousIsInvoke = true;
      } else {
        // Create a gap.
        PreviousIsInvoke = false;
      }
    }
  }

  // If some instruction between the previous try-range and the end of the
  // function may throw, create a call-site entry with no landing pad for the
  // region following the try-range.
  if (SawPotentiallyThrowing &&
      TAI->getExceptionHandlingType() == ExceptionHandling::Dwarf) {
    CallSiteEntry Site = { LastLabel, 0, 0, 0 };
    CallSites.push_back(Site);
  }
}

/// EmitExceptionTable - Emit landing pads and actions.
///
/// The general organization of the table is complex, but the basic concepts are
/// easy.  First there is a header which describes the location and organization
/// of the three components that follow.
/// 
///  1. The landing pad site information describes the range of code covered by
///     the try.  In our case it's an accumulation of the ranges covered by the
///     invokes in the try.  There is also a reference to the landing pad that
///     handles the exception once processed.  Finally an index into the actions
///     table.
///  2. The action table, in our case, is composed of pairs of type IDs and next
///     action offset.  Starting with the action index from the landing pad
///     site, each type ID is checked for a match to the current exception.  If
///     it matches then the exception and type id are passed on to the landing
///     pad.  Otherwise the next action is looked up.  This chain is terminated
///     with a next action of zero.  If no type id is found the the frame is
///     unwound and handling continues.
///  3. Type ID table contains references to all the C++ typeinfo for all
///     catches in the function.  This tables is reversed indexed base 1.
void DwarfException::EmitExceptionTable() {
  const std::vector<GlobalVariable *> &TypeInfos = MMI->getTypeInfos();
  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  if (PadInfos.empty()) return;

  // Sort the landing pads in order of their type ids.  This is used to fold
  // duplicate actions.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());

  for (unsigned i = 0, N = PadInfos.size(); i != N; ++i)
    LandingPads.push_back(&PadInfos[i]);

  std::sort(LandingPads.begin(), LandingPads.end(), PadLT);

  // Compute the actions table and gather the first action index for each
  // landing pad site.
  SmallVector<ActionEntry, 32> Actions;
  SmallVector<unsigned, 64> FirstActions;
  unsigned SizeActions = ComputeActionsTable(LandingPads, Actions, FirstActions);

  // Invokes and nounwind calls have entries in PadMap (due to being bracketed
  // by try-range labels when lowered).  Ordinary calls do not, so appropriate
  // try-ranges for them need be deduced when using Dwarf exception handling.
  RangeMapType PadMap;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LandingPad = LandingPads[i];
    for (unsigned j = 0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
      unsigned BeginLabel = LandingPad->BeginLabels[j];
      assert(!PadMap.count(BeginLabel) && "Duplicate landing pad labels!");
      PadRange P = { i, j };
      PadMap[BeginLabel] = P;
    }
  }

  // Compute the call-site table.
  SmallVector<CallSiteEntry, 64> CallSites;
  ComputeCallSiteTable(CallSites, PadMap, LandingPads, FirstActions);

  // Final tallies.

  // Call sites.
  const unsigned SiteStartSize  = sizeof(int32_t); // DW_EH_PE_udata4
  const unsigned SiteLengthSize = sizeof(int32_t); // DW_EH_PE_udata4
  const unsigned LandingPadSize = sizeof(int32_t); // DW_EH_PE_udata4
  unsigned SizeSites;

  bool HaveTTData = (TAI->getExceptionHandlingType() == ExceptionHandling::SjLj)
    ? (!TypeInfos.empty() || !FilterIds.empty()) : true;


  if (TAI->getExceptionHandlingType() == ExceptionHandling::SjLj) {
    SizeSites = 0;
  } else
    SizeSites = CallSites.size() *
      (SiteStartSize + SiteLengthSize + LandingPadSize);
  for (unsigned i = 0, e = CallSites.size(); i < e; ++i) {
    SizeSites += TargetAsmInfo::getULEB128Size(CallSites[i].Action);
    if (TAI->getExceptionHandlingType() == ExceptionHandling::SjLj)
      SizeSites += TargetAsmInfo::getULEB128Size(i);
  }
  // Type infos.
  const unsigned TypeInfoSize = TD->getPointerSize(); // DW_EH_PE_absptr
  unsigned SizeTypes = TypeInfos.size() * TypeInfoSize;

  unsigned TypeOffset = sizeof(int8_t) + // Call site format
    TargetAsmInfo::getULEB128Size(SizeSites) + // Call-site table length
    SizeSites + SizeActions + SizeTypes;

  unsigned TotalSize = sizeof(int8_t) + // LPStart format
                       sizeof(int8_t) + // TType format
       (HaveTTData ?
          TargetAsmInfo::getULEB128Size(TypeOffset) : 0) + // TType base offset
                       TypeOffset;

  unsigned SizeAlign = (4 - TotalSize) & 3;

  // Begin the exception table.
  const MCSection *LSDASection = Asm->getObjFileLowering().getLSDASection();
  Asm->OutStreamer.SwitchSection(LSDASection);
  Asm->EmitAlignment(2, 0, 0, false);
  O << "GCC_except_table" << SubprogramCount << ":\n";

  for (unsigned i = 0; i != SizeAlign; ++i) {
    Asm->EmitInt8(0);
    Asm->EOL("Padding");
  }

  EmitLabel("exception", SubprogramCount);
  if (TAI->getExceptionHandlingType() == ExceptionHandling::SjLj) {
    std::string SjLjName = "_lsda_";
    SjLjName += MF->getFunction()->getName().str();
    EmitLabel(SjLjName.c_str(), 0);
  }

  // Emit the header.
  Asm->EmitInt8(dwarf::DW_EH_PE_omit);
  Asm->EOL("@LPStart format (DW_EH_PE_omit)");

#if 0
  if (TypeInfos.empty() && FilterIds.empty()) {
    // If there are no typeinfos or filters, there is nothing to emit, optimize
    // by specifying the "omit" encoding.
    Asm->EmitInt8(dwarf::DW_EH_PE_omit);
    Asm->EOL("@TType format (DW_EH_PE_omit)");
  } else {
    // Okay, we have actual filters or typeinfos to emit.  As such, we need to
    // pick a type encoding for them.  We're about to emit a list of pointers to
    // typeinfo objects at the end of the LSDA.  However, unless we're in static
    // mode, this reference will require a relocation by the dynamic linker.
    //
    // Because of this, we have a couple of options:
    //   1) If we are in -static mode, we can always use an absolute reference
    //      from the LSDA, because the static linker will resolve it.
    //   2) Otherwise, if the LSDA section is writable, we can output the direct
    //      reference to the typeinfo and allow the dynamic linker to relocate
    //      it.  Since it is in a writable section, the dynamic linker won't
    //      have a problem.
    //   3) Finally, if we're in PIC mode and the LDSA section isn't writable,
    //      we need to use some form of indirection.  For example, on Darwin,
    //      we can output a statically-relocatable reference to a dyld stub. The
    //      offset to the stub is constant, but the contents are in a section
    //      that is updated by the dynamic linker.  This is easy enough, but we
    //      need to tell the personality function of the unwinder to indirect
    //      through the dyld stub.
    //
    // FIXME: When this is actually implemented, we'll have to emit the stubs
    // somewhere.  This predicate should be moved to a shared location that is
    // in target-independent code.
    //
    if (LSDASection->isWritable() ||
        Asm->TM.getRelocationModel() == Reloc::Static) {
      Asm->EmitInt8(DW_EH_PE_absptr);
      Asm->EOL("TType format (DW_EH_PE_absptr)");
    } else {
      Asm->EmitInt8(DW_EH_PE_pcrel | DW_EH_PE_indirect | DW_EH_PE_sdata4);
      Asm->EOL("TType format (DW_EH_PE_pcrel | DW_EH_PE_indirect"
               " | DW_EH_PE_sdata4)");
    }
    Asm->EmitULEB128Bytes(TypeOffset);
    Asm->EOL("TType base offset");
  }
#else
  // For SjLj exceptions, if there is no TypeInfo, then we just explicitly
  // say that we're omitting that bit.
  // FIXME: does this apply to Dwarf also? The above #if 0 implies yes?
  if (!HaveTTData) {
    Asm->EmitInt8(dwarf::DW_EH_PE_omit);
    Asm->EOL("@TType format (DW_EH_PE_omit)");
  } else {
    Asm->EmitInt8(dwarf::DW_EH_PE_absptr);
    Asm->EOL("@TType format (DW_EH_PE_absptr)");
    Asm->EmitULEB128Bytes(TypeOffset);
    Asm->EOL("@TType base offset");
  }
#endif

  // SjLj Exception handilng
  if (TAI->getExceptionHandlingType() == ExceptionHandling::SjLj) {
    Asm->EmitInt8(dwarf::DW_EH_PE_udata4);
    Asm->EOL("Call site format (DW_EH_PE_udata4)");
    Asm->EmitULEB128Bytes(SizeSites);
    Asm->EOL("Call site table length");

    // Emit the landing pad site information.
    unsigned idx = 0;
    for (SmallVectorImpl<CallSiteEntry>::const_iterator
         I = CallSites.begin(), E = CallSites.end(); I != E; ++I, ++idx) {
      const CallSiteEntry &S = *I;

      // Offset of the landing pad, counted in 16-byte bundles relative to the
      // @LPStart address.
      Asm->EmitULEB128Bytes(idx);
      Asm->EOL("Landing pad");

      // Offset of the first associated action record, relative to the start of
      // the action table. This value is biased by 1 (1 indicates the start of
      // the action table), and 0 indicates that there are no actions.
      Asm->EmitULEB128Bytes(S.Action);
      Asm->EOL("Action");
    }
  } else {
    // DWARF Exception handling
    assert(TAI->getExceptionHandlingType() == ExceptionHandling::Dwarf);

    // The call-site table is a list of all call sites that may throw an
    // exception (including C++ 'throw' statements) in the procedure
    // fragment. It immediately follows the LSDA header. Each entry indicates,
    // for a given call, the first corresponding action record and corresponding
    // landing pad.
    //
    // The table begins with the number of bytes, stored as an LEB128
    // compressed, unsigned integer. The records immediately follow the record
    // count. They are sorted in increasing call-site address. Each record
    // indicates:
    //
    //   * The position of the call-site.
    //   * The position of the landing pad.
    //   * The first action record for that call site.
    //
    // A missing entry in the call-site table indicates that a call is not
    // supposed to throw. Such calls include:
    //
    //   * Calls to destructors within cleanup code. C++ semantics forbids these
    //     calls to throw.
    //   * Calls to intrinsic routines in the standard library which are known
    //     not to throw (sin, memcpy, et al).
    //
    // If the runtime does not find the call-site entry for a given call, it
    // will call `terminate()'.

    // Emit the landing pad call site table.
    Asm->EmitInt8(dwarf::DW_EH_PE_udata4);
    Asm->EOL("Call site format (DW_EH_PE_udata4)");
    Asm->EmitULEB128Bytes(SizeSites);
    Asm->EOL("Call site table size");

    for (SmallVectorImpl<CallSiteEntry>::const_iterator
         I = CallSites.begin(), E = CallSites.end(); I != E; ++I) {
      const CallSiteEntry &S = *I;
      const char *BeginTag;
      unsigned BeginNumber;

      if (!S.BeginLabel) {
        BeginTag = "eh_func_begin";
        BeginNumber = SubprogramCount;
      } else {
        BeginTag = "label";
        BeginNumber = S.BeginLabel;
      }

      // Offset of the call site relative to the previous call site, counted in
      // number of 16-byte bundles. The first call site is counted relative to
      // the start of the procedure fragment.
      EmitSectionOffset(BeginTag, "eh_func_begin", BeginNumber, SubprogramCount,
                        true, true);
      Asm->EOL("Region start");

      if (!S.EndLabel)
        EmitDifference("eh_func_end", SubprogramCount, BeginTag, BeginNumber,
                       true);
      else
        EmitDifference("label", S.EndLabel, BeginTag, BeginNumber, true);

      Asm->EOL("Region length");

      // Offset of the landing pad, counted in 16-byte bundles relative to the
      // @LPStart address.
      if (!S.PadLabel)
        Asm->EmitInt32(0);
      else
        EmitSectionOffset("label", "eh_func_begin", S.PadLabel, SubprogramCount,
                          true, true);

      Asm->EOL("Landing pad");

      // Offset of the first associated action record, relative to the start of
      // the action table. This value is biased by 1 (1 indicates the start of
      // the action table), and 0 indicates that there are no actions.
      Asm->EmitULEB128Bytes(S.Action);
      Asm->EOL("Action");
    }
  }

  // Emit the Action Table.
  for (SmallVectorImpl<ActionEntry>::const_iterator
         I = Actions.begin(), E = Actions.end(); I != E; ++I) {
    const ActionEntry &Action = *I;

    // Type Filter
    //
    //   Used by the runtime to match the type of the thrown exception to the
    //   type of the catch clauses or the types in the exception specification.

    Asm->EmitSLEB128Bytes(Action.ValueForTypeID);
    Asm->EOL("TypeInfo index");

    // Action Record
    //
    //   Self-relative signed displacement in bytes of the next action record,
    //   or 0 if there is no next action record.

    Asm->EmitSLEB128Bytes(Action.NextAction);
    Asm->EOL("Next action");
  }

  // Emit the Catch Clauses. The code for the catch clauses following the same
  // try is similar to a switch statement. The catch clause action record
  // informs the runtime about the type of a catch clause and about the
  // associated switch value.
  //
  //  Action Record Fields:
  //  
  //   * Filter Value
  //     Positive value, starting at 1. Index in the types table of the
  //     __typeinfo for the catch-clause type. 1 is the first word preceding
  //     TTBase, 2 is the second word, and so on. Used by the runtime to check
  //     if the thrown exception type matches the catch-clause type. Back-end
  //     generated switch statements check against this value.
  //
  //   * Next
  //     Signed offset, in bytes from the start of this field, to the next
  //     chained action record, or zero if none.
  //
  // The order of the action records determined by the next field is the order
  // of the catch clauses as they appear in the source code, and must be kept in
  // the same order. As a result, changing the order of the catch clause would
  // change the semantics of the program.
  for (std::vector<GlobalVariable *>::const_reverse_iterator
         I = TypeInfos.rbegin(), E = TypeInfos.rend(); I != E; ++I) {
    const GlobalVariable *GV = *I;
    PrintRelDirective();

    if (GV) {
      std::string GLN;
      O << Asm->getGlobalLinkName(GV, GLN);
    } else {
      O << "0";
    }

    Asm->EOL("TypeInfo");
  }

  // Emit the Type Table.
  for (std::vector<unsigned>::const_iterator
         I = FilterIds.begin(), E = FilterIds.end(); I < E; ++I) {
    unsigned TypeID = *I;
    Asm->EmitULEB128Bytes(TypeID);
    Asm->EOL("Filter TypeInfo index");
  }

  Asm->EmitAlignment(2, 0, 0, false);
}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfException::EndModule() {
  if (TAI->getExceptionHandlingType() != ExceptionHandling::Dwarf)
    return;
  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  if (shouldEmitMovesModule || shouldEmitTableModule) {
    const std::vector<Function *> Personalities = MMI->getPersonalities();
    for (unsigned i = 0; i < Personalities.size(); ++i)
      EmitCommonEHFrame(Personalities[i], i);

    for (std::vector<FunctionEHFrameInfo>::iterator I = EHFrames.begin(),
           E = EHFrames.end(); I != E; ++I)
      EmitEHFrame(*I);
  }

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}

/// BeginFunction - Gather pre-function exception information.  Assumes being
/// emitted immediately after the function entry point.
void DwarfException::BeginFunction(MachineFunction *MF) {
  if (TimePassesIsEnabled)
    ExceptionTimer->startTimer();

  this->MF = MF;
  shouldEmitTable = shouldEmitMoves = false;

  if (MMI && TAI->doesSupportExceptionHandling()) {
    // Map all labels and get rid of any dead landing pads.
    MMI->TidyLandingPads();

    // If any landing pads survive, we need an EH table.
    if (MMI->getLandingPads().size())
      shouldEmitTable = true;

    // See if we need frame move info.
    if (!MF->getFunction()->doesNotThrow() || UnwindTablesMandatory)
      shouldEmitMoves = true;

    if (shouldEmitMoves || shouldEmitTable)
      // Assumes in correct section after the entry point.
      EmitLabel("eh_func_begin", ++SubprogramCount);
  }

  shouldEmitTableModule |= shouldEmitTable;
  shouldEmitMovesModule |= shouldEmitMoves;

  if (TimePassesIsEnabled)
    ExceptionTimer->stopTimer();
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfException::EndFunction() {
  if (TimePassesIsEnabled) 
    ExceptionTimer->startTimer();

  if (shouldEmitMoves || shouldEmitTable) {
    EmitLabel("eh_func_end", SubprogramCount);
    EmitExceptionTable();

    // Save EH frame information
    EHFrames.push_back(
        FunctionEHFrameInfo(getAsm()->getCurrentFunctionEHName(MF),
                            SubprogramCount,
                            MMI->getPersonalityIndex(),
                            MF->getFrameInfo()->hasCalls(),
                            !MMI->getLandingPads().empty(),
                            MMI->getFrameMoves(),
                            MF->getFunction()));
  }

  if (TimePassesIsEnabled) 
    ExceptionTimer->stopTimer();
}
