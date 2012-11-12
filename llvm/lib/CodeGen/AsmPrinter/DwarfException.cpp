//===-- CodeGen/AsmPrinter/DwarfException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing DWARF exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

DwarfException::DwarfException(AsmPrinter *A)
  : Asm(A), MMI(Asm->MMI) {}

DwarfException::~DwarfException() {}

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
    Offset -= MCAsmInfo::getULEB128Size(*I);
  }

  FirstActions.reserve(LandingPads.size());

  int FirstAction = 0;
  unsigned SizeActions = 0;
  const LandingPadInfo *PrevLPI = 0;

  for (SmallVectorImpl<const LandingPadInfo *>::const_iterator
         I = LandingPads.begin(), E = LandingPads.end(); I != E; ++I) {
    const LandingPadInfo *LPI = *I;
    const std::vector<int> &TypeIds = LPI->TypeIds;
    unsigned NumShared = PrevLPI ? SharedTypeIds(LPI, PrevLPI) : 0;
    unsigned SizeSiteActions = 0;

    if (NumShared < TypeIds.size()) {
      unsigned SizeAction = 0;
      unsigned PrevAction = (unsigned)-1;

      if (NumShared) {
        unsigned SizePrevIds = PrevLPI->TypeIds.size();
        assert(Actions.size());
        PrevAction = Actions.size() - 1;
        SizeAction =
          MCAsmInfo::getSLEB128Size(Actions[PrevAction].NextAction) +
          MCAsmInfo::getSLEB128Size(Actions[PrevAction].ValueForTypeID);

        for (unsigned j = NumShared; j != SizePrevIds; ++j) {
          assert(PrevAction != (unsigned)-1 && "PrevAction is invalid!");
          SizeAction -=
            MCAsmInfo::getSLEB128Size(Actions[PrevAction].ValueForTypeID);
          SizeAction += -Actions[PrevAction].NextAction;
          PrevAction = Actions[PrevAction].Previous;
        }
      }

      // Compute the actions.
      for (unsigned J = NumShared, M = TypeIds.size(); J != M; ++J) {
        int TypeID = TypeIds[J];
        assert(-1 - TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
        int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
        unsigned SizeTypeID = MCAsmInfo::getSLEB128Size(ValueForTypeID);

        int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
        SizeAction = SizeTypeID + MCAsmInfo::getSLEB128Size(NextAction);
        SizeSiteActions += SizeAction;

        ActionEntry Action = { ValueForTypeID, NextAction, PrevAction };
        Actions.push_back(Action);
        PrevAction = Actions.size() - 1;
      }

      // Record the first action of the landing pad site.
      FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
    } // else identical - re-use previous FirstAction

    // Information used when created the call-site table. The action record
    // field of the call site record is the offset of the first associated
    // action record, relative to the start of the actions table. This value is
    // biased by 1 (1 indicating the start of the actions table), and 0
    // indicates that there are no actions.
    FirstActions.push_back(FirstAction);

    // Compute this sites contribution to size.
    SizeActions += SizeSiteActions;

    PrevLPI = LPI;
  }

  return SizeActions;
}

/// CallToNoUnwindFunction - Return `true' if this is a call to a function
/// marked `nounwind'. Return `false' otherwise.
bool DwarfException::CallToNoUnwindFunction(const MachineInstr *MI) {
  assert(MI->isCall() && "This should be a call instruction!");

  bool MarkedNoUnwind = false;
  bool SawFunc = false;

  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);

    if (!MO.isGlobal()) continue;

    const Function *F = dyn_cast<Function>(MO.getGlobal());
    if (F == 0) continue;

    if (SawFunc) {
      // Be conservative. If we have more than one function operand for this
      // call, then we can't make the assumption that it's the callee and
      // not a parameter to the call.
      //
      // FIXME: Determine if there's a way to say that `F' is the callee or
      // parameter.
      MarkedNoUnwind = false;
      break;
    }

    MarkedNoUnwind = F->doesNotThrow();
    SawFunc = true;
  }

  return MarkedNoUnwind;
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
  MCSymbol *LastLabel = 0;

  // Whether there is a potentially throwing instruction (currently this means
  // an ordinary call) between the end of the previous try-range and now.
  bool SawPotentiallyThrowing = false;

  // Whether the last CallSite entry was for an invoke.
  bool PreviousIsInvoke = false;

  // Visit all instructions in order of address.
  for (MachineFunction::const_iterator I = Asm->MF->begin(), E = Asm->MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
         MI != E; ++MI) {
      if (!MI->isLabel()) {
        if (MI->isCall())
          SawPotentiallyThrowing |= !CallToNoUnwindFunction(MI);
        continue;
      }

      // End of the previous try-range?
      MCSymbol *BeginLabel = MI->getOperand(0).getMCSymbol();
      if (BeginLabel == LastLabel)
        SawPotentiallyThrowing = false;

      // Beginning of a new try-range?
      RangeMapType::const_iterator L = PadMap.find(BeginLabel);
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
      if (SawPotentiallyThrowing && Asm->MAI->isExceptionHandlingDwarf()) {
        CallSiteEntry Site = { LastLabel, BeginLabel, 0, 0 };
        CallSites.push_back(Site);
        PreviousIsInvoke = false;
      }

      LastLabel = LandingPad->EndLabels[P.RangeIndex];
      assert(BeginLabel && LastLabel && "Invalid landing pad!");

      if (!LandingPad->LandingPadLabel) {
        // Create a gap.
        PreviousIsInvoke = false;
      } else {
        // This try-range is for an invoke.
        CallSiteEntry Site = {
          BeginLabel,
          LastLabel,
          LandingPad->LandingPadLabel,
          FirstActions[P.PadIndex]
        };

        // Try to merge with the previous call-site. SJLJ doesn't do this
        if (PreviousIsInvoke && Asm->MAI->isExceptionHandlingDwarf()) {
          CallSiteEntry &Prev = CallSites.back();
          if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
            // Extend the range of the previous entry.
            Prev.EndLabel = Site.EndLabel;
            continue;
          }
        }

        // Otherwise, create a new call-site.
        if (Asm->MAI->isExceptionHandlingDwarf())
          CallSites.push_back(Site);
        else {
          // SjLj EH must maintain the call sites in the order assigned
          // to them by the SjLjPrepare pass.
          unsigned SiteNo = MMI->getCallSiteBeginLabel(BeginLabel);
          if (CallSites.size() < SiteNo)
            CallSites.resize(SiteNo);
          CallSites[SiteNo - 1] = Site;
        }
        PreviousIsInvoke = true;
      }
    }
  }

  // If some instruction between the previous try-range and the end of the
  // function may throw, create a call-site entry with no landing pad for the
  // region following the try-range.
  if (SawPotentiallyThrowing && Asm->MAI->isExceptionHandlingDwarf()) {
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
///     with a next action of zero.  If no type id is found then the frame is
///     unwound and handling continues.
///  3. Type ID table contains references to all the C++ typeinfo for all
///     catches in the function.  This tables is reverse indexed base 1.
void DwarfException::EmitExceptionTable() {
  const std::vector<const GlobalVariable *> &TypeInfos = MMI->getTypeInfos();
  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();

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
  unsigned SizeActions=ComputeActionsTable(LandingPads, Actions, FirstActions);

  // Invokes and nounwind calls have entries in PadMap (due to being bracketed
  // by try-range labels when lowered).  Ordinary calls do not, so appropriate
  // try-ranges for them need be deduced when using DWARF exception handling.
  RangeMapType PadMap;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LandingPad = LandingPads[i];
    for (unsigned j = 0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
      MCSymbol *BeginLabel = LandingPad->BeginLabels[j];
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
  bool IsSJLJ = Asm->MAI->getExceptionHandlingType() == ExceptionHandling::SjLj;
  bool HaveTTData = IsSJLJ ? (!TypeInfos.empty() || !FilterIds.empty()) : true;

  unsigned CallSiteTableLength;
  if (IsSJLJ)
    CallSiteTableLength = 0;
  else {
    unsigned SiteStartSize  = 4; // dwarf::DW_EH_PE_udata4
    unsigned SiteLengthSize = 4; // dwarf::DW_EH_PE_udata4
    unsigned LandingPadSize = 4; // dwarf::DW_EH_PE_udata4
    CallSiteTableLength =
      CallSites.size() * (SiteStartSize + SiteLengthSize + LandingPadSize);
  }

  for (unsigned i = 0, e = CallSites.size(); i < e; ++i) {
    CallSiteTableLength += MCAsmInfo::getULEB128Size(CallSites[i].Action);
    if (IsSJLJ)
      CallSiteTableLength += MCAsmInfo::getULEB128Size(i);
  }

  // Type infos.
  const MCSection *LSDASection = Asm->getObjFileLowering().getLSDASection();
  unsigned TTypeEncoding;
  unsigned TypeFormatSize;

  if (!HaveTTData) {
    // For SjLj exceptions, if there is no TypeInfo, then we just explicitly say
    // that we're omitting that bit.
    TTypeEncoding = dwarf::DW_EH_PE_omit;
    // dwarf::DW_EH_PE_absptr
    TypeFormatSize = Asm->getDataLayout().getPointerSize();
  } else {
    // Okay, we have actual filters or typeinfos to emit.  As such, we need to
    // pick a type encoding for them.  We're about to emit a list of pointers to
    // typeinfo objects at the end of the LSDA.  However, unless we're in static
    // mode, this reference will require a relocation by the dynamic linker.
    //
    // Because of this, we have a couple of options:
    //
    //   1) If we are in -static mode, we can always use an absolute reference
    //      from the LSDA, because the static linker will resolve it.
    //
    //   2) Otherwise, if the LSDA section is writable, we can output the direct
    //      reference to the typeinfo and allow the dynamic linker to relocate
    //      it.  Since it is in a writable section, the dynamic linker won't
    //      have a problem.
    //
    //   3) Finally, if we're in PIC mode and the LDSA section isn't writable,
    //      we need to use some form of indirection.  For example, on Darwin,
    //      we can output a statically-relocatable reference to a dyld stub. The
    //      offset to the stub is constant, but the contents are in a section
    //      that is updated by the dynamic linker.  This is easy enough, but we
    //      need to tell the personality function of the unwinder to indirect
    //      through the dyld stub.
    //
    // FIXME: When (3) is actually implemented, we'll have to emit the stubs
    // somewhere.  This predicate should be moved to a shared location that is
    // in target-independent code.
    //
    TTypeEncoding = Asm->getObjFileLowering().getTTypeEncoding();
    TypeFormatSize = Asm->GetSizeOfEncodedValue(TTypeEncoding);
  }

  // Begin the exception table.
  // Sometimes we want not to emit the data into separate section (e.g. ARM
  // EHABI). In this case LSDASection will be NULL.
  if (LSDASection)
    Asm->OutStreamer.SwitchSection(LSDASection);
  Asm->EmitAlignment(2);

  // Emit the LSDA.
  MCSymbol *GCCETSym =
    Asm->OutContext.GetOrCreateSymbol(Twine("GCC_except_table")+
                                      Twine(Asm->getFunctionNumber()));
  Asm->OutStreamer.EmitLabel(GCCETSym);
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("exception",
                                                Asm->getFunctionNumber()));

  if (IsSJLJ)
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("_LSDA_",
                                                  Asm->getFunctionNumber()));

  // Emit the LSDA header.
  Asm->EmitEncodingByte(dwarf::DW_EH_PE_omit, "@LPStart");
  Asm->EmitEncodingByte(TTypeEncoding, "@TType");

  // The type infos need to be aligned. GCC does this by inserting padding just
  // before the type infos. However, this changes the size of the exception
  // table, so you need to take this into account when you output the exception
  // table size. However, the size is output using a variable length encoding.
  // So by increasing the size by inserting padding, you may increase the number
  // of bytes used for writing the size. If it increases, say by one byte, then
  // you now need to output one less byte of padding to get the type infos
  // aligned. However this decreases the size of the exception table. This
  // changes the value you have to output for the exception table size. Due to
  // the variable length encoding, the number of bytes used for writing the
  // length may decrease. If so, you then have to increase the amount of
  // padding. And so on. If you look carefully at the GCC code you will see that
  // it indeed does this in a loop, going on and on until the values stabilize.
  // We chose another solution: don't output padding inside the table like GCC
  // does, instead output it before the table.
  unsigned SizeTypes = TypeInfos.size() * TypeFormatSize;
  unsigned CallSiteTableLengthSize =
    MCAsmInfo::getULEB128Size(CallSiteTableLength);
  unsigned TTypeBaseOffset =
    sizeof(int8_t) +                            // Call site format
    CallSiteTableLengthSize +                   // Call site table length size
    CallSiteTableLength +                       // Call site table length
    SizeActions +                               // Actions size
    SizeTypes;
  unsigned TTypeBaseOffsetSize = MCAsmInfo::getULEB128Size(TTypeBaseOffset);
  unsigned TotalSize =
    sizeof(int8_t) +                            // LPStart format
    sizeof(int8_t) +                            // TType format
    (HaveTTData ? TTypeBaseOffsetSize : 0) +    // TType base offset size
    TTypeBaseOffset;                            // TType base offset
  unsigned SizeAlign = (4 - TotalSize) & 3;

  if (HaveTTData) {
    // Account for any extra padding that will be added to the call site table
    // length.
    Asm->EmitULEB128(TTypeBaseOffset, "@TType base offset", SizeAlign);
    SizeAlign = 0;
  }

  bool VerboseAsm = Asm->OutStreamer.isVerboseAsm();

  // SjLj Exception handling
  if (IsSJLJ) {
    Asm->EmitEncodingByte(dwarf::DW_EH_PE_udata4, "Call site");

    // Add extra padding if it wasn't added to the TType base offset.
    Asm->EmitULEB128(CallSiteTableLength, "Call site table length", SizeAlign);

    // Emit the landing pad site information.
    unsigned idx = 0;
    for (SmallVectorImpl<CallSiteEntry>::const_iterator
         I = CallSites.begin(), E = CallSites.end(); I != E; ++I, ++idx) {
      const CallSiteEntry &S = *I;

      // Offset of the landing pad, counted in 16-byte bundles relative to the
      // @LPStart address.
      if (VerboseAsm) {
        Asm->OutStreamer.AddComment(">> Call Site " + Twine(idx) + " <<");
        Asm->OutStreamer.AddComment("  On exception at call site "+Twine(idx));
      }
      Asm->EmitULEB128(idx);

      // Offset of the first associated action record, relative to the start of
      // the action table. This value is biased by 1 (1 indicates the start of
      // the action table), and 0 indicates that there are no actions.
      if (VerboseAsm) {
        if (S.Action == 0)
          Asm->OutStreamer.AddComment("  Action: cleanup");
        else
          Asm->OutStreamer.AddComment("  Action: " +
                                      Twine((S.Action - 1) / 2 + 1));
      }
      Asm->EmitULEB128(S.Action);
    }
  } else {
    // DWARF Exception handling
    assert(Asm->MAI->isExceptionHandlingDwarf());

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
    // supposed to throw.

    // Emit the landing pad call site table.
    Asm->EmitEncodingByte(dwarf::DW_EH_PE_udata4, "Call site");

    // Add extra padding if it wasn't added to the TType base offset.
    Asm->EmitULEB128(CallSiteTableLength, "Call site table length", SizeAlign);

    unsigned Entry = 0;
    for (SmallVectorImpl<CallSiteEntry>::const_iterator
         I = CallSites.begin(), E = CallSites.end(); I != E; ++I) {
      const CallSiteEntry &S = *I;

      MCSymbol *EHFuncBeginSym =
        Asm->GetTempSymbol("eh_func_begin", Asm->getFunctionNumber());

      MCSymbol *BeginLabel = S.BeginLabel;
      if (BeginLabel == 0)
        BeginLabel = EHFuncBeginSym;
      MCSymbol *EndLabel = S.EndLabel;
      if (EndLabel == 0)
        EndLabel = Asm->GetTempSymbol("eh_func_end", Asm->getFunctionNumber());


      // Offset of the call site relative to the previous call site, counted in
      // number of 16-byte bundles. The first call site is counted relative to
      // the start of the procedure fragment.
      if (VerboseAsm)
        Asm->OutStreamer.AddComment(">> Call Site " + Twine(++Entry) + " <<");
      Asm->EmitLabelDifference(BeginLabel, EHFuncBeginSym, 4);
      if (VerboseAsm)
        Asm->OutStreamer.AddComment(Twine("  Call between ") +
                                    BeginLabel->getName() + " and " +
                                    EndLabel->getName());
      Asm->EmitLabelDifference(EndLabel, BeginLabel, 4);

      // Offset of the landing pad, counted in 16-byte bundles relative to the
      // @LPStart address.
      if (!S.PadLabel) {
        if (VerboseAsm)
          Asm->OutStreamer.AddComment("    has no landing pad");
        Asm->OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
      } else {
        if (VerboseAsm)
          Asm->OutStreamer.AddComment(Twine("    jumps to ") +
                                      S.PadLabel->getName());
        Asm->EmitLabelDifference(S.PadLabel, EHFuncBeginSym, 4);
      }

      // Offset of the first associated action record, relative to the start of
      // the action table. This value is biased by 1 (1 indicates the start of
      // the action table), and 0 indicates that there are no actions.
      if (VerboseAsm) {
        if (S.Action == 0)
          Asm->OutStreamer.AddComment("  On action: cleanup");
        else
          Asm->OutStreamer.AddComment("  On action: " +
                                      Twine((S.Action - 1) / 2 + 1));
      }
      Asm->EmitULEB128(S.Action);
    }
  }

  // Emit the Action Table.
  int Entry = 0;
  for (SmallVectorImpl<ActionEntry>::const_iterator
         I = Actions.begin(), E = Actions.end(); I != E; ++I) {
    const ActionEntry &Action = *I;

    if (VerboseAsm) {
      // Emit comments that decode the action table.
      Asm->OutStreamer.AddComment(">> Action Record " + Twine(++Entry) + " <<");
    }

    // Type Filter
    //
    //   Used by the runtime to match the type of the thrown exception to the
    //   type of the catch clauses or the types in the exception specification.
    if (VerboseAsm) {
      if (Action.ValueForTypeID > 0)
        Asm->OutStreamer.AddComment("  Catch TypeInfo " +
                                    Twine(Action.ValueForTypeID));
      else if (Action.ValueForTypeID < 0)
        Asm->OutStreamer.AddComment("  Filter TypeInfo " +
                                    Twine(Action.ValueForTypeID));
      else
        Asm->OutStreamer.AddComment("  Cleanup");
    }
    Asm->EmitSLEB128(Action.ValueForTypeID);

    // Action Record
    //
    //   Self-relative signed displacement in bytes of the next action record,
    //   or 0 if there is no next action record.
    if (VerboseAsm) {
      if (Action.NextAction == 0) {
        Asm->OutStreamer.AddComment("  No further actions");
      } else {
        unsigned NextAction = Entry + (Action.NextAction + 1) / 2;
        Asm->OutStreamer.AddComment("  Continue to action "+Twine(NextAction));
      }
    }
    Asm->EmitSLEB128(Action.NextAction);
  }

  // Emit the Catch TypeInfos.
  if (VerboseAsm && !TypeInfos.empty()) {
    Asm->OutStreamer.AddComment(">> Catch TypeInfos <<");
    Asm->OutStreamer.AddBlankLine();
    Entry = TypeInfos.size();
  }

  for (std::vector<const GlobalVariable *>::const_reverse_iterator
         I = TypeInfos.rbegin(), E = TypeInfos.rend(); I != E; ++I) {
    const GlobalVariable *GV = *I;
    if (VerboseAsm)
      Asm->OutStreamer.AddComment("TypeInfo " + Twine(Entry--));
    if (GV)
      Asm->EmitReference(GV, TTypeEncoding);
    else
      Asm->OutStreamer.EmitIntValue(0,Asm->GetSizeOfEncodedValue(TTypeEncoding),
                                    0);
  }

  // Emit the Exception Specifications.
  if (VerboseAsm && !FilterIds.empty()) {
    Asm->OutStreamer.AddComment(">> Filter TypeInfos <<");
    Asm->OutStreamer.AddBlankLine();
    Entry = 0;
  }
  for (std::vector<unsigned>::const_iterator
         I = FilterIds.begin(), E = FilterIds.end(); I < E; ++I) {
    unsigned TypeID = *I;
    if (VerboseAsm) {
      --Entry;
      if (TypeID != 0)
        Asm->OutStreamer.AddComment("FilterInfo " + Twine(Entry));
    }

    Asm->EmitULEB128(TypeID);
  }

  Asm->EmitAlignment(2);
}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfException::EndModule() {
  llvm_unreachable("Should be implemented");
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void DwarfException::BeginFunction(const MachineFunction *MF) {
  llvm_unreachable("Should be implemented");
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfException::EndFunction() {
  llvm_unreachable("Should be implemented");
}
