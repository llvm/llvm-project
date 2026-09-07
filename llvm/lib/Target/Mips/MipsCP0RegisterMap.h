//===- MipsCP0RegisterMap.h - Co-processor register names -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the mapping between MIPS CP0 register names and their
/// (register, select) encodings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSCP0REGMAP_H
#define LLVM_LIB_TARGET_MIPS_MIPSCP0REGMAP_H
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

struct CP0SelRegister {
  StringRef Name;
  int RegNum;
  int Select;
};

class MipsCP0SelMap {
  DenseMap<unsigned, int> EncIndexMap;
  StringMap<int> NameIndexMap;

public:
  int getEncIndexMap(unsigned Reg) {
    auto It = EncIndexMap.find(Reg);
    return It != EncIndexMap.end() ? It->second : -1;
  }

  int getNameIndexMap(StringRef Name) {
    auto It = NameIndexMap.find(Name);
    return It != NameIndexMap.end() ? It->second : -1;
  }

  MipsCP0SelMap() {
    static const struct CP0SelRegister Regs[] = {{"index", 0, 0},
                                                 {"mvpcontrol", 0, 1},
                                                 {"mvpconf0", 0, 2},
                                                 {"mvpconf1", 0, 3},
                                                 {"vpcontrol", 0, 4},
                                                 {"random", 1, 0},
                                                 {"vpecontrol", 1, 1},
                                                 {"vpeconf0", 1, 2},
                                                 {"vpeconf1", 1, 3},
                                                 {"yqmask", 1, 4},
                                                 {"vpeschedule", 1, 5},
                                                 {"vpeschefback", 1, 6},
                                                 {"vpeopt", 1, 7},
                                                 {"entrylo0", 2, 0},
                                                 {"tcstatus", 2, 1},
                                                 {"tcbind", 2, 2},
                                                 {"tcrestart", 2, 3},
                                                 {"tchalt", 2, 4},
                                                 {"tccontext", 2, 5},
                                                 {"tcschedule", 2, 6},
                                                 {"tcschefback", 2, 7},
                                                 {"entrylo1", 3, 0},
                                                 {"globalnumber", 3, 1},
                                                 {"tcopt", 3, 7},
                                                 {"context", 4, 0},
                                                 {"contextconfig", 4, 1},
                                                 {"userlocal", 4, 2},
                                                 {"xcontextconfig", 4, 3},
                                                 {"debugcontextid", 4, 4},
                                                 {"memorymapid", 4, 5},
                                                 {"pagemask", 5, 0},
                                                 {"pagegrain", 5, 1},
                                                 {"segctl0", 5, 2},
                                                 {"segctl1", 5, 3},
                                                 {"segctl2", 5, 4},
                                                 {"pwbase", 5, 5},
                                                 {"pwfield", 5, 6},
                                                 {"pwsize", 5, 7},
                                                 {"wired", 6, 0},
                                                 {"srsconf0", 6, 1},
                                                 {"srsconf1", 6, 2},
                                                 {"srsconf2", 6, 3},
                                                 {"srsconf3", 6, 4},
                                                 {"srsconf4", 6, 5},
                                                 {"pwctl", 6, 6},
                                                 {"hwrena", 7, 0},
                                                 {"badvaddr", 8, 0},
                                                 {"badinst", 8, 1},
                                                 {"badinstrp", 8, 2},
                                                 {"badinstrx", 8, 3},
                                                 {"count", 9, 0},
                                                 {"entryhi", 10, 0},
                                                 {"guestctl1", 10, 4},
                                                 {"guestctl2", 10, 5},
                                                 {"guestctl3", 10, 6},
                                                 {"compare", 11, 0},
                                                 {"guestctl0ext", 11, 4},
                                                 {"status", 12, 0},
                                                 {"intctl", 12, 1},
                                                 {"srsctl", 12, 2},
                                                 {"srsmap", 12, 3},
                                                 {"view_ipl", 12, 4},
                                                 {"srsmap2", 12, 5},
                                                 {"guestctl0", 12, 6},
                                                 {"gtoffset", 12, 7},
                                                 {"cause", 13, 0},
                                                 {"view_ripl", 13, 4},
                                                 {"nestedexc", 13, 5},
                                                 {"epc", 14, 0},
                                                 {"nestedepc", 14, 2},
                                                 {"prid", 15, 0},
                                                 {"ebase", 15, 1},
                                                 {"cdmmbase", 15, 2},
                                                 {"cmgcrbase", 15, 3},
                                                 {"bevva", 15, 4},
                                                 {"config", 16, 0},
                                                 {"config1", 16, 1},
                                                 {"config2", 16, 2},
                                                 {"config3", 16, 3},
                                                 {"config4", 16, 4},
                                                 {"config5", 16, 5},
                                                 {"lladdr", 17, 0},
                                                 {"maar", 17, 1},
                                                 {"maari", 17, 2},
                                                 {"watchlo0", 18, 0},
                                                 {"watchlo1", 18, 1},
                                                 {"watchlo2", 18, 2},
                                                 {"watchlo3", 18, 3},
                                                 {"watchlo4", 18, 4},
                                                 {"watchlo5", 18, 5},
                                                 {"watchlo6", 18, 6},
                                                 {"watchlo7", 18, 7},
                                                 {"watchlo8", 18, 8},
                                                 {"watchlo9", 18, 9},
                                                 {"watchlo10", 18, 10},
                                                 {"watchlo11", 18, 11},
                                                 {"watchlo12", 18, 12},
                                                 {"watchlo13", 18, 13},
                                                 {"watchlo14", 18, 14},
                                                 {"watchlo15", 18, 15},
                                                 {"watchhi0", 19, 0},
                                                 {"watchhi1", 19, 1},
                                                 {"watchhi2", 19, 2},
                                                 {"watchhi3", 19, 3},
                                                 {"watchhi4", 19, 4},
                                                 {"watchhi5", 19, 5},
                                                 {"watchhi6", 19, 6},
                                                 {"watchhi7", 19, 7},
                                                 {"watchhi8", 19, 8},
                                                 {"watchhi9", 19, 9},
                                                 {"watchhi10", 19, 10},
                                                 {"watchhi11", 19, 11},
                                                 {"watchhi12", 19, 12},
                                                 {"watchhi13", 19, 13},
                                                 {"watchhi14", 19, 14},
                                                 {"watchhi15", 19, 15},
                                                 {"xcontext", 20, 0},
                                                 {"debug", 23, 0},
                                                 {"tracecontrol", 23, 1},
                                                 {"tracecontrol2", 23, 2},
                                                 {"usertracedata1", 23, 3},
                                                 {"traceibpc", 23, 4},
                                                 {"tracedbpc", 23, 5},
                                                 {"debug2", 23, 6},
                                                 {"depc", 24, 0},
                                                 {"tracecontrol3", 24, 2},
                                                 {"usertracedata2", 24, 3},
                                                 {"perfctl0", 25, 0},
                                                 {"perfcnt0", 25, 1},
                                                 {"perfctl1", 25, 2},
                                                 {"perfcnt1", 25, 3},
                                                 {"perfctl2", 25, 4},
                                                 {"perfcnt2", 25, 5},
                                                 {"perfctl3", 25, 6},
                                                 {"perfcnt3", 25, 7},
                                                 {"perfctl4", 25, 8},
                                                 {"perfcnt4", 25, 9},
                                                 {"perfctl5", 25, 10},
                                                 {"perfcnt5", 25, 11},
                                                 {"perfctl6", 25, 12},
                                                 {"perfcnt6", 25, 13},
                                                 {"perfctl7", 25, 14},
                                                 {"perfcnt7", 25, 15},
                                                 {"errctl", 26, 0},
                                                 {"cacheerr", 27, 0},
                                                 {"itaglo", 28, 0},
                                                 {"idatalo", 28, 1},
                                                 {"dtaglo", 28, 2},
                                                 {"ddatalo", 28, 3},
                                                 {"itaghi", 29, 0},
                                                 {"idatahi", 29, 1},
                                                 {"dtaghi", 29, 2},
                                                 {"ddatahi", 29, 3},
                                                 {"errorepc", 30, 0},
                                                 {"desave", 31, 0},
                                                 {"kscratch1", 31, 2},
                                                 {"kscratch2", 31, 3},
                                                 {"kscratch3", 31, 4},
                                                 {"kscratch4", 31, 5},
                                                 {"kscratch5", 31, 6},
                                                 {"kscratch6", 31, 7}};
    int Index = 0;
    for (auto Reg : Regs) {
      NameIndexMap[Reg.Name] = Index;
      unsigned RegEnc = (Reg.RegNum << 5) | Reg.Select;
      EncIndexMap[RegEnc] = Index++;
    }
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_MIPSCP0REGMAP_H
