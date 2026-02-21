#ifndef MISC_H_INCLUDED__
#define MISC_H_INCLUDED__

#include <string>

#ifndef MAXLEN_STR
  #define MAXLEN_STR 8192
#endif

//#include "string_extras.h"
// We could use also boost::algorithm::ends_with() and starts_with()
// From
//  stackoverflow.com/questions/8095088/how-to-check-string-start-in-c/8095132
static bool startsWith(const std::string &haystack, const std::string &needle) {
  // We avoid making a copy of your string with strstr() or searching more
  //   than at index 0
  return needle.length() <= haystack.length() &&
         std::equal(needle.begin(), needle.end(), haystack.begin());
}

// From
// stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
static bool endsWith(const std::string &s, const std::string &suffix) {
  return s.rfind(suffix) == (s.size() - suffix.size());
}
static inline bool ends_with(std::string const &value,
                             std::string const &ending) {
  // TODO: check if OK
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static std::string stringScanf(std::string aText, char *formatStr) {
  char strTmp[MAXLEN_STR];

  assert(formatStr[0] == '%' && formatStr[1] == 's');

  // TODO: implement without sscanf()
  // Maybe use as said at
  //  https://stackoverflow.com/questions/6104821/c-equivalent-of-sscanf:
  //   "The formatting isn't as easy but check out stringstream.
  //    See also istringstream and ostringstream for input and
  //    output buffers formatting."
  sscanf(aText.c_str(), formatStr, strTmp);

  return std::string(strTmp);
}

static std::string stringPrintf(char *formatStr, void *aPtr) {
  char strTmp[MAXLEN_STR];

  // TODO: implement without sprintf()
  sprintf(strTmp, formatStr, aPtr);

  return std::string(strTmp);
}
static std::string stringPrintf(char *formatStr, long aVal) {
  char strTmp[MAXLEN_STR];

  // TODO: implement without sprintf()
  sprintf(strTmp, formatStr, aVal);

  return std::string(strTmp);
}

#ifdef INCLUDE_SUNIT_DUMP

#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Inspired from SystemZHazardRecognizer.cpp

#ifndef NDEBUG // Debug output

// The SUnit (Scheduling Unit) class no longer has the dump() method,
//   so we create a helper method for it here.
// Inspired from SystemZHazardRecognizer.h

/// Resolves and cache a resolved scheduling class for an SUnit.
static const MCSchedClassDesc *getSchedClass(SUnit *SU) {
  if (!SU->SchedClass) { // && SchedModel->hasInstrSchedModel()
    return NULL;
    // TODO: SU->SchedClass = SchedModel->resolveSchedClass(SU->getInstr());
  }

  return SU->SchedClass;
}

static void dumpSU(llvm::SUnit *SU, raw_ostream &OS) {
  OS << "SU(" << SU->NodeNum << "):";
  // OS << TII->getName(SU->getInstr()->getOpcode());
  OS << SU->getInstr()->getOpcode();

  const MCSchedClassDesc *SC = getSchedClass(SU);
  if (!SC->isValid())
    return;

  /*
  // TODO: make this compile

  for (TargetSchedModel::ProcResIter
         PI = SchedModel->getWriteProcResBegin(SC),
         PE = SchedModel->getWriteProcResEnd(SC); PI != PE; ++PI) {
    const MCProcResourceDesc &PRD =
      *SchedModel->getProcResource(PI->ProcResourceIdx);
    std::string FU(PRD.Name);
    // trim e.g. Z13_FXaUnit -> FXa
    FU = FU.substr(FU.find("_") + 1);
    size_t Pos = FU.find("Unit");
    if (Pos != std::string::npos)
      FU.resize(Pos);
    if (FU == "LS") // LSUnit -> LSU
      FU = "LSU";
    OS << "/" << FU;

    if (PI->Cycles > 1)
      OS << "(" << PI->Cycles << "cyc)";
  }
  */

  if (SC->NumMicroOps > 1)
    OS << "/" << SC->NumMicroOps << "uops";
  if (SC->BeginGroup && SC->EndGroup)
    OS << "/GroupsAlone";
  else if (SC->BeginGroup)
    OS << "/BeginsGroup";
  else if (SC->EndGroup)
    OS << "/EndsGroup";
  if (SU->isUnbuffered)
    OS << "/Unbuffered";
  /*
  // TODO: make this compile
  if (has4RegOps(SU->getInstr()))
    OS << "/4RegOps";
  */
}
#endif // #ifndef NDEBUG

#endif // INCLUDED_SUNIT_DUMP

#endif // MISC_H_INCLUDED__
