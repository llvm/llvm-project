#include "SC32MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

SC32MCAsmInfo::SC32MCAsmInfo() {
  HasSingleParameterDotFile = false;
  HasDotTypeDotSizeDirective = false;
  CommentString = ";";
}

void SC32MCAsmInfo::printSwitchToSection(const MCSection &Section,
                                         uint32_t Subsection, const Triple &T,
                                         raw_ostream &OS) const {
  OS << "\tSECTION " << Section.getName() << '\n';
}
