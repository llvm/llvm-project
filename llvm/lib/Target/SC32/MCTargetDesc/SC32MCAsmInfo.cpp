#include "SC32MCAsmInfo.h"

using namespace llvm;

SC32MCAsmInfo::SC32MCAsmInfo() {
  HasSingleParameterDotFile = false;
  HasDotTypeDotSizeDirective = false;
  CommentString = ";";
}
