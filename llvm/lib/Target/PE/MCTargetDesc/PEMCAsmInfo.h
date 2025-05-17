#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/TargetRegistry.h"

namespace llvm{

class Triple;

class PEMCAsmInfo : public MCAsmInfoELF {
public:
    explicit PEMCAsmInfo(const Triple &TT);              
};
}