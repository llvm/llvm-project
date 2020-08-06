#include "P2ELFStreamer.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/FormattedStream.h"

#include "P2MCTargetDesc.h"

namespace llvm {

    P2ELFStreamer::P2ELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI) : P2TargetStreamer(S) {
        MCAssembler &MCA = getStreamer().getAssembler();
        unsigned EFlags = MCA.getELFHeaderEFlags();
        MCA.setELFHeaderEFlags(EFlags);
    }

} // end namespace llvm
