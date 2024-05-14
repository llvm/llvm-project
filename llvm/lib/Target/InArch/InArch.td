include "llvm/Target/Target.td"
include "InArchRegisterInfo.td"
include "InArchCallingConv.td"
include "InArchInstrInfo.td"

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def InArchInstrInfo : InstrInfo;
def InArch : Target {
  let InstructionSet = InArchInstrInfo;
}