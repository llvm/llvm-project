include "llvm/Target/Target.td"

include "GenpayRegisterInfo.td"
include "GenpayInstrInfo.td"

def GenpayInstrInfo : InstrInfo;

def GenpayAsmWriter : AsmWriter {
  let AsmWriterClassName = "InstPrinter";
  let PassSubtarget = 0;
  let Variant = 0;
}

def Genpay : Target {
  let InstructionSet = GenpayInstrInfo;
  let AssemblyWriters = [GenpayAsmWriter];
}
