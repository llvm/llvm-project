// HelloMinimal.td
class Instruction {
  string AsmString = "";
}

def NOP : Instruction {
  let AsmString = "nop";
}
