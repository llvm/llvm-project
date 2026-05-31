// AIR doesn't actually use registers, but this gets the boilerplate code
// generated through tablegen.
let Namespace = "air" in {
def AIRReg : Register<"air">;
def AIRClass : RegisterClass<"air", [i32], 32, (add AIRReg)>;
}

class AIRInst : Instruction {
  let Namespace = "air";
  let DecoderNamespace = "air";

  dag OutOperandList = (outs);
  dag InOperandList = (ins);
  let AsmString = "dummy";
  let Pattern = [];
}

def DummyInst : AIRInst;
