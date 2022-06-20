// DXIL doesn't actually use registers, but this gets the boilerplate code
// generated through tablegen.
let Namespace = "DXIL" in {
def DXIL : Register<"DXIL">;
def DXILClass : RegisterClass<"DXIL", [i32], 32, (add DXIL)>;
}

class DXILInst : Instruction {
  let Namespace = "DXIL";
  let DecoderNamespace = "DXIL";

  dag OutOperandList = (outs);
  dag InOperandList =  (ins);
  let AsmString = "dummy";
  let Pattern = [];
}

def DummyInst : DXILInst;
