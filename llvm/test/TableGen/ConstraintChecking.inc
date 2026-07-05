include "llvm/Target/Target.td"

def TestTarget : Target;

class Encoding : Instruction {
  field bits<8> Inst;
}

class TestReg<string name, bits<1> enc> : Register<name, []> {
    let HWEncoding{15-1} = 0;
    let HWEncoding{0} = enc;
}

def R0 : TestReg<"R0", 0>;
def R1 : TestReg<"R1", 1>;
def Reg : RegisterClass<"TestTarget", [i32], 32, (sequence "R%d", 0, 1)>;

class TestInstructionWithConstraints<string cstr> : Encoding {
  dag OutOperandList = (outs Reg:$dest1, Reg:$dest2);
  dag InOperandList = (ins Reg:$src1, Reg:$src2);
  string AsmString = "mnemonic $dest1, $dest2, $src1, $src2";
  string AsmVariantName = "";
  let Constraints = cstr;
  field bits<1> dest1;
  field bits<1> dest2;
  field bits<1> src1;
  field bits<1> src2;
  let Inst{7-4} = 0b1010;
  let Inst{3} = dest1{0};
  let Inst{2} = dest2{0};
  let Inst{1} = src1{0};
  let Inst{0} = src2{0};
}
