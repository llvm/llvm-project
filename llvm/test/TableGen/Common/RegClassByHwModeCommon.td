include "llvm/Target/Target.td"

class MyReg<string n> : Register<n> {
  let Namespace = "MyTarget";
}

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let Size = size;
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
def X2 : MyReg<"x2">;
def X3 : MyReg<"x3">;
def X4 : MyReg<"x4">;
def X5 : MyReg<"x5">;
def X6 : MyReg<"x6">;
def XRegs : RegisterClass<"MyTarget", [i64], 64, (add X0, X1, X2, X3, X4, X5, X6)>;
def Y0 : MyReg<"y0">;
def Y1 : MyReg<"y1">;
def Y2 : MyReg<"y2">;
def Y3 : MyReg<"y3">;
def Y4 : MyReg<"y4">;
def Y5 : MyReg<"y5">;
def Y6 : MyReg<"y6">;
def YRegs : RegisterClass<"MyTarget", [i64], 64, (add Y0, Y1, Y2, Y3, Y4, Y5, Y6)>;

class TestInstruction : Instruction {
  let Size = 2;
  let Namespace = "MyTarget";
  let hasSideEffects = false;
  let hasExtraSrcRegAllocReq = false;
  let hasExtraDefRegAllocReq = false;

  field bits<16> Inst;
  bits<3> dst;
  bits<3> src;
  bits<3> opcode;

  let Inst{2-0} = dst;
  let Inst{5-3} = src;
  let Inst{7-5} = opcode;
}
