// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s -DTEST1 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1
// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s -DTEST2 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2
// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s -DTEST3 2>&1 | FileCheck %s --check-prefix=CHECK-TEST3

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
    let InstructionSet = archInstrInfo;
}

#ifdef TEST1
// CHECK-TEST1: [[#@LINE+1]]:5: error: foo: Size is 16 bits, but Inst bits beyond that are not zero/unset
def foo : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let Size = 2;
  field bits<24> Inst;
  field bits<24> SoftFail = 0;
  bits<8> factor;
  let Inst{15...8} = factor{7...0};
  let Inst{20} = 1;
}
#endif

#ifdef TEST2
// CHECK-TEST2: [[#@LINE+1]]:5: error: foo: Size is 16 bits, but SoftFail bits beyond that are not zero/unset
def foo : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let Size = 2;
  field bits<24> Inst;
  field bits<24> SoftFail = 0;
  bits<8> factor;
  let Inst{15...8} = factor{7...0};
  let Inst{23...16} = 0;
  let SoftFail{20} = 1;
}
#endif

#ifdef TEST3
// CHECK-TEST3: [[#@LINE+1]]:5: error: bar: Size is 16 bits, but Inst specifies only 8 bits
def bar : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let Size = 2;
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  bits<4> factor;
  let Inst{4...1} = factor{3...0};
}
#endif
