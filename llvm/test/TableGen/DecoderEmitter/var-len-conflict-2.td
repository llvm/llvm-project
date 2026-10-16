// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s -o /dev/null 2>&1 \
// RUN:   | FileCheck --strict-whitespace %s

include "llvm/Target/Target.td"

class I : Instruction {
  let InOperandList = (ins i32imm:$op, i32imm:$op2, i32imm:$op3);
  let OutOperandList = (outs);
}

// Check pretty printing of decoding conflicts.

// CHECK:      {{^}}Decoding Conflict:
// CHECK-NEXT: {{^}}                    ........
// CHECK-NEXT: {{^}}            ..............11
// CHECK-NEXT: {{^}}            .......0......11
// CHECK-NEXT: {{^}}            _______0______11  I16_0
// CHECK-NEXT: {{^}}            _______0______11  I16_1
// CHECK-NEXT: {{^}}    _______0_______0______11  I24_0

def I8_0  : I { dag Inst = (descend (operand "$op", 6), 0b00); }
def I8_1  : I { dag Inst = (descend (operand "$op", 6), 0b01); }
def I16_0 : I { dag Inst = (descend (operand "$op2", 7), 0b0,
                                    (operand "$op", 6), 0b11); }
def I16_1 : I { dag Inst = (descend (operand "$op2", 7), 0b0,
                                    (operand "$op", 6), 0b11); }
def I24_0 : I { dag Inst = (descend (operand "$op3", 7), 0b0,
                                    (operand "$op2", 7), 0b0,
                                    (operand "$op", 6), 0b11); }

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
