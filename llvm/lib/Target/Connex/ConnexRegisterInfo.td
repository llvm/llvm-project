//===-- ConnexRegisterInfo.td - Connex Register defs -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Connex register file
//===----------------------------------------------------------------------===//


// Scalar??? registers are identified with 4-bit ID numbers.
// Ri - 16-bit integer registers
class Ri<bits<16> Enc, string n> : Register<n> {
  let Namespace = "Connex";
  let HWEncoding = Enc;
}

// Scalar registers of integer type
def  R0 : Ri< 0, "r0">, DwarfRegNum<[0]>;
def  R1 : Ri< 1, "r1">, DwarfRegNum<[1]>;
def  R2 : Ri< 2, "r2">, DwarfRegNum<[2]>;
def  R3 : Ri< 3, "r3">, DwarfRegNum<[3]>;
def  R4 : Ri< 4, "r4">, DwarfRegNum<[4]>;
def  R5 : Ri< 5, "r5">, DwarfRegNum<[5]>;
def  R6 : Ri< 6, "r6">, DwarfRegNum<[6]>;
def  R7 : Ri< 7, "r7">, DwarfRegNum<[7]>;
def  R8 : Ri< 8, "r8">, DwarfRegNum<[8]>;
def  R9 : Ri< 9, "r9">, DwarfRegNum<[9]>;
def R10 : Ri<10, "r10">, DwarfRegNum<[10]>;
def R11 : Ri<11, "r11">, DwarfRegNum<[11]>;
def R12 : Ri<12, "r12">, DwarfRegNum<[12]>;
def R13 : Ri<13, "r13">, DwarfRegNum<[13]>;
def R14 : Ri<14, "r14">, DwarfRegNum<[14]>;
def R15 : Ri<15, "r15">, DwarfRegNum<[15]>;
def R16 : Ri<16, "r16">, DwarfRegNum<[16]>;
def R17 : Ri<17, "r17">, DwarfRegNum<[17]>;
def R18 : Ri<18, "r18">, DwarfRegNum<[18]>;
def R19 : Ri<19, "r19">, DwarfRegNum<[19]>;
def R20 : Ri<20, "r20">, DwarfRegNum<[20]>;
def R21 : Ri<21, "r21">, DwarfRegNum<[21]>;
def R22 : Ri<22, "r22">, DwarfRegNum<[22]>;
def R23 : Ri<23, "r23">, DwarfRegNum<[23]>;
def R24 : Ri<24, "r24">, DwarfRegNum<[24]>;
def R25 : Ri<25, "r25">, DwarfRegNum<[25]>;
def R26 : Ri<26, "r26">, DwarfRegNum<[26]>;
def R27 : Ri<27, "r27">, DwarfRegNum<[27]>;
def R28 : Ri<28, "r28">, DwarfRegNum<[28]>;
def R29 : Ri<29, "r29">, DwarfRegNum<[29]>;
def R30 : Ri<30, "r30">, DwarfRegNum<[30]>;
def R31 : Ri<30, "r31">, DwarfRegNum<[31]>;








// Register classes.

//===----------------------------------------------------------------------===//
//  Declarations that describe the Connex register file
//===----------------------------------------------------------------------===//
let Namespace = "Connex" in {


// Subregister indices. - See also llvm/lib/Target/X86/X86RegisterInfo.td

// From llvm.org/svn/llvm-project/llvm/trunk/include/llvm/Target/Target.td


/*
// SubRegIndex - Use instances of SubRegIndex to identify subregisters.
//class SubRegIndex<int size, int offset = 0>
*/

// 16 lanes Connex
// This is for 16-bits subregisters (a Connex vector register has type 16 x i16)
//
def sub_16_00    : SubRegIndex<16,    0>;
def sub_16_01    : SubRegIndex<16,   16>;
def sub_16_02    : SubRegIndex<16,   32>;
def sub_16_03    : SubRegIndex<16,   48>;
def sub_16_04    : SubRegIndex<16,   64>;
def sub_16_05    : SubRegIndex<16,   80>;
def sub_16_06    : SubRegIndex<16,   96>;
def sub_16_07    : SubRegIndex<16,  112>;
def sub_16_08    : SubRegIndex<16,  128>;
def sub_16_09    : SubRegIndex<16,  144>;
def sub_16_10    : SubRegIndex<16,  160>;
def sub_16_11    : SubRegIndex<16,  176>;
def sub_16_12    : SubRegIndex<16,  192>;
def sub_16_13    : SubRegIndex<16,  208>;
def sub_16_14    : SubRegIndex<16,  224>;
def sub_16_15    : SubRegIndex<16,  240>;
} // END: let Namespace = "Connex" in






// Register Operands.

class ConnexAsmRegOperand : AsmOperandClass {
  let ParserMethod = "parseAnyRegister";
}

def ConnexVectorAsmOperand : ConnexAsmRegOperand {
  let Name = "ConnexVectorAsmReg";
}




class ConnexRegWithSubRegs<bits<16> Enc, string n, list<Register> subregs>
    : RegisterWithSubRegs<n, subregs> {
  let HWEncoding = Enc;
  let Namespace = "Connex";
}

// We have banks of 32 registers each.
class ConnexVectorElementReg<bits<16> Enc, string n> : Register<n> {
  let HWEncoding = Enc;
  let Namespace = "Connex";
}

class SUBREGS<bits<16> Enc, string n> : ConnexVectorElementReg<Enc, n>;

// 8 lanes Connex, with registers of 8 x 16-bits (type v8i16)
class VectorRegPacked16bits<bits<16> Enc, string n, list<Register> subregs>
            : ConnexRegWithSubRegs<Enc, n, subregs> {
  // For 16-bit subregisters:
  let SubRegIndices = [ sub_16_00, sub_16_01, sub_16_02, sub_16_03,
                        sub_16_04, sub_16_05, sub_16_06, sub_16_07];
  let CoveredBySubRegs = 1;
}


//!!!!TODO: implement aliasing between 64, 16 and 32 subregs - NOT sure if required


let Namespace = "Connex" in {
  // 8 lanes vector registers (type v8i16 - 8 x 16-bits)
  // 32 vector registers
  foreach RegId = 0-31 in
    foreach I = 0-7 in
      def SR16b_#RegId#_#I : SUBREGS<I, "sr16_"#RegId#"_"#I>,
                                     DwarfRegNum<[!add(I, 2048)]>;

  // 8 lanes vector registers (type v8i16)
  foreach RegId = 0-31 in
    def Wh#RegId : VectorRegPacked16bits<0, "R("#RegId#")",
                                       [
                                        !cast<SUBREGS>("SR16b_"#RegId#"_0"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_1"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_2"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_3"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_4"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_5"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_6"),
                                        !cast<SUBREGS>("SR16b_"#RegId#"_7")]>,
                        DwarfRegNum<[!add(RegId, 32)]>;
} // END: let Namespace = "Connex" in



/*
From http://llvm.org/docs/WritingAnLLVMBackend.html#defining-a-register-class:
 <<To define a RegisterClass, use the following 4 arguments:
  - The first argument of the definition is the name of the namespace.
  - The second argument is a list of ValueType register type values that are
       defined in include/llvm/CodeGen/ValueTypes.td.
    Defined values include integer types (such as i16, i32, and i1 for Boolean),
      floating-point types (f32, f64), and vector types (for example, v8i16 for
      an 8 x i16 vector).
    All registers in a RegisterClass must have the same ValueType, but some
      registers may store vector data in different configurations.
    For example a register that can process a 128-bit vector may be able to
      handle 16 8-bit integer elements, 8 16-bit integers, 4 32-bit integers,
      and so on.
  - The third argument of the RegisterClass definition specifies the alignment
     required of the registers when they are stored or loaded to memory.
     (Alex: the alignment is expressed in number of bits of the register)
  - The final argument, regList, specifies which registers are in this class.
    If an alternative allocation order method is not specified, then regList
      also defines the order of allocation used by the register allocator.
    Besides simply listing registers with (add R0, R1, ...), more advanced set
     operators are available. See include/llvm/Target/Target.td for more
     information.>>
*/




// These are taken from Mips - in the file ConnexRegisterInfo.td these registers
//   (RegisterClass) are defined with the other register altogether - same
//   namespace, same target


// TODO: currently I use v8i16 --> vector alignment is 128 bits; but CVL
//   (Connex vector length) can be arbitrary --> MEGA-TODO: find a work-around
//   for this
def VectorH: RegisterClass<"Connex", [v8i16 /*, v128f16 */],
                           // NUM_REGS = 32
                           128, // Works, but 256-bytes alignment wastes too
                                // much: 2048, NOTE: using 0 for alignment gives
                                // an unexplainable "Stack dump"
                           //64, // Works, but 256-bytes alignment wastes too
                                 // much: 2048, NOTE: using 0 for alignment
                                 // gives an unexplainable "Stack dump"
                                 //NUM_REGS=64
                           // TODO_CHANGE_BACKEND
                           // NUM_REGS = 32
                           (sequence "Wh%u", 0, 31)>;


def VectorHOpnd : RegisterOperand<VectorH> {
  let ParserMatchClass = ConnexVectorAsmOperand;
}



// NUM_REGS = 32
// 32 registers of 128 elements, 1 bit each (v128i1)
foreach RegId = 0-31 in
    def BoolMask#RegId : ConnexVectorElementReg<0, "BoolMask"#RegId>,
                                                DwarfRegNum<[!add(RegId, 10)]>;

// Inspired from llvm/lib/Target/X86/X86RegisterInfo.td:
// NUM_REGS = 32
def BoolMask: RegisterClass<"Connex", [v8i1], 32,
                            (sequence "BoolMask%u", 0, 31)>;
def BoolMaskOpnd : RegisterOperand<BoolMask> {
  let ParserMatchClass = ConnexVectorAsmOperand;
}









// The GPR class of scalar registers

def GPR : RegisterClass<"Connex", [i64], 64, (add (sequence "R%u", 0, 31))>;

def GPR64AsmOperand : ConnexAsmRegOperand {
  let Name = "GPR64AsmReg";
}

// Taken from MipsRegisterInfo.td
def GPR64Opnd : RegisterOperand<GPR> {
  let ParserMatchClass = GPR64AsmOperand;
}

