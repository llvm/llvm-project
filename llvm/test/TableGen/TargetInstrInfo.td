// This test describes how we eventually want to describe instructions in
// the target independent code generators.
// RUN: tblgen %s

// Target indep stuff.
class Instruction {   // Would have other stuff eventually
  bit isTwoAddress = 0;
  string AssemblyString;
}
class RegisterClass;

class RTLNode;

def ops;                 // Marker for operand list.

// Various expressions used in RTL descriptions.
def imm8    : RTLNode;
def imm32   : RTLNode;
def addr    : RTLNode;

def set     : RTLNode;
def signext : RTLNode;
def zeroext : RTLNode;
def plus    : RTLNode;
def and     : RTLNode;
def xor     : RTLNode;
def shl     : RTLNode;
def load    : RTLNode;
def store   : RTLNode;
def unspec  : RTLNode;

// Start of X86 specific stuff.

def R8  : RegisterClass;
def R16 : RegisterClass;
def R32 : RegisterClass;

def CL;  // As are currently defined
def AL;
def AX;
def EDX;

class Format<bits<5> val> {
  bits<5> Value = val;
}

def Pseudo     : Format<0>; def RawFrm     : Format<1>;
def AddRegFrm  : Format<2>; def MRMDestReg : Format<3>;
def MRMDestMem : Format<4>; def MRMSrcReg  : Format<5>;
def MRMSrcMem  : Format<6>;
def MRM0r  : Format<16>; def MRM1r  : Format<17>; def MRM2r  : Format<18>;
def MRM3r  : Format<19>; def MRM4r  : Format<20>; def MRM5r  : Format<21>;
def MRM6r  : Format<22>; def MRM7r  : Format<23>;
def MRM0m  : Format<24>; def MRM1m  : Format<25>; def MRM2m  : Format<26>;
def MRM3m  : Format<27>; def MRM4m  : Format<28>; def MRM5m  : Format<29>;
def MRM6m  : Format<30>; def MRM7m  : Format<31>;


class Inst<dag opnds, string asmstr, bits<8> opcode,
           Format f, list<dag> rtl> : Instruction {
  dag Operands = opnds;
  string AssemblyString = asmstr;
  bits<8> Opcode = opcode;
  Format Format = f;
  list<dag> RTL = rtl;
}


// Start of instruction definitions, the real point of this file.
//
// Note that these patterns show a couple of important things:
//  1. The order and contents of the operands of the MachineInstr are
//     described here.  Eventually we can do away with this when everything
//     is generated from the description.
//  2. The asm string is captured here, which makes it possible to get rid of
//     a ton of hacks in the various printers and a bunch of flags.
//  3. Target specific properties (e.g. Format) can still be captured as
//     needed.
//  4. We capture the behavior of the instruction with a simplified RTL-like
//     expression.
//  5. The use/def properties for each operand are automatically inferred from
//     the pattern.
//  6. Address expressions should become first-class entities.

// Simple copy instruction.  isMoveInstr could easily be inferred from this,
// as could TargetRegisterInfo::copyRegToReg.
def MOV8rr : Inst<(ops R8:$dst, R8:$src),
                  "mov $dst, $src", 0x88, MRMDestReg,
                  [(set R8:$dst, R8:$src)]>;

// Simple immediate initialization.
def MOV8ri : Inst<(ops R8:$dst, imm8:$src),
                  "mov $dst, $src", 0xB0, AddRegFrm,
                  [(set R8:$dst, imm8:$src)]>;

// Two address instructions are described as three-addr instructions, with
// the special target-independent isTwoAddress flag set.  The asm pattern
// should not refer to the $src1, this would be enforced by the
// TargetInstrInfo tablegen backend.
let isTwoAddress = 1 in
def AND8rr : Inst<(ops R8:$dst, R8:$src1, R8:$src2),
                  "and $dst, $src2", 0x20, MRMDestReg,
                  [(set R8:$dst, (and R8:$src1, R8:$src2))]>;

// Instructions that have explicit uses/defs make them explicit in the RTL.
// Instructions that need extra stuff emitted in the assembly can, trivially.
let isTwoAddress = 1 in
def SHL32rCL : Inst<(ops R32:$dst, R32:$src),
                  "shl $dst, CL", 0xD2, MRM4r,
                  [(set R32:$dst, (shl R32:$src, CL))]>;

// The RTL list is a list, allowing complex instructions to be defined easily.
// Temporary 'internal' registers can be used to break instructions appart.
let isTwoAddress = 1 in
def XOR32mi : Inst<(ops addr:$addr, imm32:$imm),
                   "xor $dst, $src2", 0x81, MRM6m,
                   [(set R32:$tmp1, (load addr:$addr)),
                    (set R32:$tmp2, (xor R32:$tmp1, imm32:$imm)),
                    (store addr:$addr, R32:$tmp2)]>;

// Alternatively, if each tmporary register is only used once, the instruction
// can just be described in nested form.  This would be the canonical 
// representation the target generator would convert the above into.  Pick your
// favorite indentation scheme.
let isTwoAddress = 1 in
def AND32mr : Inst<(ops addr:$addr, R32:$src),
                   "xor $dst, $src2", 0x81, MRM6m,
                   [(store addr:$addr,
                       (and
                            (load addr:$addr),
                            R32:$src)
                       )
                   ]>;

// Describing complex instructions is not too hard!  Note how implicit uses/defs
// become explicit here.
def CBW : Inst<(ops),
               "cbw", 0x98, RawFrm,
               [(set AX, (signext AL))]>;

// Noop, does nothing.
def NOOP : Inst<(ops), "nop", 0x90, RawFrm, []>;


// Instructions that don't expect optimization can use unspec.
def IN8rr : Inst<(ops), "in AL, EDX", 0xEC, RawFrm,
                 [(set AL, (unspec EDX))]>;

