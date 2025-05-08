//===-- ParasolRegisterInfo.td - Parasol Register defs ---------*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Parasol register file
//===----------------------------------------------------------------------===//

let Namespace = "Parasol" in {
class ParasolReg<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

// Encrypted registers
// Change this 6 to the ceil(log2(num_registers)) desired
class ParasolPointerReg<bits<6> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{5-0} = Enc;
  let AltNames = alt;
} 

// integer register numbers
class ParasolIntReg<bits<6> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{5-0} = Enc;
  let AltNames = alt;
}

class ParasolFakeReg<bits<6> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{5-0} = Enc;
  let AltNames = alt;
}

// TODO: Add some registers for the constants zero and one?

} // end Namespace

//===----------------------------------------------------------------------===//
//  Pointer registers
//===----------------------------------------------------------------------===//

// Define 64 pointer registers using a loop
foreach i = 0-63 in {
  def P#i : ParasolPointerReg<i, "p"#i>;
}

// Add to the encrypted register class
def PR : RegisterClass<"Parasol", [i32], 32, (sequence "P%u", 0, 63)>;

//===----------------------------------------------------------------------===//
//  Integer registers
//===----------------------------------------------------------------------===//

// Define 64 integer registers using a loop
foreach i = 0-63 in {
  def R#i : ParasolIntReg<i, "r"#i>;
}

// Add to the integer register class
// NOTE: Due to a bug in LLVM this ValueType list needs to be specified in
// largest to smallest order. Otherwise when put into a register bank during
// globalisel the size of the register bank will be too small (it seems to
// choose the first element as the size of the class).
def IR : RegisterClass<"Parasol", [i32, i16, i8, i1], 32, (sequence "R%u", 0, 63)>;

//===----------------------------------------------------------------------===//
//  Control registers
//===----------------------------------------------------------------------===//

// We have some fake registers to handle things like function pointers.
foreach i = 0-31 in {
  def X#i : ParasolFakeReg<i, "x"#i>;
}

// The order of registers represents the preferred allocation sequence.
// Registers are listed in the order caller-save, callee-save, specials.
def GPR : RegisterClass<"Parasol", [i32], 32, (add
    (sequence "X%u", 10, 17),
    (sequence "X%u", 5, 7),
    (sequence "X%u", 28, 31),
    (sequence "X%u", 8, 9),
    (sequence "X%u", 18, 27),
    (sequence "X%u", 0, 4)
  )>;

def SP : RegisterClass<"Parasol", [i32], 32, (add X2)>;
