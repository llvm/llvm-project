// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s 2>&1 | FileCheck -check-prefix=SDAG %s
// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../include %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;


def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;


// With one address space
def pat_frag_a : PatFrag <(ops node:$ptr), (load node:$ptr), [{}]> {
  let AddressSpaces = [ 999 ];
  let IsLoad = 1; // FIXME: Can this be inferred?
  let MemoryVT = i32;
  let MinAlignment = 2;
}

// With multiple address spaces
def pat_frag_b : PatFrag <(ops node:$ptr), (load node:$ptr), [{}]> {
  let AddressSpaces = [ 123, 455 ];
  let IsLoad = 1; // FIXME: Can this be inferred?
  let MemoryVT = i32;
}

def inst_a : Instruction {
  let OutOperandList = (outs GPR32:$dst);
  let InOperandList = (ins GPR32:$src);
}

def inst_b : Instruction {
  let OutOperandList = (outs GPR32:$dst);
  let InOperandList = (ins GPR32:$src);
}

def inst_c : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins GPR32:$src0, GPR32:$src1);
}

def inst_d : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins GPR32:$src0, GPR32:$src1);
}

// SDAG: case 2: {
// SDAG-NEXT: // Predicate_pat_frag_b
// SDAG-NEXT: SDNode *N = Node;
// SDAG-NEXT: (void)N;
// SDAG-NEXT: unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();
// SDAG-NEXT: if (AddrSpace != 123 && AddrSpace != 455)
// SDAG-NEXT: return false;
// SDAG-NEXT: if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
// SDAG-NEXT: return true;


// GISEL: GIM_Try, /*On fail goto*//*Label 0*/ {{[0-9]+}}, // Rule ID 0 //
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// GISEL-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// GISEL-NEXT: GIM_CheckMemoryAddressSpace, /*MI*/0, /*MMO*/0, /*NumAddrSpace*/2, /*AddrSpace*/123, /*AddrSpace*/455,
// GISEL-NEXT: GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/4,
// GISEL-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
def : Pat <
  (pat_frag_b GPR32:$src),
  (inst_b GPR32:$src)
>;


// SDAG: case 3: {
// SDAG: // Predicate_pat_frag_a
// SDAG-NEXT: SDNode *N = Node;
// SDAG-NEXT: (void)N;
// SDAG-NEXT: unsigned AddrSpace = cast<MemSDNode>(N)->getAddressSpace();

// SDAG-NEXT: if (AddrSpace != 999)
// SDAG-NEXT: return false;
// SDAG-NEXT: if (cast<MemSDNode>(N)->getAlign() < Align(2))
// SDAG-NEXT: return false;
// SDAG-NEXT: if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
// SDAG-NEXT: return true;

// GISEL: GIM_Try, /*On fail goto*//*Label 1*/ {{[0-9]+}}, // Rule ID 1 //
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_LOAD,
// GISEL-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// GISEL-NEXT: GIM_CheckMemoryAddressSpace, /*MI*/0, /*MMO*/0, /*NumAddrSpace*/1, /*AddrSpace*/999,
// GISEL-NEXT: GIM_CheckMemoryAlignment, /*MI*/0, /*MMO*/0, /*MinAlign*/2,
// GISEL-NEXT: GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/4,
// GISEL-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
def : Pat <
  (pat_frag_a GPR32:$src),
  (inst_a GPR32:$src)
>;


def truncstorei16_addrspace : PatFrag<(ops node:$val, node:$ptr),
                                (truncstore node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i16;
  let AddressSpaces = [ 123, 455 ];
}

// Test truncstore without a specific MemoryVT
// GISEL: GIM_Try, /*On fail goto*//*Label 2*/ {{[0-9]+}}, // Rule ID 2 //
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_STORE,
// GISEL-NEXT: GIM_CheckMemorySizeLessThanLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// GISEL-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(int64_t)AtomicOrdering::NotAtomic,
// GISEL-NEXT: // MIs[0] src0
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
def : Pat <
  (truncstore GPR32:$src0, GPR32:$src1),
  (inst_c GPR32:$src0, GPR32:$src1)
>;

// Test non-truncstore has a size equal to LLT check.
// GISEL: GIM_Try, /*On fail goto*//*Label 3*/ {{[0-9]+}}, // Rule ID 3 //
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_STORE,
// GISEL-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
def : Pat <
  (store GPR32:$src0, GPR32:$src1),
  (inst_d GPR32:$src0, GPR32:$src1)
>;

// Test truncstore with specific MemoryVT
// GISEL: GIM_Try, /*On fail goto*//*Label 4*/ {{[0-9]+}}, // Rule ID 4 //
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_STORE,
// GISEL-NEXT: GIM_CheckMemorySizeLessThanLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// GISEL-NEXT: GIM_CheckMemoryAddressSpace, /*MI*/0, /*MMO*/0, /*NumAddrSpace*/2, /*AddrSpace*/123, /*AddrSpace*/455,
// GISEL-NEXT: GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/2,
def : Pat <
  (truncstorei16_addrspace GPR32:$src0, GPR32:$src1),
  (inst_c GPR32:$src0, GPR32:$src1)
>;
