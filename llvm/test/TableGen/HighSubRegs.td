// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s

include "llvm/Target/Target.td"
def TestTarget : Target;

def hi16 : SubRegIndex<16, 16>; // hi 16 bits of 32-bit register
def hi32 : SubRegIndex<32, 32>; // hi 32 bits of 64-bit register

def R16 : Register<"">;

def R32 : Register<""> {
  let SubRegs = [R16];
  let SubRegIndices = [hi16];
}

def R64 : Register<""> {
  let SubRegs = [R32];
  let SubRegIndices = [hi32];
}

def TestRC : RegisterClass<"Test", [i64], 0, (add R64)>;

// CHECK-LABEL: Register R16:

// CHECK-LABEL: Register R32:
// CHECK: SubReg hi16 = R16

// CHECK-LABEL: Register R64:
// CHECK: SubReg hi16 = R16
// CHECK: SubReg hi32 = R32

// FIXME: Wrong subregindex for R16 as a subreg of R64.
