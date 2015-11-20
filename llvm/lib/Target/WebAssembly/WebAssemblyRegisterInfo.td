//WebAssemblyRegisterInfo.td-Describe the WebAssembly Registers -*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file describes the WebAssembly register classes and some nominal
/// physical registers.
///
//===----------------------------------------------------------------------===//

class WebAssemblyReg<string n> : Register<n> {
  let Namespace = "WebAssembly";
}

class WebAssemblyRegClass<list<ValueType> regTypes, int alignment, dag regList>
     : RegisterClass<"WebAssembly", regTypes, alignment, regList>;

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

// Special registers used as the frame and stack pointer.
//
// WebAssembly may someday supports mixed 32-bit and 64-bit heaps in the same
// application, which requires separate width FP and SP.
def FP32 : WebAssemblyReg<"%FP32">;
def FP64 : WebAssemblyReg<"%FP64">;
def SP32 : WebAssemblyReg<"%SP32">;
def SP64 : WebAssemblyReg<"%SP64">;

// The register allocation framework requires register classes have at least
// one register, so we define a few for the floating point register classes
// since we otherwise don't need a physical register in those classes.
def F32_0 : WebAssemblyReg<"%f32.0">;
def F64_0 : WebAssemblyReg<"%f64.0">;

// The expression stack "register". This is an opaque entity which serves to
// order uses and defs that must remain in FIFO order.
def EXPR_STACK : WebAssemblyReg<"STACK">;

//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//

def I32 : WebAssemblyRegClass<[i32], 32, (add FP32, SP32)>;
def I64 : WebAssemblyRegClass<[i64], 64, (add FP64, SP64)>;
def F32 : WebAssemblyRegClass<[f32], 32, (add F32_0)>;
def F64 : WebAssemblyRegClass<[f64], 64, (add F64_0)>;
