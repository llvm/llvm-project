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

// TODO(jfb) The following comes from NVPTX. Is it really needed, or can we do
//           away with it? Try deleting once the backend works.
// WebAssembly uses virtual registers, but the backend defines a few physical
// registers here to keep SDAG and the MachineInstr layers happy.
foreach i = 0-4 in {
  def I#i : WebAssemblyReg<"%i."#i>; // i32
  def L#i : WebAssemblyReg<"%l."#i>; // i64
  def F#i : WebAssemblyReg<"%f."#i>; // f32
  def D#i : WebAssemblyReg<"%d."#i>; // f64
}

//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//

def I32 : WebAssemblyRegClass<[i32], 32, (add (sequence "I%u", 0, 4), SP32)>;
def I64 : WebAssemblyRegClass<[i64], 64, (add (sequence "L%u", 0, 4), SP64)>;
def F32 : WebAssemblyRegClass<[f32], 32, (add (sequence "F%u", 0, 4))>;
def F64 : WebAssemblyRegClass<[f64], 64, (add (sequence "D%u", 0, 4))>;
