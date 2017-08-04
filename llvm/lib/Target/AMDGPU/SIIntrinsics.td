//===-- SIIntrinsics.td - SI Intrinsic defs ----------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Backend internal SI Intrinsic Definitions. User code should not
// directly use these.
//
//===----------------------------------------------------------------------===//


let TargetPrefix = "SI", isTarget = 1 in {
  def int_SI_load_const : Intrinsic <[llvm_float_ty], [llvm_anyint_ty, llvm_i32_ty], [IntrNoMem]>;

  // Fully-flexible TBUFFER_STORE_FORMAT_* except for the ADDR64 bit, which is not exposed
  def int_SI_tbuffer_store : Intrinsic <
    [],
    [llvm_anyint_ty, // rsrc(SGPR)
     llvm_anyint_ty, // vdata(VGPR), overloaded for types i32, v2i32, v4i32
     llvm_i32_ty,    // num_channels(imm), selects opcode suffix: 1=X, 2=XY, 3=XYZ, 4=XYZW
     llvm_i32_ty,    // vaddr(VGPR)
     llvm_i32_ty,    // soffset(SGPR)
     llvm_i32_ty,    // inst_offset(imm)
     llvm_i32_ty,    // dfmt(imm)
     llvm_i32_ty,    // nfmt(imm)
     llvm_i32_ty,    // offen(imm)
     llvm_i32_ty,    // idxen(imm)
     llvm_i32_ty,    // glc(imm)
     llvm_i32_ty,    // slc(imm)
     llvm_i32_ty],   // tfe(imm)
    []>;

  // Fully-flexible BUFFER_LOAD_DWORD_* except for the ADDR64 bit, which is not exposed
  def int_SI_buffer_load_dword : Intrinsic <
    [llvm_anyint_ty], // vdata(VGPR), overloaded for types i32, v2i32, v4i32
    [llvm_anyint_ty,  // rsrc(SGPR)
     llvm_anyint_ty,  // vaddr(VGPR)
     llvm_i32_ty,     // soffset(SGPR)
     llvm_i32_ty,     // inst_offset(imm)
     llvm_i32_ty,     // offen(imm)
     llvm_i32_ty,     // idxen(imm)
     llvm_i32_ty,     // glc(imm)
     llvm_i32_ty,     // slc(imm)
     llvm_i32_ty],    // tfe(imm)
    [IntrReadMem, IntrArgMemOnly]>;

} // End TargetPrefix = "SI", isTarget = 1
