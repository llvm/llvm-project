//===--------------- IntrinsicsRipple.td - RIpple intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "ripple" in {

// Ripple dim; takes as input a processing element (PE) identifier (e.g., 0 vector,
// 1 thread, 2 core) and dimension index; returns it's multi-dimensional index.
def int_ripple_block_index : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [llvm_ptr_ty, LLVMMatchType<0>],
                              [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

// Ripple get dimension size; takes as input a PE identifier and dimension
// index. Returns the amount of PEs in for this dimension.
def int_ripple_block_getsize : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                  [llvm_ptr_ty, LLVMMatchType<0>],
                                  [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

// Ripple set tensor dimension size; takes as input a PE identifier, a dimension
// index and the size.
def int_ripple_block_setshape : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_anyint_ty,
                                    LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                                    LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                                    LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                                    LLVMMatchType<0>],
                                    [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<0>>]>;

// RIpple tensor reduction operator;
// The first argument is a bitset to represent which dimensions are being
// reduced, e.g., 0x1 is a reduction on dimension 0, 0x2 dimension 1, 0x3 on
// dimensions 0 and 1.
let IntrProperties = [IntrNoMem, IntrWillReturn, IntrSpeculatable,
                      ImmArg<ArgIndex<0>>] in {

  def int_ripple_reduce_add : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_mul : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_and : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_or : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_xor : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_smax : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_smin : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_umax : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_umin : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  // Floating point reductions
  def int_ripple_reduce_fadd : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_fmul : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_fmin : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_fmax : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_fminimum : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;
  def int_ripple_reduce_fmaximum : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                          [llvm_i64_ty, LLVMMatchType<0>]>;

}

def int_ripple_broadcast : DefaultAttrsIntrinsic<
    [llvm_any_ty],
    [llvm_ptr_ty, llvm_i64_ty, LLVMMatchType<0>],
    [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<1>>]>;

// Ripple block slicing
// Ripple slice accepts one input and
// 10 int64_t immediates defining the slice to be taken
// negative parameters mean "take all the elements of this dimension"
let IntrProperties = [IntrNoMem, IntrWillReturn, IntrSpeculatable,
    ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
    ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
    ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>, ImmArg<ArgIndex<9>>,
    ImmArg<ArgIndex<10>>] in {
  def int_ripple_slice : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>,
    llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,  llvm_i64_ty,
     llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
      llvm_i64_ty, llvm_i64_ty]>;
}


// Ripple shuffle operator;
// The first and second operand are tensors whose values are shuffled, the third
// operand is a pointer to a function that performs the index-mapping and
// the boolean indicates if we shuffle 1 (false) or 2 (true) inputs.
def int_ripple_shuffle: DefaultAttrsIntrinsic<[llvm_any_ty],
                                              [LLVMMatchType<0>, LLVMMatchType<0>,
                                                llvm_i1_ty, llvm_ptr_ty],
                                              [IntrWillReturn, ImmArg<ArgIndex<2>>]>;

def int_ripple_fpext: DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                            [llvm_anyfloat_ty],
                                            [IntrWillReturn, IntrNoMem, IntrSpeculatable]>;

def int_ripple_fptrunc: DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                            [llvm_anyfloat_ty],
                                            [IntrWillReturn, IntrNoMem, IntrSpeculatable]>;

} // End TargetPrefix = "ripple"
