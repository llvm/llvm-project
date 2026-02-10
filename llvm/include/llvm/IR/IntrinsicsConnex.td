//===- IntrinsicsConnex.td - Defines Connex-S intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Connex-specific intrinsics.
//
//===----------------------------------------------------------------------===//

// All Connex-S vector processor intrinsics start with "llvm.connex."
//
let TargetPrefix = "connex" in {

  /*
   * Note: all intrinsics defined in these .td files start with
   *   the int_ prefix (from intrinsic). For this file they start with
   *   int_connex prefix - otherwise we get the following TableGen error
   *   <<error:Intrinsic 'int_end_repeat' does not start with 'llvm.connex.'!>>
   *
   * The LLVM IR intrinsics extend the LLVM language s.t. we can use
   *   these instructions in an LLVM IR program. We also need to define the
   *   corresponding assembly instructions in the back end TableGen files.
   */

  /* Note: Following Intrinsics.td:
       class Intrinsic<list<LLVMType> ret_types,
                list<LLVMType> param_types = [],
                list<IntrinsicProperty> properties = [],
                string name = "">
  */


  /* Small-note:
       llvm_i64_ty makes simpler my LLVM IR generation in the LoopVectorize.cpp
      module:
       def int_connex_repeat_x_times : Intrinsic<[], [llvm_i64_ty], []>;
     But llvm_i32_ty is in accordance to the original i32 type of n.vec in the
      LoopVectorize.cpp module:
        def int_connex_repeat_x_times : Intrinsic<[], [llvm_i32_ty], []>;

     Small-note: We get inspired from include/llvm/IR/IntrinsicsPowerPC.td:
      // Intrinsics used to generate ctrl-based loops.
      def int_ppc_mtctr : Intrinsic<[], [llvm_anyint_ty], []>;

     Small-note: Trying to use a polymorphic definition, which requires
       specifying the actual type in Function::Create(FunctionType::get(), ...)
       is:
         def int_connex_repeat_x_times : Intrinsic<[], [llvm_anyint_ty], []>;
       When instantiating it in LoopVectorize.cpp like this:
         Value *instrinsicFunc = Intrinsic::getDeclaration(M,
                                            Intrinsic::connex_repeat_x_times);
        it gives error at runtime:
          llvm::ArrayRef<T>::operator[](size_t) const [with T = llvm::Type*;
             size_t = long unsigned int]: Assertion `Index < Length &&
              "Invalid index!"' failed.
  */
  def int_connex_repeat_x_times : Intrinsic<[], [llvm_i64_ty], []>;
  def int_connex_end_repeat : Intrinsic<[], [], []>;

  /* Note: Possibly useful in the future.
     Connex OPINCAA's END_REPEAT does not have a relative offset,
       as the standard Connex assembly ijmpnzdec instruction,
       since it falls on Opincaa to compute the jump back relative offset.
      We can also use a setlc to position it outside the loop created by the
       ijmpnzdec instruction by using it inside a delay-slot instruction.

      def int_connex_setlc : Intrinsic<[], [llvm_i16_ty], []>;
      def int_connex_ijmpnzdec : Intrinsic<[], [], []>;
  */



  /* IMPORTANT: REDUCE cannot return a value. It is the duty of the host (CPU)
      to read the result itself from the REDUCE issued by Connex-S.
   Therefore this definition is incorrect:
      def int_connex_reduce : Intrinsic<[llvm_i32_ty], [llvm_v128i16_ty], []>;
  */
  /* Also good:
      def int_connex_reduce : Intrinsic<[], [llvm_v128i16_ty], []>;
      def int_connex_reduce_i32 : Intrinsic<[], [llvm_v64i32_ty], []>;
      def int_connex_reduce_f16 : Intrinsic<[], [llvm_v128f16_ty], []>;
  */
  def int_connex_reduce : Intrinsic<[], [llvm_anyvector_ty], []>;

  /* Note: ctpop is already defined in Intrinsics.td.
     So the below definition is not required:
       def int_connex_ctpop : Intrinsic<[llvm_v8i16_ty],
                                        [llvm_v8i16_ty], []>;
  */


  // Inherited BPF scalar intrinsics: Specialized loads from packet
  def int_connex_load_byte : ClangBuiltin<"__builtin_connex_load_byte">,
           Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem]>;
  def int_connex_load_half : ClangBuiltin<"__builtin_connex_load_half">,
           Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem]>;
  def int_connex_load_word : ClangBuiltin<"__builtin_connex_load_word">,
           Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem]>;
  def int_connex_pseudo : ClangBuiltin<"__builtin_connex_pseudo">,
           Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty]>;
}

