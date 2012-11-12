/*===-- target_ocaml.c - LLVM Ocaml Glue ------------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's ocaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/Target.h"
#include "caml/alloc.h"

/* string -> DataLayout.t */
CAMLprim LLVMTargetDataRef llvm_targetdata_create(value StringRep) {
  return LLVMCreateTargetData(String_val(StringRep));
}

/* DataLayout.t -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_targetdata_add(LLVMTargetDataRef TD, LLVMPassManagerRef PM){
  LLVMAddTargetData(TD, PM);
  return Val_unit;
}

/* DataLayout.t -> string */
CAMLprim value llvm_targetdata_as_string(LLVMTargetDataRef TD) {
  char *StringRep = LLVMCopyStringRepOfTargetData(TD);
  value Copy = copy_string(StringRep);
  LLVMDisposeMessage(StringRep);
  return Copy;
}

/* DataLayout.t -> unit */
CAMLprim value llvm_targetdata_dispose(LLVMTargetDataRef TD) {
  LLVMDisposeTargetData(TD);
  return Val_unit;
}

/* DataLayout.t -> Endian.t */
CAMLprim value llvm_byte_order(LLVMTargetDataRef TD) {
  return Val_int(LLVMByteOrder(TD));
}

/* DataLayout.t -> int */
CAMLprim value llvm_pointer_size(LLVMTargetDataRef TD) {
  return Val_int(LLVMPointerSize(TD));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_size_in_bits(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMSizeOfTypeInBits(TD, Ty));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_store_size(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMStoreSizeOfType(TD, Ty));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_abi_size(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMABISizeOfType(TD, Ty));
}

/* DataLayout.t -> Llvm.lltype -> int */
CAMLprim value llvm_abi_align(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return Val_int(LLVMABIAlignmentOfType(TD, Ty));
}

/* DataLayout.t -> Llvm.lltype -> int */
CAMLprim value llvm_stack_align(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return Val_int(LLVMCallFrameAlignmentOfType(TD, Ty));
}

/* DataLayout.t -> Llvm.lltype -> int */
CAMLprim value llvm_preferred_align(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return Val_int(LLVMPreferredAlignmentOfType(TD, Ty));
}

/* DataLayout.t -> Llvm.llvalue -> int */
CAMLprim value llvm_preferred_align_of_global(LLVMTargetDataRef TD,
                                              LLVMValueRef GlobalVar) {
  return Val_int(LLVMPreferredAlignmentOfGlobal(TD, GlobalVar));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t -> int */
CAMLprim value llvm_element_at_offset(LLVMTargetDataRef TD, LLVMTypeRef Ty,
                                      value Offset) {
  return Val_int(LLVMElementAtOffset(TD, Ty, Int_val(Offset)));
}

/* DataLayout.t -> Llvm.lltype -> int -> Int64.t */
CAMLprim value llvm_offset_of_element(LLVMTargetDataRef TD, LLVMTypeRef Ty,
                                      value Index) {
  return caml_copy_int64(LLVMOffsetOfElement(TD, Ty, Int_val(Index)));
}
