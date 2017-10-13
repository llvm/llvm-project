/*===-- llvm_ocaml.c - LLVM OCaml Glue --------------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's OCaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "llvm-c/Core.h"
#include "llvm-c/Support.h"
#include "caml/alloc.h"
#include "caml/custom.h"
#include "caml/memory.h"
#include "caml/fail.h"
#include "caml/callback.h"

value llvm_string_of_message(char* Message) {
  value String = caml_copy_string(Message);
  LLVMDisposeMessage(Message);

  return String;
}

void llvm_raise(value Prototype, char *Message) {
  CAMLparam1(Prototype);
  caml_raise_with_arg(Prototype, llvm_string_of_message(Message));
  CAMLnoreturn;
}

static value llvm_fatal_error_handler;

static void llvm_fatal_error_trampoline(const char *Reason) {
  callback(llvm_fatal_error_handler, caml_copy_string(Reason));
}

CAMLprim value llvm_install_fatal_error_handler(value Handler) {
  LLVMInstallFatalErrorHandler(llvm_fatal_error_trampoline);
  llvm_fatal_error_handler = Handler;
  caml_register_global_root(&llvm_fatal_error_handler);
  return Val_unit;
}

CAMLprim value llvm_reset_fatal_error_handler(value Unit) {
  caml_remove_global_root(&llvm_fatal_error_handler);
  LLVMResetFatalErrorHandler();
  return Val_unit;
}

CAMLprim value llvm_enable_pretty_stacktrace(value Unit) {
  LLVMEnablePrettyStackTrace();
  return Val_unit;
}

CAMLprim value llvm_parse_command_line_options(value Overview, value Args) {
  char *COverview;
  if (Overview == Val_int(0)) {
    COverview = NULL;
  } else {
    COverview = String_val(Field(Overview, 0));
  }
  LLVMParseCommandLineOptions(Wosize_val(Args), (const char* const*) Op_val(Args), COverview);
  return Val_unit;
}

static value alloc_variant(int tag, void *Value) {
  value Iter = alloc_small(1, tag);
  Field(Iter, 0) = Val_op(Value);
  return Iter;
}

/* Macro to convert the C first/next/last/prev idiom to the Ocaml llpos/
   llrev_pos idiom. */
#define DEFINE_ITERATORS(camlname, cname, pty, cty, pfun) \
  /* llmodule -> ('a, 'b) llpos */                        \
  CAMLprim value llvm_##camlname##_begin(pty Mom) {       \
    cty First = LLVMGetFirst##cname(Mom);                 \
    if (First)                                            \
      return alloc_variant(1, First);                     \
    return alloc_variant(0, Mom);                         \
  }                                                       \
                                                          \
  /* llvalue -> ('a, 'b) llpos */                         \
  CAMLprim value llvm_##camlname##_succ(cty Kid) {        \
    cty Next = LLVMGetNext##cname(Kid);                   \
    if (Next)                                             \
      return alloc_variant(1, Next);                      \
    return alloc_variant(0, pfun(Kid));                   \
  }                                                       \
                                                          \
  /* llmodule -> ('a, 'b) llrev_pos */                    \
  CAMLprim value llvm_##camlname##_end(pty Mom) {         \
    cty Last = LLVMGetLast##cname(Mom);                   \
    if (Last)                                             \
      return alloc_variant(1, Last);                      \
    return alloc_variant(0, Mom);                         \
  }                                                       \
                                                          \
  /* llvalue -> ('a, 'b) llrev_pos */                     \
  CAMLprim value llvm_##camlname##_pred(cty Kid) {        \
    cty Prev = LLVMGetPrevious##cname(Kid);               \
    if (Prev)                                             \
      return alloc_variant(1, Prev);                      \
    return alloc_variant(0, pfun(Kid));                   \
  }

/*===-- Context error handling --------------------------------------------===*/

void llvm_diagnostic_handler_trampoline(LLVMDiagnosticInfoRef DI,
                                        void *DiagnosticContext) {
  caml_callback(*((value *)DiagnosticContext), (value)DI);
}

/* Diagnostic.t -> string */
CAMLprim value llvm_get_diagnostic_description(value Diagnostic) {
  return llvm_string_of_message(
      LLVMGetDiagInfoDescription((LLVMDiagnosticInfoRef)Diagnostic));
}

/* Diagnostic.t -> DiagnosticSeverity.t */
CAMLprim value llvm_get_diagnostic_severity(value Diagnostic) {
  return Val_int(LLVMGetDiagInfoSeverity((LLVMDiagnosticInfoRef)Diagnostic));
}

static void llvm_remove_diagnostic_handler(LLVMContextRef C) {
  if (LLVMContextGetDiagnosticHandler(C) ==
      llvm_diagnostic_handler_trampoline) {
    value *Handler = (value *)LLVMContextGetDiagnosticContext(C);
    remove_global_root(Handler);
    free(Handler);
  }
}

/* llcontext -> (Diagnostic.t -> unit) option -> unit */
CAMLprim value llvm_set_diagnostic_handler(LLVMContextRef C, value Handler) {
  llvm_remove_diagnostic_handler(C);
  if (Handler == Val_int(0)) {
    LLVMContextSetDiagnosticHandler(C, NULL, NULL);
  } else {
    value *DiagnosticContext = malloc(sizeof(value));
    if (DiagnosticContext == NULL)
      caml_raise_out_of_memory();
    caml_register_global_root(DiagnosticContext);
    *DiagnosticContext = Field(Handler, 0);
    LLVMContextSetDiagnosticHandler(C, llvm_diagnostic_handler_trampoline,
                                    DiagnosticContext);
  }
  return Val_unit;
}

/*===-- Contexts ----------------------------------------------------------===*/

/* unit -> llcontext */
CAMLprim LLVMContextRef llvm_create_context(value Unit) {
  return LLVMContextCreate();
}

/* llcontext -> unit */
CAMLprim value llvm_dispose_context(LLVMContextRef C) {
  llvm_remove_diagnostic_handler(C);
  LLVMContextDispose(C);
  return Val_unit;
}

/* unit -> llcontext */
CAMLprim LLVMContextRef llvm_global_context(value Unit) {
  return LLVMGetGlobalContext();
}

/* llcontext -> string -> int */
CAMLprim value llvm_mdkind_id(LLVMContextRef C, value Name) {
  unsigned MDKindID = LLVMGetMDKindIDInContext(C, String_val(Name),
                                               caml_string_length(Name));
  return Val_int(MDKindID);
}

/*===-- Attributes --------------------------------------------------------===*/

/* string -> llattrkind */
CAMLprim value llvm_enum_attr_kind(value Name) {
  unsigned Kind = LLVMGetEnumAttributeKindForName(
                        String_val(Name), caml_string_length(Name));
  if(Kind == 0)
    caml_raise_with_arg(*caml_named_value("Llvm.UnknownAttribute"), Name);
  return Val_int(Kind);
}

/* llcontext -> int -> int64 -> llattribute */
CAMLprim LLVMAttributeRef
llvm_create_enum_attr_by_kind(LLVMContextRef C, value Kind, value Value) {
  return LLVMCreateEnumAttribute(C, Int_val(Kind), Int64_val(Value));
}

/* llattribute -> bool */
CAMLprim value llvm_is_enum_attr(LLVMAttributeRef A) {
  return Val_int(LLVMIsEnumAttribute(A));
}

/* llattribute -> llattrkind */
CAMLprim value llvm_get_enum_attr_kind(LLVMAttributeRef A) {
  return Val_int(LLVMGetEnumAttributeKind(A));
}

/* llattribute -> int64 */
CAMLprim value llvm_get_enum_attr_value(LLVMAttributeRef A) {
  return caml_copy_int64(LLVMGetEnumAttributeValue(A));
}

/* llcontext -> kind:string -> name:string -> llattribute */
CAMLprim LLVMAttributeRef llvm_create_string_attr(LLVMContextRef C,
                                                  value Kind, value Value) {
  return LLVMCreateStringAttribute(C,
                        String_val(Kind), caml_string_length(Kind),
                        String_val(Value), caml_string_length(Value));
}

/* llattribute -> bool */
CAMLprim value llvm_is_string_attr(LLVMAttributeRef A) {
  return Val_int(LLVMIsStringAttribute(A));
}

/* llattribute -> string */
CAMLprim value llvm_get_string_attr_kind(LLVMAttributeRef A) {
  unsigned Length;
  const char *String = LLVMGetStringAttributeKind(A, &Length);
  value Result = caml_alloc_string(Length);
  memcpy(String_val(Result), String, Length);
  return Result;
}

/* llattribute -> string */
CAMLprim value llvm_get_string_attr_value(LLVMAttributeRef A) {
  unsigned Length;
  const char *String = LLVMGetStringAttributeValue(A, &Length);
  value Result = caml_alloc_string(Length);
  memcpy(String_val(Result), String, Length);
  return Result;
}

/*===-- Modules -----------------------------------------------------------===*/

/* llcontext -> string -> llmodule */
CAMLprim LLVMModuleRef llvm_create_module(LLVMContextRef C, value ModuleID) {
  return LLVMModuleCreateWithNameInContext(String_val(ModuleID), C);
}

/* llmodule -> unit */
CAMLprim value llvm_dispose_module(LLVMModuleRef M) {
  LLVMDisposeModule(M);
  return Val_unit;
}

/* llmodule -> string */
CAMLprim value llvm_target_triple(LLVMModuleRef M) {
  return caml_copy_string(LLVMGetTarget(M));
}

/* string -> llmodule -> unit */
CAMLprim value llvm_set_target_triple(value Trip, LLVMModuleRef M) {
  LLVMSetTarget(M, String_val(Trip));
  return Val_unit;
}

/* llmodule -> string */
CAMLprim value llvm_data_layout(LLVMModuleRef M) {
  return caml_copy_string(LLVMGetDataLayout(M));
}

/* string -> llmodule -> unit */
CAMLprim value llvm_set_data_layout(value Layout, LLVMModuleRef M) {
  LLVMSetDataLayout(M, String_val(Layout));
  return Val_unit;
}

/* llmodule -> unit */
CAMLprim value llvm_dump_module(LLVMModuleRef M) {
  LLVMDumpModule(M);
  return Val_unit;
}

/* string -> llmodule -> unit */
CAMLprim value llvm_print_module(value Filename, LLVMModuleRef M) {
  char* Message;

  if(LLVMPrintModuleToFile(M, String_val(Filename), &Message))
    llvm_raise(*caml_named_value("Llvm.IoError"), Message);

  return Val_unit;
}

/* llmodule -> string */
CAMLprim value llvm_string_of_llmodule(LLVMModuleRef M) {
  CAMLparam0();
  CAMLlocal1(ModuleStr);
  char* ModuleCStr;

  ModuleCStr = LLVMPrintModuleToString(M);
  ModuleStr = caml_copy_string(ModuleCStr);
  LLVMDisposeMessage(ModuleCStr);

  CAMLreturn(ModuleStr);
}

/* llmodule -> string -> unit */
CAMLprim value llvm_set_module_inline_asm(LLVMModuleRef M, value Asm) {
  LLVMSetModuleInlineAsm(M, String_val(Asm));
  return Val_unit;
}

/*===-- Types -------------------------------------------------------------===*/

/* lltype -> TypeKind.t */
CAMLprim value llvm_classify_type(LLVMTypeRef Ty) {
  return Val_int(LLVMGetTypeKind(Ty));
}

CAMLprim value llvm_type_is_sized(LLVMTypeRef Ty) {
    return Val_bool(LLVMTypeIsSized(Ty));
}

/* lltype -> llcontext */
CAMLprim LLVMContextRef llvm_type_context(LLVMTypeRef Ty) {
  return LLVMGetTypeContext(Ty);
}

/* lltype -> unit */
CAMLprim value llvm_dump_type(LLVMTypeRef Val) {
#ifdef LLVM_ENABLE_DUMP
  LLVMDumpType(Val);
#else
  caml_raise_with_arg(*caml_named_value("Llvm.FeatureDisabled"),
      caml_copy_string("dump"));
#endif
  return Val_unit;
}

/* lltype -> string */
CAMLprim value llvm_string_of_lltype(LLVMTypeRef M) {
  CAMLparam0();
  CAMLlocal1(TypeStr);
  char* TypeCStr;

  TypeCStr = LLVMPrintTypeToString(M);
  TypeStr = caml_copy_string(TypeCStr);
  LLVMDisposeMessage(TypeCStr);

  CAMLreturn(TypeStr);
}

/*--... Operations on integer types ........................................--*/

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_i1_type (LLVMContextRef Context) {
  return LLVMInt1TypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_i8_type (LLVMContextRef Context) {
  return LLVMInt8TypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_i16_type (LLVMContextRef Context) {
  return LLVMInt16TypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_i32_type (LLVMContextRef Context) {
  return LLVMInt32TypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_i64_type (LLVMContextRef Context) {
  return LLVMInt64TypeInContext(Context);
}

/* llcontext -> int -> lltype */
CAMLprim LLVMTypeRef llvm_integer_type(LLVMContextRef Context, value Width) {
  return LLVMIntTypeInContext(Context, Int_val(Width));
}

/* lltype -> int */
CAMLprim value llvm_integer_bitwidth(LLVMTypeRef IntegerTy) {
  return Val_int(LLVMGetIntTypeWidth(IntegerTy));
}

/*--... Operations on real types ...........................................--*/

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_float_type(LLVMContextRef Context) {
  return LLVMFloatTypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_double_type(LLVMContextRef Context) {
  return LLVMDoubleTypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_x86fp80_type(LLVMContextRef Context) {
  return LLVMX86FP80TypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_fp128_type(LLVMContextRef Context) {
  return LLVMFP128TypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_ppc_fp128_type(LLVMContextRef Context) {
  return LLVMPPCFP128TypeInContext(Context);
}

/*--... Operations on function types .......................................--*/

/* lltype -> lltype array -> lltype */
CAMLprim LLVMTypeRef llvm_function_type(LLVMTypeRef RetTy, value ParamTys) {
  return LLVMFunctionType(RetTy, (LLVMTypeRef *) ParamTys,
                          Wosize_val(ParamTys), 0);
}

/* lltype -> lltype array -> lltype */
CAMLprim LLVMTypeRef llvm_var_arg_function_type(LLVMTypeRef RetTy,
                                                value ParamTys) {
  return LLVMFunctionType(RetTy, (LLVMTypeRef *) ParamTys,
                          Wosize_val(ParamTys), 1);
}

/* lltype -> bool */
CAMLprim value llvm_is_var_arg(LLVMTypeRef FunTy) {
  return Val_bool(LLVMIsFunctionVarArg(FunTy));
}

/* lltype -> lltype array */
CAMLprim value llvm_param_types(LLVMTypeRef FunTy) {
  value Tys = alloc(LLVMCountParamTypes(FunTy), 0);
  LLVMGetParamTypes(FunTy, (LLVMTypeRef *) Tys);
  return Tys;
}

/*--... Operations on struct types .........................................--*/

/* llcontext -> lltype array -> lltype */
CAMLprim LLVMTypeRef llvm_struct_type(LLVMContextRef C, value ElementTypes) {
  return LLVMStructTypeInContext(C, (LLVMTypeRef *) ElementTypes,
                                 Wosize_val(ElementTypes), 0);
}

/* llcontext -> lltype array -> lltype */
CAMLprim LLVMTypeRef llvm_packed_struct_type(LLVMContextRef C,
                                             value ElementTypes) {
  return LLVMStructTypeInContext(C, (LLVMTypeRef *) ElementTypes,
                                 Wosize_val(ElementTypes), 1);
}

/* llcontext -> string -> lltype */
CAMLprim LLVMTypeRef llvm_named_struct_type(LLVMContextRef C,
                                            value Name) {
  return LLVMStructCreateNamed(C, String_val(Name));
}

CAMLprim value llvm_struct_set_body(LLVMTypeRef Ty,
                                    value ElementTypes,
                                    value Packed) {
  LLVMStructSetBody(Ty, (LLVMTypeRef *) ElementTypes,
                    Wosize_val(ElementTypes), Bool_val(Packed));
  return Val_unit;
}

/* lltype -> string option */
CAMLprim value llvm_struct_name(LLVMTypeRef Ty)
{
  CAMLparam0();
  const char *C = LLVMGetStructName(Ty);
  if (C) {
    CAMLlocal1(result);
    result = caml_alloc_small(1, 0);
    Store_field(result, 0, caml_copy_string(C));
    CAMLreturn(result);
  }
  CAMLreturn(Val_int(0));
}

/* lltype -> lltype array */
CAMLprim value llvm_struct_element_types(LLVMTypeRef StructTy) {
  value Tys = alloc(LLVMCountStructElementTypes(StructTy), 0);
  LLVMGetStructElementTypes(StructTy, (LLVMTypeRef *) Tys);
  return Tys;
}

/* lltype -> bool */
CAMLprim value llvm_is_packed(LLVMTypeRef StructTy) {
  return Val_bool(LLVMIsPackedStruct(StructTy));
}

/* lltype -> bool */
CAMLprim value llvm_is_opaque(LLVMTypeRef StructTy) {
  return Val_bool(LLVMIsOpaqueStruct(StructTy));
}

/*--... Operations on array, pointer, and vector types .....................--*/

/* lltype -> lltype array */
CAMLprim value llvm_subtypes(LLVMTypeRef Ty) {
    CAMLparam0();
    CAMLlocal1(Arr);

    unsigned Size = LLVMGetNumContainedTypes(Ty);

    Arr = caml_alloc(Size, 0);

    LLVMGetSubtypes(Ty, (LLVMTypeRef *) Arr);

    CAMLreturn(Arr);
}

/* lltype -> int -> lltype */
CAMLprim LLVMTypeRef llvm_array_type(LLVMTypeRef ElementTy, value Count) {
  return LLVMArrayType(ElementTy, Int_val(Count));
}

/* lltype -> lltype */
CAMLprim LLVMTypeRef llvm_pointer_type(LLVMTypeRef ElementTy) {
  return LLVMPointerType(ElementTy, 0);
}

/* lltype -> int -> lltype */
CAMLprim LLVMTypeRef llvm_qualified_pointer_type(LLVMTypeRef ElementTy,
                                                 value AddressSpace) {
  return LLVMPointerType(ElementTy, Int_val(AddressSpace));
}

/* lltype -> int -> lltype */
CAMLprim LLVMTypeRef llvm_vector_type(LLVMTypeRef ElementTy, value Count) {
  return LLVMVectorType(ElementTy, Int_val(Count));
}

/* lltype -> int */
CAMLprim value llvm_array_length(LLVMTypeRef ArrayTy) {
  return Val_int(LLVMGetArrayLength(ArrayTy));
}

/* lltype -> int */
CAMLprim value llvm_address_space(LLVMTypeRef PtrTy) {
  return Val_int(LLVMGetPointerAddressSpace(PtrTy));
}

/* lltype -> int */
CAMLprim value llvm_vector_size(LLVMTypeRef VectorTy) {
  return Val_int(LLVMGetVectorSize(VectorTy));
}

/*--... Operations on other types ..........................................--*/

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_void_type (LLVMContextRef Context) {
  return LLVMVoidTypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_label_type(LLVMContextRef Context) {
  return LLVMLabelTypeInContext(Context);
}

/* llcontext -> lltype */
CAMLprim LLVMTypeRef llvm_x86_mmx_type(LLVMContextRef Context) {
  return LLVMX86MMXTypeInContext(Context);
}

CAMLprim value llvm_type_by_name(LLVMModuleRef M, value Name)
{
  CAMLparam1(Name);
  LLVMTypeRef Ty = LLVMGetTypeByName(M, String_val(Name));
  if (Ty) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) Ty;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/*===-- VALUES ------------------------------------------------------------===*/

/* llvalue -> lltype */
CAMLprim LLVMTypeRef llvm_type_of(LLVMValueRef Val) {
  return LLVMTypeOf(Val);
}

/* keep in sync with ValueKind.t */
enum ValueKind {
  NullValue=0,
  Argument,
  BasicBlock,
  InlineAsm,
  MDNode,
  MDString,
  BlockAddress,
  ConstantAggregateZero,
  ConstantArray,
  ConstantDataArray,
  ConstantDataVector,
  ConstantExpr,
  ConstantFP,
  ConstantInt,
  ConstantPointerNull,
  ConstantStruct,
  ConstantVector,
  Function,
  GlobalAlias,
  GlobalVariable,
  UndefValue,
  Instruction
};

/* llvalue -> ValueKind.t */
#define DEFINE_CASE(Val, Kind) \
    do {if (LLVMIsA##Kind(Val)) CAMLreturn(Val_int(Kind));} while(0)

CAMLprim value llvm_classify_value(LLVMValueRef Val) {
  CAMLparam0();
  if (!Val)
    CAMLreturn(Val_int(NullValue));
  if (LLVMIsAConstant(Val)) {
    DEFINE_CASE(Val, BlockAddress);
    DEFINE_CASE(Val, ConstantAggregateZero);
    DEFINE_CASE(Val, ConstantArray);
    DEFINE_CASE(Val, ConstantDataArray);
    DEFINE_CASE(Val, ConstantDataVector);
    DEFINE_CASE(Val, ConstantExpr);
    DEFINE_CASE(Val, ConstantFP);
    DEFINE_CASE(Val, ConstantInt);
    DEFINE_CASE(Val, ConstantPointerNull);
    DEFINE_CASE(Val, ConstantStruct);
    DEFINE_CASE(Val, ConstantVector);
  }
  if (LLVMIsAInstruction(Val)) {
    CAMLlocal1(result);
    result = caml_alloc_small(1, 0);
    Store_field(result, 0, Val_int(LLVMGetInstructionOpcode(Val)));
    CAMLreturn(result);
  }
  if (LLVMIsAGlobalValue(Val)) {
    DEFINE_CASE(Val, Function);
    DEFINE_CASE(Val, GlobalAlias);
    DEFINE_CASE(Val, GlobalVariable);
  }
  DEFINE_CASE(Val, Argument);
  DEFINE_CASE(Val, BasicBlock);
  DEFINE_CASE(Val, InlineAsm);
  DEFINE_CASE(Val, MDNode);
  DEFINE_CASE(Val, MDString);
  DEFINE_CASE(Val, UndefValue);
  failwith("Unknown Value class");
}

/* llvalue -> string */
CAMLprim value llvm_value_name(LLVMValueRef Val) {
  return caml_copy_string(LLVMGetValueName(Val));
}

/* string -> llvalue -> unit */
CAMLprim value llvm_set_value_name(value Name, LLVMValueRef Val) {
  LLVMSetValueName(Val, String_val(Name));
  return Val_unit;
}

/* llvalue -> unit */
CAMLprim value llvm_dump_value(LLVMValueRef Val) {
  LLVMDumpValue(Val);
  return Val_unit;
}

/* llvalue -> string */
CAMLprim value llvm_string_of_llvalue(LLVMValueRef M) {
  CAMLparam0();
  CAMLlocal1(ValueStr);
  char* ValueCStr;

  ValueCStr = LLVMPrintValueToString(M);
  ValueStr = caml_copy_string(ValueCStr);
  LLVMDisposeMessage(ValueCStr);

  CAMLreturn(ValueStr);
}

/* llvalue -> llvalue -> unit */
CAMLprim value llvm_replace_all_uses_with(LLVMValueRef OldVal,
                                          LLVMValueRef NewVal) {
  LLVMReplaceAllUsesWith(OldVal, NewVal);
  return Val_unit;
}

/*--... Operations on users ................................................--*/

/* llvalue -> int -> llvalue */
CAMLprim LLVMValueRef llvm_operand(LLVMValueRef V, value I) {
  return LLVMGetOperand(V, Int_val(I));
}

/* llvalue -> int -> lluse */
CAMLprim LLVMUseRef llvm_operand_use(LLVMValueRef V, value I) {
  return LLVMGetOperandUse(V, Int_val(I));
}

/* llvalue -> int -> llvalue -> unit */
CAMLprim value llvm_set_operand(LLVMValueRef U, value I, LLVMValueRef V) {
  LLVMSetOperand(U, Int_val(I), V);
  return Val_unit;
}

/* llvalue -> int */
CAMLprim value llvm_num_operands(LLVMValueRef V) {
  return Val_int(LLVMGetNumOperands(V));
}

/*--... Operations on constants of (mostly) any type .......................--*/

/* llvalue -> bool */
CAMLprim value llvm_is_constant(LLVMValueRef Val) {
  return Val_bool(LLVMIsConstant(Val));
}

/* llvalue -> bool */
CAMLprim value llvm_is_null(LLVMValueRef Val) {
  return Val_bool(LLVMIsNull(Val));
}

/* llvalue -> bool */
CAMLprim value llvm_is_undef(LLVMValueRef Val) {
  return Val_bool(LLVMIsUndef(Val));
}

/* llvalue -> Opcode.t */
CAMLprim value llvm_constexpr_get_opcode(LLVMValueRef Val) {
  return LLVMIsAConstantExpr(Val) ?
      Val_int(LLVMGetConstOpcode(Val)) : Val_int(0);
}

/*--... Operations on instructions .........................................--*/

/* llvalue -> bool */
CAMLprim value llvm_has_metadata(LLVMValueRef Val) {
  return Val_bool(LLVMHasMetadata(Val));
}

/* llvalue -> int -> llvalue option */
CAMLprim value llvm_metadata(LLVMValueRef Val, value MDKindID) {
  CAMLparam1(MDKindID);
  LLVMValueRef MD;
  if ((MD = LLVMGetMetadata(Val, Int_val(MDKindID)))) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) MD;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* llvalue -> int -> llvalue -> unit */
CAMLprim value llvm_set_metadata(LLVMValueRef Val, value MDKindID,
                                 LLVMValueRef MD) {
  LLVMSetMetadata(Val, Int_val(MDKindID), MD);
  return Val_unit;
}

/* llvalue -> int -> unit */
CAMLprim value llvm_clear_metadata(LLVMValueRef Val, value MDKindID) {
  LLVMSetMetadata(Val, Int_val(MDKindID), NULL);
  return Val_unit;
}


/*--... Operations on metadata .............................................--*/

/* llcontext -> string -> llvalue */
CAMLprim LLVMValueRef llvm_mdstring(LLVMContextRef C, value S) {
  return LLVMMDStringInContext(C, String_val(S), caml_string_length(S));
}

/* llcontext -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_mdnode(LLVMContextRef C, value ElementVals) {
  return LLVMMDNodeInContext(C, (LLVMValueRef*) Op_val(ElementVals),
                             Wosize_val(ElementVals));
}

/* llcontext -> llvalue */
CAMLprim LLVMValueRef llvm_mdnull(LLVMContextRef C) {
  return NULL;
}

/* llvalue -> string option */
CAMLprim value llvm_get_mdstring(LLVMValueRef V) {
  CAMLparam0();
  const char *S;
  unsigned Len;

  if ((S = LLVMGetMDString(V, &Len))) {
    CAMLlocal2(Option, Str);

    Str = caml_alloc_string(Len);
    memcpy(String_val(Str), S, Len);
    Option = alloc(1,0);
    Store_field(Option, 0, Str);
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

CAMLprim value llvm_get_mdnode_operands(LLVMValueRef V) {
  CAMLparam0();
  CAMLlocal1(Operands);
  unsigned int n;

  n = LLVMGetMDNodeNumOperands(V);
  Operands = alloc(n, 0);
  LLVMGetMDNodeOperands(V, (LLVMValueRef *)  Operands);
  CAMLreturn(Operands);
}

/* llmodule -> string -> llvalue array */
CAMLprim value llvm_get_namedmd(LLVMModuleRef M, value Name)
{
  CAMLparam1(Name);
  CAMLlocal1(Nodes);
  Nodes = alloc(LLVMGetNamedMetadataNumOperands(M, String_val(Name)), 0);
  LLVMGetNamedMetadataOperands(M, String_val(Name), (LLVMValueRef *) Nodes);
  CAMLreturn(Nodes);
}

/* llmodule -> string -> llvalue -> unit */
CAMLprim value llvm_append_namedmd(LLVMModuleRef M, value Name, LLVMValueRef Val) {
  LLVMAddNamedMetadataOperand(M, String_val(Name), Val);
  return Val_unit;
}

/*--... Operations on scalar constants .....................................--*/

/* lltype -> int -> llvalue */
CAMLprim LLVMValueRef llvm_const_int(LLVMTypeRef IntTy, value N) {
  return LLVMConstInt(IntTy, (long long) Long_val(N), 1);
}

/* lltype -> Int64.t -> bool -> llvalue */
CAMLprim LLVMValueRef llvm_const_of_int64(LLVMTypeRef IntTy, value N,
                                          value SExt) {
  return LLVMConstInt(IntTy, Int64_val(N), Bool_val(SExt));
}

/* llvalue -> Int64.t */
CAMLprim value llvm_int64_of_const(LLVMValueRef Const)
{
  CAMLparam0();
  if (LLVMIsAConstantInt(Const) &&
      LLVMGetIntTypeWidth(LLVMTypeOf(Const)) <= 64) {
    value Option = alloc(1, 0);
    Field(Option, 0) = caml_copy_int64(LLVMConstIntGetSExtValue(Const));
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* lltype -> string -> int -> llvalue */
CAMLprim LLVMValueRef llvm_const_int_of_string(LLVMTypeRef IntTy, value S,
                                               value Radix) {
  return LLVMConstIntOfStringAndSize(IntTy, String_val(S), caml_string_length(S),
                                     Int_val(Radix));
}

/* lltype -> float -> llvalue */
CAMLprim LLVMValueRef llvm_const_float(LLVMTypeRef RealTy, value N) {
  return LLVMConstReal(RealTy, Double_val(N));
}


/* llvalue -> float */
CAMLprim value llvm_float_of_const(LLVMValueRef Const)
{
  CAMLparam0();
  CAMLlocal1(Option);
  LLVMBool LosesInfo;
  double Result;

  if (LLVMIsAConstantFP(Const)) {
    Result = LLVMConstRealGetDouble(Const, &LosesInfo);
    if (LosesInfo)
        CAMLreturn(Val_int(0));

    Option = alloc(1, 0);
    Field(Option, 0) = caml_copy_double(Result);
    CAMLreturn(Option);
  }

  CAMLreturn(Val_int(0));
}

/* lltype -> string -> llvalue */
CAMLprim LLVMValueRef llvm_const_float_of_string(LLVMTypeRef RealTy, value S) {
  return LLVMConstRealOfStringAndSize(RealTy, String_val(S),
                                      caml_string_length(S));
}

/*--... Operations on composite constants ..................................--*/

/* llcontext -> string -> llvalue */
CAMLprim LLVMValueRef llvm_const_string(LLVMContextRef Context, value Str,
                                        value NullTerminate) {
  return LLVMConstStringInContext(Context, String_val(Str), string_length(Str),
                                  1);
}

/* llcontext -> string -> llvalue */
CAMLprim LLVMValueRef llvm_const_stringz(LLVMContextRef Context, value Str,
                                         value NullTerminate) {
  return LLVMConstStringInContext(Context, String_val(Str), string_length(Str),
                                  0);
}

/* lltype -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_array(LLVMTypeRef ElementTy,
                                               value ElementVals) {
  return LLVMConstArray(ElementTy, (LLVMValueRef*) Op_val(ElementVals),
                        Wosize_val(ElementVals));
}

/* llcontext -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_struct(LLVMContextRef C, value ElementVals) {
  return LLVMConstStructInContext(C, (LLVMValueRef *) Op_val(ElementVals),
                                  Wosize_val(ElementVals), 0);
}

/* lltype -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_named_struct(LLVMTypeRef Ty, value ElementVals) {
    return LLVMConstNamedStruct(Ty, (LLVMValueRef *) Op_val(ElementVals),  Wosize_val(ElementVals));
}

/* llcontext -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_packed_struct(LLVMContextRef C,
                                               value ElementVals) {
  return LLVMConstStructInContext(C, (LLVMValueRef *) Op_val(ElementVals),
                                  Wosize_val(ElementVals), 1);
}

/* llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_vector(value ElementVals) {
  return LLVMConstVector((LLVMValueRef*) Op_val(ElementVals),
                         Wosize_val(ElementVals));
}

/* llvalue -> string option */
CAMLprim value llvm_string_of_const(LLVMValueRef Const) {
  const char *S;
  size_t Len;
  CAMLparam0();
  CAMLlocal2(Option, Str);

  if(LLVMIsAConstantDataSequential(Const) && LLVMIsConstantString(Const)) {
    S = LLVMGetAsString(Const, &Len);
    Str = caml_alloc_string(Len);
    memcpy(String_val(Str), S, Len);

    Option = alloc(1, 0);
    Field(Option, 0) = Str;
    CAMLreturn(Option);
  } else {
    CAMLreturn(Val_int(0));
  }
}

/* llvalue -> int -> llvalue */
CAMLprim LLVMValueRef llvm_const_element(LLVMValueRef Const, value N) {
  return LLVMGetElementAsConstant(Const, Int_val(N));
}

/*--... Constant expressions ...............................................--*/

/* Icmp.t -> llvalue -> llvalue -> llvalue */
CAMLprim LLVMValueRef llvm_const_icmp(value Pred,
                                      LLVMValueRef LHSConstant,
                                      LLVMValueRef RHSConstant) {
  return LLVMConstICmp(Int_val(Pred) + LLVMIntEQ, LHSConstant, RHSConstant);
}

/* Fcmp.t -> llvalue -> llvalue -> llvalue */
CAMLprim LLVMValueRef llvm_const_fcmp(value Pred,
                                      LLVMValueRef LHSConstant,
                                      LLVMValueRef RHSConstant) {
  return LLVMConstFCmp(Int_val(Pred), LHSConstant, RHSConstant);
}

/* llvalue -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_gep(LLVMValueRef ConstantVal, value Indices) {
  return LLVMConstGEP(ConstantVal, (LLVMValueRef*) Op_val(Indices),
                      Wosize_val(Indices));
}

/* llvalue -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_const_in_bounds_gep(LLVMValueRef ConstantVal,
                                               value Indices) {
  return LLVMConstInBoundsGEP(ConstantVal, (LLVMValueRef*) Op_val(Indices),
                              Wosize_val(Indices));
}

/* llvalue -> lltype -> is_signed:bool -> llvalue */
CAMLprim LLVMValueRef llvm_const_intcast(LLVMValueRef CV, LLVMTypeRef T,
                                         value IsSigned) {
  return LLVMConstIntCast(CV, T, Bool_val(IsSigned));
}

/* llvalue -> int array -> llvalue */
CAMLprim LLVMValueRef llvm_const_extractvalue(LLVMValueRef Aggregate,
                                              value Indices) {
  CAMLparam1(Indices);
  int size = Wosize_val(Indices);
  int i;
  LLVMValueRef result;

  unsigned* idxs = (unsigned*)malloc(size * sizeof(unsigned));
  for (i = 0; i < size; i++) {
    idxs[i] = Int_val(Field(Indices, i));
  }

  result = LLVMConstExtractValue(Aggregate, idxs, size);
  free(idxs);
  CAMLreturnT(LLVMValueRef, result);
}

/* llvalue -> llvalue -> int array -> llvalue */
CAMLprim LLVMValueRef llvm_const_insertvalue(LLVMValueRef Aggregate,
                                             LLVMValueRef Val, value Indices) {
  CAMLparam1(Indices);
  int size = Wosize_val(Indices);
  int i;
  LLVMValueRef result;

  unsigned* idxs = (unsigned*)malloc(size * sizeof(unsigned));
  for (i = 0; i < size; i++) {
    idxs[i] = Int_val(Field(Indices, i));
  }

  result = LLVMConstInsertValue(Aggregate, Val, idxs, size);
  free(idxs);
  CAMLreturnT(LLVMValueRef, result);
}

/* lltype -> string -> string -> bool -> bool -> llvalue */
CAMLprim LLVMValueRef llvm_const_inline_asm(LLVMTypeRef Ty, value Asm,
                                     value Constraints, value HasSideEffects,
                                     value IsAlignStack) {
  return LLVMConstInlineAsm(Ty, String_val(Asm), String_val(Constraints),
                            Bool_val(HasSideEffects), Bool_val(IsAlignStack));
}

/*--... Operations on global variables, functions, and aliases (globals) ...--*/

/* llvalue -> bool */
CAMLprim value llvm_is_declaration(LLVMValueRef Global) {
  return Val_bool(LLVMIsDeclaration(Global));
}

/* llvalue -> Linkage.t */
CAMLprim value llvm_linkage(LLVMValueRef Global) {
  return Val_int(LLVMGetLinkage(Global));
}

/* Linkage.t -> llvalue -> unit */
CAMLprim value llvm_set_linkage(value Linkage, LLVMValueRef Global) {
  LLVMSetLinkage(Global, Int_val(Linkage));
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_unnamed_addr(LLVMValueRef Global) {
  return Val_bool(LLVMHasUnnamedAddr(Global));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_unnamed_addr(value UseUnnamedAddr, LLVMValueRef Global) {
  LLVMSetUnnamedAddr(Global, Bool_val(UseUnnamedAddr));
  return Val_unit;
}

/* llvalue -> string */
CAMLprim value llvm_section(LLVMValueRef Global) {
  return caml_copy_string(LLVMGetSection(Global));
}

/* string -> llvalue -> unit */
CAMLprim value llvm_set_section(value Section, LLVMValueRef Global) {
  LLVMSetSection(Global, String_val(Section));
  return Val_unit;
}

/* llvalue -> Visibility.t */
CAMLprim value llvm_visibility(LLVMValueRef Global) {
  return Val_int(LLVMGetVisibility(Global));
}

/* Visibility.t -> llvalue -> unit */
CAMLprim value llvm_set_visibility(value Viz, LLVMValueRef Global) {
  LLVMSetVisibility(Global, Int_val(Viz));
  return Val_unit;
}

/* llvalue -> DLLStorageClass.t */
CAMLprim value llvm_dll_storage_class(LLVMValueRef Global) {
  return Val_int(LLVMGetDLLStorageClass(Global));
}

/* DLLStorageClass.t -> llvalue -> unit */
CAMLprim value llvm_set_dll_storage_class(value Viz, LLVMValueRef Global) {
  LLVMSetDLLStorageClass(Global, Int_val(Viz));
  return Val_unit;
}

/* llvalue -> int */
CAMLprim value llvm_alignment(LLVMValueRef Global) {
  return Val_int(LLVMGetAlignment(Global));
}

/* int -> llvalue -> unit */
CAMLprim value llvm_set_alignment(value Bytes, LLVMValueRef Global) {
  LLVMSetAlignment(Global, Int_val(Bytes));
  return Val_unit;
}

/*--... Operations on uses .................................................--*/

/* llvalue -> lluse option */
CAMLprim value llvm_use_begin(LLVMValueRef Val) {
  CAMLparam0();
  LLVMUseRef First;
  if ((First = LLVMGetFirstUse(Val))) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) First;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* lluse -> lluse option */
CAMLprim value llvm_use_succ(LLVMUseRef U) {
  CAMLparam0();
  LLVMUseRef Next;
  if ((Next = LLVMGetNextUse(U))) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) Next;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* lluse -> llvalue */
CAMLprim LLVMValueRef llvm_user(LLVMUseRef UR) {
  return LLVMGetUser(UR);
}

/* lluse -> llvalue */
CAMLprim LLVMValueRef llvm_used_value(LLVMUseRef UR) {
  return LLVMGetUsedValue(UR);
}

/*--... Operations on global variables .....................................--*/

DEFINE_ITERATORS(global, Global, LLVMModuleRef, LLVMValueRef,
                 LLVMGetGlobalParent)

/* lltype -> string -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_declare_global(LLVMTypeRef Ty, value Name,
                                          LLVMModuleRef M) {
  LLVMValueRef GlobalVar;
  if ((GlobalVar = LLVMGetNamedGlobal(M, String_val(Name)))) {
    if (LLVMGetElementType(LLVMTypeOf(GlobalVar)) != Ty)
      return LLVMConstBitCast(GlobalVar, LLVMPointerType(Ty, 0));
    return GlobalVar;
  }
  return LLVMAddGlobal(M, Ty, String_val(Name));
}

/* lltype -> string -> int -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_declare_qualified_global(LLVMTypeRef Ty, value Name,
                                                    value AddressSpace,
                                                    LLVMModuleRef M) {
  LLVMValueRef GlobalVar;
  if ((GlobalVar = LLVMGetNamedGlobal(M, String_val(Name)))) {
    if (LLVMGetElementType(LLVMTypeOf(GlobalVar)) != Ty)
      return LLVMConstBitCast(GlobalVar,
                              LLVMPointerType(Ty, Int_val(AddressSpace)));
    return GlobalVar;
  }
  return LLVMAddGlobalInAddressSpace(M, Ty, String_val(Name),
                                     Int_val(AddressSpace));
}

/* string -> llmodule -> llvalue option */
CAMLprim value llvm_lookup_global(value Name, LLVMModuleRef M) {
  CAMLparam1(Name);
  LLVMValueRef GlobalVar;
  if ((GlobalVar = LLVMGetNamedGlobal(M, String_val(Name)))) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) GlobalVar;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* string -> llvalue -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_define_global(value Name, LLVMValueRef Initializer,
                                         LLVMModuleRef M) {
  LLVMValueRef GlobalVar = LLVMAddGlobal(M, LLVMTypeOf(Initializer),
                                         String_val(Name));
  LLVMSetInitializer(GlobalVar, Initializer);
  return GlobalVar;
}

/* string -> llvalue -> int -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_define_qualified_global(value Name,
                                                   LLVMValueRef Initializer,
                                                   value AddressSpace,
                                                   LLVMModuleRef M) {
  LLVMValueRef GlobalVar = LLVMAddGlobalInAddressSpace(M,
                                                       LLVMTypeOf(Initializer),
                                                       String_val(Name),
                                                       Int_val(AddressSpace));
  LLVMSetInitializer(GlobalVar, Initializer);
  return GlobalVar;
}

/* llvalue -> unit */
CAMLprim value llvm_delete_global(LLVMValueRef GlobalVar) {
  LLVMDeleteGlobal(GlobalVar);
  return Val_unit;
}

/* llvalue -> llvalue -> unit */
CAMLprim value llvm_set_initializer(LLVMValueRef ConstantVal,
                                    LLVMValueRef GlobalVar) {
  LLVMSetInitializer(GlobalVar, ConstantVal);
  return Val_unit;
}

/* llvalue -> unit */
CAMLprim value llvm_remove_initializer(LLVMValueRef GlobalVar) {
  LLVMSetInitializer(GlobalVar, NULL);
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_thread_local(LLVMValueRef GlobalVar) {
  return Val_bool(LLVMIsThreadLocal(GlobalVar));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_thread_local(value IsThreadLocal,
                                     LLVMValueRef GlobalVar) {
  LLVMSetThreadLocal(GlobalVar, Bool_val(IsThreadLocal));
  return Val_unit;
}

/* llvalue -> ThreadLocalMode.t */
CAMLprim value llvm_thread_local_mode(LLVMValueRef GlobalVar) {
  return Val_int(LLVMGetThreadLocalMode(GlobalVar));
}

/* ThreadLocalMode.t -> llvalue -> unit */
CAMLprim value llvm_set_thread_local_mode(value ThreadLocalMode,
                                          LLVMValueRef GlobalVar) {
  LLVMSetThreadLocalMode(GlobalVar, Int_val(ThreadLocalMode));
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_externally_initialized(LLVMValueRef GlobalVar) {
  return Val_bool(LLVMIsExternallyInitialized(GlobalVar));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_externally_initialized(value IsExternallyInitialized,
                                               LLVMValueRef GlobalVar) {
  LLVMSetExternallyInitialized(GlobalVar, Bool_val(IsExternallyInitialized));
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_global_constant(LLVMValueRef GlobalVar) {
  return Val_bool(LLVMIsGlobalConstant(GlobalVar));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_global_constant(value Flag, LLVMValueRef GlobalVar) {
  LLVMSetGlobalConstant(GlobalVar, Bool_val(Flag));
  return Val_unit;
}

/*--... Operations on aliases ..............................................--*/

CAMLprim LLVMValueRef llvm_add_alias(LLVMModuleRef M, LLVMTypeRef Ty,
                                     LLVMValueRef Aliasee, value Name) {
  return LLVMAddAlias(M, Ty, Aliasee, String_val(Name));
}

/*--... Operations on functions ............................................--*/

DEFINE_ITERATORS(function, Function, LLVMModuleRef, LLVMValueRef,
                 LLVMGetGlobalParent)

/* string -> lltype -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_declare_function(value Name, LLVMTypeRef Ty,
                                            LLVMModuleRef M) {
  LLVMValueRef Fn;
  if ((Fn = LLVMGetNamedFunction(M, String_val(Name)))) {
    if (LLVMGetElementType(LLVMTypeOf(Fn)) != Ty)
      return LLVMConstBitCast(Fn, LLVMPointerType(Ty, 0));
    return Fn;
  }
  return LLVMAddFunction(M, String_val(Name), Ty);
}

/* string -> llmodule -> llvalue option */
CAMLprim value llvm_lookup_function(value Name, LLVMModuleRef M) {
  CAMLparam1(Name);
  LLVMValueRef Fn;
  if ((Fn = LLVMGetNamedFunction(M, String_val(Name)))) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) Fn;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* string -> lltype -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_define_function(value Name, LLVMTypeRef Ty,
                                           LLVMModuleRef M) {
  LLVMValueRef Fn = LLVMAddFunction(M, String_val(Name), Ty);
  LLVMAppendBasicBlockInContext(LLVMGetTypeContext(Ty), Fn, "entry");
  return Fn;
}

/* llvalue -> unit */
CAMLprim value llvm_delete_function(LLVMValueRef Fn) {
  LLVMDeleteFunction(Fn);
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_intrinsic(LLVMValueRef Fn) {
  return Val_bool(LLVMGetIntrinsicID(Fn));
}

/* llvalue -> int */
CAMLprim value llvm_function_call_conv(LLVMValueRef Fn) {
  return Val_int(LLVMGetFunctionCallConv(Fn));
}

/* int -> llvalue -> unit */
CAMLprim value llvm_set_function_call_conv(value Id, LLVMValueRef Fn) {
  LLVMSetFunctionCallConv(Fn, Int_val(Id));
  return Val_unit;
}

/* llvalue -> string option */
CAMLprim value llvm_gc(LLVMValueRef Fn) {
  const char *GC;
  CAMLparam0();
  CAMLlocal2(Name, Option);

  if ((GC = LLVMGetGC(Fn))) {
    Name = caml_copy_string(GC);

    Option = alloc(1, 0);
    Field(Option, 0) = Name;
    CAMLreturn(Option);
  } else {
    CAMLreturn(Val_int(0));
  }
}

/* string option -> llvalue -> unit */
CAMLprim value llvm_set_gc(value GC, LLVMValueRef Fn) {
  LLVMSetGC(Fn, GC == Val_int(0)? 0 : String_val(Field(GC, 0)));
  return Val_unit;
}

/* llvalue -> llattribute -> int -> unit */
CAMLprim value llvm_add_function_attr(LLVMValueRef F, LLVMAttributeRef A,
                                      value Index) {
  LLVMAddAttributeAtIndex(F, Int_val(Index), A);
  return Val_unit;
}

/* llvalue -> int -> llattribute array */
CAMLprim value llvm_function_attrs(LLVMValueRef F, value Index) {
  unsigned Length = LLVMGetAttributeCountAtIndex(F, Int_val(Index));
  value Array = caml_alloc(Length, 0);
  LLVMGetAttributesAtIndex(F, Int_val(Index),
                           (LLVMAttributeRef *) Op_val(Array));
  return Array;
}

/* llvalue -> llattrkind -> int -> unit */
CAMLprim value llvm_remove_enum_function_attr(LLVMValueRef F, value Kind,
                                              value Index) {
  LLVMRemoveEnumAttributeAtIndex(F, Int_val(Index), Int_val(Kind));
  return Val_unit;
}

/* llvalue -> string -> int -> unit */
CAMLprim value llvm_remove_string_function_attr(LLVMValueRef F, value Kind,
                                                value Index) {
  LLVMRemoveStringAttributeAtIndex(F, Int_val(Index), String_val(Kind),
                                   caml_string_length(Kind));
  return Val_unit;
}

/*--... Operations on parameters ...........................................--*/

DEFINE_ITERATORS(param, Param, LLVMValueRef, LLVMValueRef, LLVMGetParamParent)

/* llvalue -> int -> llvalue */
CAMLprim LLVMValueRef llvm_param(LLVMValueRef Fn, value Index) {
  return LLVMGetParam(Fn, Int_val(Index));
}

/* llvalue -> llvalue */
CAMLprim value llvm_params(LLVMValueRef Fn) {
  value Params = alloc(LLVMCountParams(Fn), 0);
  LLVMGetParams(Fn, (LLVMValueRef *) Op_val(Params));
  return Params;
}

/*--... Operations on basic blocks .........................................--*/

DEFINE_ITERATORS(
  block, BasicBlock, LLVMValueRef, LLVMBasicBlockRef, LLVMGetBasicBlockParent)

/* llbasicblock -> llvalue option */
CAMLprim value llvm_block_terminator(LLVMBasicBlockRef Block)
{
  CAMLparam0();
  LLVMValueRef Term = LLVMGetBasicBlockTerminator(Block);
  if (Term) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) Term;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* llvalue -> llbasicblock array */
CAMLprim value llvm_basic_blocks(LLVMValueRef Fn) {
  value MLArray = alloc(LLVMCountBasicBlocks(Fn), 0);
  LLVMGetBasicBlocks(Fn, (LLVMBasicBlockRef *) Op_val(MLArray));
  return MLArray;
}

/* llbasicblock -> unit */
CAMLprim value llvm_delete_block(LLVMBasicBlockRef BB) {
  LLVMDeleteBasicBlock(BB);
  return Val_unit;
}

/* llbasicblock -> unit */
CAMLprim value llvm_remove_block(LLVMBasicBlockRef BB) {
  LLVMRemoveBasicBlockFromParent(BB);
  return Val_unit;
}

/* llbasicblock -> llbasicblock -> unit */
CAMLprim value llvm_move_block_before(LLVMBasicBlockRef Pos, LLVMBasicBlockRef BB) {
  LLVMMoveBasicBlockBefore(BB, Pos);
  return Val_unit;
}

/* llbasicblock -> llbasicblock -> unit */
CAMLprim value llvm_move_block_after(LLVMBasicBlockRef Pos, LLVMBasicBlockRef BB) {
  LLVMMoveBasicBlockAfter(BB, Pos);
  return Val_unit;
}

/* string -> llvalue -> llbasicblock */
CAMLprim LLVMBasicBlockRef llvm_append_block(LLVMContextRef Context, value Name,
                                             LLVMValueRef Fn) {
  return LLVMAppendBasicBlockInContext(Context, Fn, String_val(Name));
}

/* string -> llbasicblock -> llbasicblock */
CAMLprim LLVMBasicBlockRef llvm_insert_block(LLVMContextRef Context, value Name,
                                             LLVMBasicBlockRef BB) {
  return LLVMInsertBasicBlockInContext(Context, BB, String_val(Name));
}

/* llvalue -> bool */
CAMLprim value llvm_value_is_block(LLVMValueRef Val) {
  return Val_bool(LLVMValueIsBasicBlock(Val));
}

/*--... Operations on instructions .........................................--*/

DEFINE_ITERATORS(instr, Instruction, LLVMBasicBlockRef, LLVMValueRef,
                 LLVMGetInstructionParent)

/* llvalue -> Opcode.t */
CAMLprim value llvm_instr_get_opcode(LLVMValueRef Inst) {
  LLVMOpcode o;
  if (!LLVMIsAInstruction(Inst))
      failwith("Not an instruction");
  o = LLVMGetInstructionOpcode(Inst);
  assert (o <= LLVMLandingPad);
  return Val_int(o);
}

/* llvalue -> ICmp.t option */
CAMLprim value llvm_instr_icmp_predicate(LLVMValueRef Val) {
  CAMLparam0();
  int x = LLVMGetICmpPredicate(Val);
  if (x) {
    value Option = alloc(1, 0);
    Field(Option, 0) = Val_int(x - LLVMIntEQ);
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* llvalue -> FCmp.t option */
CAMLprim value llvm_instr_fcmp_predicate(LLVMValueRef Val) {
  CAMLparam0();
  int x = LLVMGetFCmpPredicate(Val);
  if (x) {
    value Option = alloc(1, 0);
    Field(Option, 0) = Val_int(x - LLVMRealPredicateFalse);
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* llvalue -> llvalue */
CAMLprim LLVMValueRef llvm_instr_clone(LLVMValueRef Inst) {
  if (!LLVMIsAInstruction(Inst))
      failwith("Not an instruction");
  return LLVMInstructionClone(Inst);
}


/*--... Operations on call sites ...........................................--*/

/* llvalue -> int */
CAMLprim value llvm_instruction_call_conv(LLVMValueRef Inst) {
  return Val_int(LLVMGetInstructionCallConv(Inst));
}

/* int -> llvalue -> unit */
CAMLprim value llvm_set_instruction_call_conv(value CC, LLVMValueRef Inst) {
  LLVMSetInstructionCallConv(Inst, Int_val(CC));
  return Val_unit;
}

/* llvalue -> llattribute -> int -> unit */
CAMLprim value llvm_add_call_site_attr(LLVMValueRef F, LLVMAttributeRef A,
                                       value Index) {
  LLVMAddCallSiteAttribute(F, Int_val(Index), A);
  return Val_unit;
}

/* llvalue -> int -> llattribute array */
CAMLprim value llvm_call_site_attrs(LLVMValueRef F, value Index) {
  unsigned Count = LLVMGetCallSiteAttributeCount(F, Int_val(Index));
  value Array = caml_alloc(Count, 0);
  LLVMGetCallSiteAttributes(F, Int_val(Index),
                            (LLVMAttributeRef *)Op_val(Array));
  return Array;
}

/* llvalue -> llattrkind -> int -> unit */
CAMLprim value llvm_remove_enum_call_site_attr(LLVMValueRef F, value Kind,
                                               value Index) {
  LLVMRemoveCallSiteEnumAttribute(F, Int_val(Index), Int_val(Kind));
  return Val_unit;
}

/* llvalue -> string -> int -> unit */
CAMLprim value llvm_remove_string_call_site_attr(LLVMValueRef F, value Kind,
                                                 value Index) {
  LLVMRemoveCallSiteStringAttribute(F, Int_val(Index), String_val(Kind),
                                    caml_string_length(Kind));
  return Val_unit;
}

/*--... Operations on call instructions (only) .............................--*/

/* llvalue -> bool */
CAMLprim value llvm_is_tail_call(LLVMValueRef CallInst) {
  return Val_bool(LLVMIsTailCall(CallInst));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_tail_call(value IsTailCall,
                                  LLVMValueRef CallInst) {
  LLVMSetTailCall(CallInst, Bool_val(IsTailCall));
  return Val_unit;
}

/*--... Operations on load/store instructions (only)........................--*/

/* llvalue -> bool */
CAMLprim value llvm_is_volatile(LLVMValueRef MemoryInst) {
  return Val_bool(LLVMGetVolatile(MemoryInst));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_volatile(value IsVolatile,
                                  LLVMValueRef MemoryInst) {
  LLVMSetVolatile(MemoryInst, Bool_val(IsVolatile));
  return Val_unit;
}


/*--.. Operations on terminators ...........................................--*/

/* llvalue -> int -> llbasicblock */
CAMLprim LLVMBasicBlockRef llvm_successor(LLVMValueRef V, value I) {
  return LLVMGetSuccessor(V, Int_val(I));
}

/* llvalue -> int -> llvalue -> unit */
CAMLprim value llvm_set_successor(LLVMValueRef U, value I, LLVMBasicBlockRef B) {
  LLVMSetSuccessor(U, Int_val(I), B);
  return Val_unit;
}

/* llvalue -> int */
CAMLprim value llvm_num_successors(LLVMValueRef V) {
  return Val_int(LLVMGetNumSuccessors(V));
}

/*--.. Operations on branch ................................................--*/

/* llvalue -> llvalue */
CAMLprim LLVMValueRef llvm_condition(LLVMValueRef V) {
  return LLVMGetCondition(V);
}

/* llvalue -> llvalue -> unit */
CAMLprim value llvm_set_condition(LLVMValueRef B, LLVMValueRef C) {
  LLVMSetCondition(B, C);
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_conditional(LLVMValueRef V) {
  return Val_bool(LLVMIsConditional(V));
}

/*--... Operations on phi nodes ............................................--*/

/* (llvalue * llbasicblock) -> llvalue -> unit */
CAMLprim value llvm_add_incoming(value Incoming, LLVMValueRef PhiNode) {
  LLVMAddIncoming(PhiNode,
                  (LLVMValueRef*) &Field(Incoming, 0),
                  (LLVMBasicBlockRef*) &Field(Incoming, 1),
                  1);
  return Val_unit;
}

/* llvalue -> (llvalue * llbasicblock) list */
CAMLprim value llvm_incoming(LLVMValueRef PhiNode) {
  unsigned I;
  CAMLparam0();
  CAMLlocal3(Hd, Tl, Tmp);

  /* Build a tuple list of them. */
  Tl = Val_int(0);
  for (I = LLVMCountIncoming(PhiNode); I != 0; ) {
    Hd = alloc(2, 0);
    Store_field(Hd, 0, (value) LLVMGetIncomingValue(PhiNode, --I));
    Store_field(Hd, 1, (value) LLVMGetIncomingBlock(PhiNode, I));

    Tmp = alloc(2, 0);
    Store_field(Tmp, 0, Hd);
    Store_field(Tmp, 1, Tl);
    Tl = Tmp;
  }

  CAMLreturn(Tl);
}

/* llvalue -> unit */
CAMLprim value llvm_delete_instruction(LLVMValueRef Instruction) {
  LLVMInstructionEraseFromParent(Instruction);
  return Val_unit;
}

/*===-- Instruction builders ----------------------------------------------===*/

#define Builder_val(v)  (*(LLVMBuilderRef *)(Data_custom_val(v)))

static void llvm_finalize_builder(value B) {
  LLVMDisposeBuilder(Builder_val(B));
}

static struct custom_operations builder_ops = {
  (char *) "Llvm.llbuilder",
  llvm_finalize_builder,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default,
  custom_compare_ext_default
};

static value alloc_builder(LLVMBuilderRef B) {
  value V = alloc_custom(&builder_ops, sizeof(LLVMBuilderRef), 0, 1);
  Builder_val(V) = B;
  return V;
}

/* llcontext -> llbuilder */
CAMLprim value llvm_builder(LLVMContextRef C) {
  return alloc_builder(LLVMCreateBuilderInContext(C));
}

/* (llbasicblock, llvalue) llpos -> llbuilder -> unit */
CAMLprim value llvm_position_builder(value Pos, value B) {
  if (Tag_val(Pos) == 0) {
    LLVMBasicBlockRef BB = (LLVMBasicBlockRef) Op_val(Field(Pos, 0));
    LLVMPositionBuilderAtEnd(Builder_val(B), BB);
  } else {
    LLVMValueRef I = (LLVMValueRef) Op_val(Field(Pos, 0));
    LLVMPositionBuilderBefore(Builder_val(B), I);
  }
  return Val_unit;
}

/* llbuilder -> llbasicblock */
CAMLprim LLVMBasicBlockRef llvm_insertion_block(value B) {
  LLVMBasicBlockRef InsertBlock = LLVMGetInsertBlock(Builder_val(B));
  if (!InsertBlock)
    caml_raise_not_found();
  return InsertBlock;
}

/* llvalue -> string -> llbuilder -> unit */
CAMLprim value llvm_insert_into_builder(LLVMValueRef I, value Name, value B) {
  LLVMInsertIntoBuilderWithName(Builder_val(B), I, String_val(Name));
  return Val_unit;
}

/*--... Metadata ...........................................................--*/

/* llbuilder -> llvalue -> unit */
CAMLprim value llvm_set_current_debug_location(value B, LLVMValueRef V) {
  LLVMSetCurrentDebugLocation(Builder_val(B), V);
  return Val_unit;
}

/* llbuilder -> unit */
CAMLprim value llvm_clear_current_debug_location(value B) {
  LLVMSetCurrentDebugLocation(Builder_val(B), NULL);
  return Val_unit;
}

/* llbuilder -> llvalue option */
CAMLprim value llvm_current_debug_location(value B) {
  CAMLparam0();
  LLVMValueRef L;
  if ((L = LLVMGetCurrentDebugLocation(Builder_val(B)))) {
    value Option = alloc(1, 0);
    Field(Option, 0) = (value) L;
    CAMLreturn(Option);
  }
  CAMLreturn(Val_int(0));
}

/* llbuilder -> llvalue -> unit */
CAMLprim value llvm_set_inst_debug_location(value B, LLVMValueRef V) {
  LLVMSetInstDebugLocation(Builder_val(B), V);
  return Val_unit;
}


/*--... Terminators ........................................................--*/

/* llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ret_void(value B) {
  return LLVMBuildRetVoid(Builder_val(B));
}

/* llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ret(LLVMValueRef Val, value B) {
  return LLVMBuildRet(Builder_val(B), Val);
}

/* llvalue array -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_aggregate_ret(value RetVals, value B) {
  return LLVMBuildAggregateRet(Builder_val(B), (LLVMValueRef *) Op_val(RetVals),
                               Wosize_val(RetVals));
}

/* llbasicblock -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_br(LLVMBasicBlockRef BB, value B) {
  return LLVMBuildBr(Builder_val(B), BB);
}

/* llvalue -> llbasicblock -> llbasicblock -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_cond_br(LLVMValueRef If,
                                         LLVMBasicBlockRef Then,
                                         LLVMBasicBlockRef Else,
                                         value B) {
  return LLVMBuildCondBr(Builder_val(B), If, Then, Else);
}

/* llvalue -> llbasicblock -> int -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_switch(LLVMValueRef Of,
                                        LLVMBasicBlockRef Else,
                                        value EstimatedCount,
                                        value B) {
  return LLVMBuildSwitch(Builder_val(B), Of, Else, Int_val(EstimatedCount));
}

/* lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_malloc(LLVMTypeRef Ty, value Name,
                                        value B)
{
  return LLVMBuildMalloc(Builder_val(B), Ty, String_val(Name));
}

/* lltype -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_array_malloc(LLVMTypeRef Ty,
                                              LLVMValueRef Val,
                                              value Name, value B)
{
  return LLVMBuildArrayMalloc(Builder_val(B), Ty, Val, String_val(Name));
}

/* llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_free(LLVMValueRef P, value B)
{
  return LLVMBuildFree(Builder_val(B), P);
}

/* llvalue -> llvalue -> llbasicblock -> unit */
CAMLprim value llvm_add_case(LLVMValueRef Switch, LLVMValueRef OnVal,
                             LLVMBasicBlockRef Dest) {
  LLVMAddCase(Switch, OnVal, Dest);
  return Val_unit;
}

/* llvalue -> llbasicblock -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_indirect_br(LLVMValueRef Addr,
                                             value EstimatedDests,
                                             value B) {
  return LLVMBuildIndirectBr(Builder_val(B), Addr, EstimatedDests);
}

/* llvalue -> llvalue -> llbasicblock -> unit */
CAMLprim value llvm_add_destination(LLVMValueRef IndirectBr,
                                    LLVMBasicBlockRef Dest) {
  LLVMAddDestination(IndirectBr, Dest);
  return Val_unit;
}

/* llvalue -> llvalue array -> llbasicblock -> llbasicblock -> string ->
   llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_invoke_nat(LLVMValueRef Fn, value Args,
                                            LLVMBasicBlockRef Then,
                                            LLVMBasicBlockRef Catch,
                                            value Name, value B) {
  return LLVMBuildInvoke(Builder_val(B), Fn, (LLVMValueRef *) Op_val(Args),
                         Wosize_val(Args), Then, Catch, String_val(Name));
}

/* llvalue -> llvalue array -> llbasicblock -> llbasicblock -> string ->
   llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_invoke_bc(value Args[], int NumArgs) {
  return llvm_build_invoke_nat((LLVMValueRef) Args[0], Args[1],
                               (LLVMBasicBlockRef) Args[2],
                               (LLVMBasicBlockRef) Args[3],
                               Args[4], Args[5]);
}

/* lltype -> llvalue -> int -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_landingpad(LLVMTypeRef Ty, LLVMValueRef PersFn,
                                            value NumClauses,  value Name,
                                            value B) {
    return LLVMBuildLandingPad(Builder_val(B), Ty, PersFn, Int_val(NumClauses),
                               String_val(Name));
}

/* llvalue -> llvalue -> unit */
CAMLprim value llvm_add_clause(LLVMValueRef LandingPadInst, LLVMValueRef ClauseVal)
{
    LLVMAddClause(LandingPadInst, ClauseVal);
    return Val_unit;
}


/* llvalue -> bool -> unit */
CAMLprim value llvm_set_cleanup(LLVMValueRef LandingPadInst, value flag)
{
    LLVMSetCleanup(LandingPadInst, Bool_val(flag));
    return Val_unit;
}

/* llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_resume(LLVMValueRef Exn, value B)
{
    return LLVMBuildResume(Builder_val(B), Exn);
}

/* llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_unreachable(value B) {
  return LLVMBuildUnreachable(Builder_val(B));
}

/*--... Arithmetic .........................................................--*/

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_add(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildAdd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nsw_add(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildNSWAdd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nuw_add(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildNUWAdd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fadd(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFAdd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sub(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildSub(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nsw_sub(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildNSWSub(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nuw_sub(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildNUWSub(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fsub(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFSub(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_mul(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildMul(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nsw_mul(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildNSWMul(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nuw_mul(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildNUWMul(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fmul(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFMul(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_udiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildUDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sdiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildSDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_exact_sdiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                            value Name, value B) {
  return LLVMBuildExactSDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fdiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_urem(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildURem(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_srem(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildSRem(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_frem(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFRem(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_shl(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildShl(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_lshr(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildLShr(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ashr(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildAShr(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_and(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildAnd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_or(LLVMValueRef LHS, LLVMValueRef RHS,
                                    value Name, value B) {
  return LLVMBuildOr(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_xor(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildXor(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_neg(LLVMValueRef X,
                                     value Name, value B) {
  return LLVMBuildNeg(Builder_val(B), X, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nsw_neg(LLVMValueRef X,
                                         value Name, value B) {
  return LLVMBuildNSWNeg(Builder_val(B), X, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_nuw_neg(LLVMValueRef X,
                                         value Name, value B) {
  return LLVMBuildNUWNeg(Builder_val(B), X, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fneg(LLVMValueRef X,
                                     value Name, value B) {
  return LLVMBuildFNeg(Builder_val(B), X, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_not(LLVMValueRef X,
                                     value Name, value B) {
  return LLVMBuildNot(Builder_val(B), X, String_val(Name));
}

/*--... Memory .............................................................--*/

/* lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_alloca(LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildAlloca(Builder_val(B), Ty, String_val(Name));
}

/* lltype -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_array_alloca(LLVMTypeRef Ty, LLVMValueRef Size,
                                              value Name, value B) {
  return LLVMBuildArrayAlloca(Builder_val(B), Ty, Size, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_load(LLVMValueRef Pointer,
                                      value Name, value B) {
  return LLVMBuildLoad(Builder_val(B), Pointer, String_val(Name));
}

/* llvalue -> llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_store(LLVMValueRef Value, LLVMValueRef Pointer,
                                       value B) {
  return LLVMBuildStore(Builder_val(B), Value, Pointer);
}

/* AtomicRMWBinOp.t -> llvalue -> llvalue -> AtomicOrdering.t ->
   bool -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_atomicrmw_native(value BinOp, LLVMValueRef Ptr,
                                                  LLVMValueRef Val, value Ord,
                                                  value ST, value Name, value B) {
  LLVMValueRef Instr;
  Instr = LLVMBuildAtomicRMW(Builder_val(B), Int_val(BinOp),
                             Ptr, Val, Int_val(Ord), Bool_val(ST));
  LLVMSetValueName(Instr, String_val(Name));
  return Instr;
}

CAMLprim LLVMValueRef llvm_build_atomicrmw_bytecode(value *argv, int argn) {
  return llvm_build_atomicrmw_native(argv[0], (LLVMValueRef) argv[1],
                                     (LLVMValueRef) argv[2], argv[3],
                                     argv[4], argv[5], argv[6]);
}

/* llvalue -> llvalue array -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_gep(LLVMValueRef Pointer, value Indices,
                                     value Name, value B) {
  return LLVMBuildGEP(Builder_val(B), Pointer,
                      (LLVMValueRef *) Op_val(Indices), Wosize_val(Indices),
                      String_val(Name));
}

/* llvalue -> llvalue array -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_in_bounds_gep(LLVMValueRef Pointer,
                                               value Indices, value Name,
                                               value B) {
  return LLVMBuildInBoundsGEP(Builder_val(B), Pointer,
                              (LLVMValueRef *) Op_val(Indices),
                              Wosize_val(Indices), String_val(Name));
}

/* llvalue -> int -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_struct_gep(LLVMValueRef Pointer,
                                               value Index, value Name,
                                               value B) {
  return LLVMBuildStructGEP(Builder_val(B), Pointer,
                              Int_val(Index), String_val(Name));
}

/* string -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_global_string(value Str, value Name, value B) {
  return LLVMBuildGlobalString(Builder_val(B), String_val(Str),
                               String_val(Name));
}

/* string -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_global_stringptr(value Str, value Name,
                                                  value B) {
  return LLVMBuildGlobalStringPtr(Builder_val(B), String_val(Str),
                                  String_val(Name));
}

/*--... Casts ..............................................................--*/

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_trunc(LLVMValueRef X, LLVMTypeRef Ty,
                                       value Name, value B) {
  return LLVMBuildTrunc(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_zext(LLVMValueRef X, LLVMTypeRef Ty,
                                      value Name, value B) {
  return LLVMBuildZExt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sext(LLVMValueRef X, LLVMTypeRef Ty,
                                      value Name, value B) {
  return LLVMBuildSExt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fptoui(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildFPToUI(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fptosi(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildFPToSI(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_uitofp(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildUIToFP(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sitofp(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildSIToFP(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fptrunc(LLVMValueRef X, LLVMTypeRef Ty,
                                         value Name, value B) {
  return LLVMBuildFPTrunc(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fpext(LLVMValueRef X, LLVMTypeRef Ty,
                                       value Name, value B) {
  return LLVMBuildFPExt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_prttoint(LLVMValueRef X, LLVMTypeRef Ty,
                                          value Name, value B) {
  return LLVMBuildPtrToInt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_inttoptr(LLVMValueRef X, LLVMTypeRef Ty,
                                          value Name, value B) {
  return LLVMBuildIntToPtr(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_bitcast(LLVMValueRef X, LLVMTypeRef Ty,
                                         value Name, value B) {
  return LLVMBuildBitCast(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_zext_or_bitcast(LLVMValueRef X, LLVMTypeRef Ty,
                                                 value Name, value B) {
  return LLVMBuildZExtOrBitCast(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sext_or_bitcast(LLVMValueRef X, LLVMTypeRef Ty,
                                                 value Name, value B) {
  return LLVMBuildSExtOrBitCast(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_trunc_or_bitcast(LLVMValueRef X,
                                                  LLVMTypeRef Ty, value Name,
                                                  value B) {
  return LLVMBuildTruncOrBitCast(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_pointercast(LLVMValueRef X, LLVMTypeRef Ty,
                                             value Name, value B) {
  return LLVMBuildPointerCast(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_intcast(LLVMValueRef X, LLVMTypeRef Ty,
                                         value Name, value B) {
  return LLVMBuildIntCast(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fpcast(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildFPCast(Builder_val(B), X, Ty, String_val(Name));
}

/*--... Comparisons ........................................................--*/

/* Icmp.t -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_icmp(value Pred,
                                      LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildICmp(Builder_val(B), Int_val(Pred) + LLVMIntEQ, LHS, RHS,
                       String_val(Name));
}

/* Fcmp.t -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fcmp(value Pred,
                                      LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFCmp(Builder_val(B), Int_val(Pred), LHS, RHS,
                       String_val(Name));
}

/*--... Miscellaneous instructions .........................................--*/

/* (llvalue * llbasicblock) list -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_phi(value Incoming, value Name, value B) {
  value Hd, Tl;
  LLVMValueRef FirstValue, PhiNode;

  assert(Incoming != Val_int(0) && "Empty list passed to Llvm.build_phi!");

  Hd = Field(Incoming, 0);
  FirstValue = (LLVMValueRef) Field(Hd, 0);
  PhiNode = LLVMBuildPhi(Builder_val(B), LLVMTypeOf(FirstValue),
                         String_val(Name));

  for (Tl = Incoming; Tl != Val_int(0); Tl = Field(Tl, 1)) {
    value Hd = Field(Tl, 0);
    LLVMAddIncoming(PhiNode, (LLVMValueRef*) &Field(Hd, 0),
                    (LLVMBasicBlockRef*) &Field(Hd, 1), 1);
  }

  return PhiNode;
}

/* lltype -> string -> llbuilder -> value */
CAMLprim LLVMValueRef llvm_build_empty_phi(LLVMTypeRef Type, value Name, value B) {
  LLVMValueRef PhiNode;

  return LLVMBuildPhi(Builder_val(B), Type, String_val(Name));

  return PhiNode;
}

/* llvalue -> llvalue array -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_call(LLVMValueRef Fn, value Params,
                                      value Name, value B) {
  return LLVMBuildCall(Builder_val(B), Fn, (LLVMValueRef *) Op_val(Params),
                       Wosize_val(Params), String_val(Name));
}

/* llvalue -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_select(LLVMValueRef If,
                                        LLVMValueRef Then, LLVMValueRef Else,
                                        value Name, value B) {
  return LLVMBuildSelect(Builder_val(B), If, Then, Else, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_va_arg(LLVMValueRef List, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildVAArg(Builder_val(B), List, Ty, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_extractelement(LLVMValueRef Vec,
                                                LLVMValueRef Idx,
                                                value Name, value B) {
  return LLVMBuildExtractElement(Builder_val(B), Vec, Idx, String_val(Name));
}

/* llvalue -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_insertelement(LLVMValueRef Vec,
                                               LLVMValueRef Element,
                                               LLVMValueRef Idx,
                                               value Name, value B) {
  return LLVMBuildInsertElement(Builder_val(B), Vec, Element, Idx,
                                String_val(Name));
}

/* llvalue -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_shufflevector(LLVMValueRef V1, LLVMValueRef V2,
                                               LLVMValueRef Mask,
                                               value Name, value B) {
  return LLVMBuildShuffleVector(Builder_val(B), V1, V2, Mask, String_val(Name));
}

/* llvalue -> int -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_extractvalue(LLVMValueRef Aggregate,
                                              value Idx, value Name, value B) {
  return LLVMBuildExtractValue(Builder_val(B), Aggregate, Int_val(Idx),
                               String_val(Name));
}

/* llvalue -> llvalue -> int -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_insertvalue(LLVMValueRef Aggregate,
                                             LLVMValueRef Val, value Idx,
                                             value Name, value B) {
  return LLVMBuildInsertValue(Builder_val(B), Aggregate, Val, Int_val(Idx),
                              String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_is_null(LLVMValueRef Val, value Name,
                                         value B) {
  return LLVMBuildIsNull(Builder_val(B), Val, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_is_not_null(LLVMValueRef Val, value Name,
                                             value B) {
  return LLVMBuildIsNotNull(Builder_val(B), Val, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ptrdiff(LLVMValueRef LHS, LLVMValueRef RHS,
                                         value Name, value B) {
  return LLVMBuildPtrDiff(Builder_val(B), LHS, RHS, String_val(Name));
}

/*===-- Memory buffers ----------------------------------------------------===*/

/* string -> llmemorybuffer
   raises IoError msg on error */
CAMLprim value llvm_memorybuffer_of_file(value Path) {
  CAMLparam1(Path);
  char *Message;
  LLVMMemoryBufferRef MemBuf;

  if (LLVMCreateMemoryBufferWithContentsOfFile(String_val(Path),
                                               &MemBuf, &Message))
    llvm_raise(*caml_named_value("Llvm.IoError"), Message);

  CAMLreturn((value) MemBuf);
}

/* unit -> llmemorybuffer
   raises IoError msg on error */
CAMLprim LLVMMemoryBufferRef llvm_memorybuffer_of_stdin(value Unit) {
  char *Message;
  LLVMMemoryBufferRef MemBuf;

  if (LLVMCreateMemoryBufferWithSTDIN(&MemBuf, &Message))
    llvm_raise(*caml_named_value("Llvm.IoError"), Message);

  return MemBuf;
}

/* ?name:string -> string -> llmemorybuffer */
CAMLprim LLVMMemoryBufferRef llvm_memorybuffer_of_string(value Name, value String) {
  LLVMMemoryBufferRef MemBuf;
  const char *NameCStr;

  if(Name == Val_int(0))
    NameCStr = "";
  else
    NameCStr = String_val(Field(Name, 0));

  MemBuf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
                String_val(String), caml_string_length(String), NameCStr);

  return MemBuf;
}

/* llmemorybuffer -> string */
CAMLprim value llvm_memorybuffer_as_string(LLVMMemoryBufferRef MemBuf) {
  value String = caml_alloc_string(LLVMGetBufferSize(MemBuf));
  memcpy(String_val(String), LLVMGetBufferStart(MemBuf),
         LLVMGetBufferSize(MemBuf));

  return String;
}

/* llmemorybuffer -> unit */
CAMLprim value llvm_memorybuffer_dispose(LLVMMemoryBufferRef MemBuf) {
  LLVMDisposeMemoryBuffer(MemBuf);
  return Val_unit;
}

/*===-- Pass Managers -----------------------------------------------------===*/

/* unit -> [ `Module ] PassManager.t */
CAMLprim LLVMPassManagerRef llvm_passmanager_create(value Unit) {
  return LLVMCreatePassManager();
}

/* llmodule -> [ `Function ] PassManager.t -> bool */
CAMLprim value llvm_passmanager_run_module(LLVMModuleRef M,
                                           LLVMPassManagerRef PM) {
  return Val_bool(LLVMRunPassManager(PM, M));
}

/* [ `Function ] PassManager.t -> bool */
CAMLprim value llvm_passmanager_initialize(LLVMPassManagerRef FPM) {
  return Val_bool(LLVMInitializeFunctionPassManager(FPM));
}

/* llvalue -> [ `Function ] PassManager.t -> bool */
CAMLprim value llvm_passmanager_run_function(LLVMValueRef F,
                                             LLVMPassManagerRef FPM) {
  return Val_bool(LLVMRunFunctionPassManager(FPM, F));
}

/* [ `Function ] PassManager.t -> bool */
CAMLprim value llvm_passmanager_finalize(LLVMPassManagerRef FPM) {
  return Val_bool(LLVMFinalizeFunctionPassManager(FPM));
}

/* PassManager.any PassManager.t -> unit */
CAMLprim value llvm_passmanager_dispose(LLVMPassManagerRef PM) {
  LLVMDisposePassManager(PM);
  return Val_unit;
}
