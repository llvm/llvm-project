(*===-- llvm/llvm.ml - LLVM Ocaml Interface --------------------------------===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


type llmodule
type lltype
type lltypehandle
type llvalue
type llbasicblock
type llbuilder
type llmoduleprovider
type llmemorybuffer

module TypeKind = struct
  type t =
  | Void
  | Float
  | Double
  | X86fp80
  | Fp128
  | Ppc_fp128
  | Label
  | Integer
  | Function
  | Struct
  | Array
  | Pointer
  | Opaque
  | Vector
end

module Linkage = struct
  type t =
  | External
  | Link_once
  | Weak
  | Appending
  | Internal
  | Dllimport
  | Dllexport
  | External_weak
  | Ghost
end

module Visibility = struct
  type t =
  | Default
  | Hidden
  | Protected
end

module CallConv = struct
  let c = 0
  let fast = 8
  let cold = 9
  let x86_stdcall = 64
  let x86_fastcall = 65
end

module Icmp = struct
  type t =
  | Eq
  | Ne
  | Ugt
  | Uge
  | Ult
  | Ule
  | Sgt
  | Sge
  | Slt
  | Sle
end

module Fcmp = struct
  type t =
  | False
  | Oeq
  | Ogt
  | Oge
  | Olt
  | Ole
  | One
  | Ord
  | Uno
  | Ueq
  | Ugt
  | Uge
  | Ult
  | Ule
  | Une
  | True
end

exception IoError of string

external register_exns : exn -> unit = "llvm_register_core_exns"
let _ = register_exns (IoError "")

type ('a, 'b) llpos =
| At_end of 'a
| Before of 'b

type ('a, 'b) llrev_pos =
| At_start of 'a
| After of 'b


(*===-- Modules -----------------------------------------------------------===*)

external create_module : string -> llmodule = "llvm_create_module"
external dispose_module : llmodule -> unit = "llvm_dispose_module"
external target_triple: llmodule -> string
                      = "llvm_target_triple"
external set_target_triple: string -> llmodule -> unit
                          = "llvm_set_target_triple"
external data_layout: llmodule -> string
                    = "llvm_data_layout"
external set_data_layout: string -> llmodule -> unit
                        = "llvm_set_data_layout"
external define_type_name : string -> lltype -> llmodule -> bool
                          = "llvm_add_type_name"
external delete_type_name : string -> llmodule -> unit
                          = "llvm_delete_type_name"
external dump_module : llmodule -> unit = "llvm_dump_module"

(*===-- Types -------------------------------------------------------------===*)

external classify_type : lltype -> TypeKind.t = "llvm_classify_type"

(*--... Operations on integer types ........................................--*)
external _i1_type : unit -> lltype = "llvm_i1_type"
external _i8_type : unit -> lltype = "llvm_i8_type"
external _i16_type : unit -> lltype = "llvm_i16_type"
external _i32_type : unit -> lltype = "llvm_i32_type"
external _i64_type : unit -> lltype = "llvm_i64_type"

let i1_type = _i1_type ()
let i8_type = _i8_type ()
let i16_type = _i16_type ()
let i32_type = _i32_type ()
let i64_type = _i64_type ()

external integer_type : int -> lltype = "llvm_integer_type"
external integer_bitwidth : lltype -> int = "llvm_integer_bitwidth"

(*--... Operations on real types ...........................................--*)
external _float_type : unit -> lltype = "llvm_float_type"
external _double_type : unit -> lltype = "llvm_double_type"
external _x86fp80_type : unit -> lltype = "llvm_x86fp80_type"
external _fp128_type : unit -> lltype = "llvm_fp128_type"
external _ppc_fp128_type : unit -> lltype = "llvm_ppc_fp128_type"

let float_type = _float_type ()
let double_type = _double_type ()
let x86fp80_type = _x86fp80_type ()
let fp128_type = _fp128_type ()
let ppc_fp128_type = _ppc_fp128_type ()

(*--... Operations on function types .......................................--*)
external function_type : lltype -> lltype array -> lltype = "llvm_function_type"
external var_arg_function_type : lltype -> lltype array -> lltype
                               = "llvm_var_arg_function_type"
external is_var_arg : lltype -> bool = "llvm_is_var_arg"
external return_type : lltype -> lltype = "LLVMGetReturnType"
external param_types : lltype -> lltype array = "llvm_param_types"

(*--... Operations on struct types .........................................--*)
external struct_type : lltype array -> lltype = "llvm_struct_type"
external packed_struct_type : lltype array -> lltype = "llvm_packed_struct_type"
external element_types : lltype -> lltype array = "llvm_element_types"
external is_packed : lltype -> bool = "llvm_is_packed"

(*--... Operations on pointer, vector, and array types .....................--*)
external array_type : lltype -> int -> lltype = "llvm_array_type"
external pointer_type : lltype -> lltype = "llvm_pointer_type"
external qualified_pointer_type : lltype -> int -> lltype
                                = "llvm_qualified_pointer_type"
external vector_type : lltype -> int -> lltype = "llvm_vector_type"

external element_type : lltype -> lltype = "LLVMGetElementType"
external array_length : lltype -> int = "llvm_array_length"
external address_space : lltype -> int = "llvm_address_space"
external vector_size : lltype -> int = "llvm_vector_size"

(*--... Operations on other types ..........................................--*)
external opaque_type : unit -> lltype = "llvm_opaque_type"
external _void_type : unit -> lltype = "llvm_void_type"
external _label_type : unit -> lltype = "llvm_label_type"

let void_type = _void_type ()
let label_type = _label_type ()

(*--... Operations on type handles .........................................--*)
external handle_to_type : lltype -> lltypehandle = "llvm_handle_to_type"
external type_of_handle : lltypehandle -> lltype = "llvm_type_of_handle"
external refine_type : lltype -> lltype -> unit = "llvm_refine_type"


(*===-- Values ------------------------------------------------------------===*)

external type_of : llvalue -> lltype = "llvm_type_of"
external value_name : llvalue -> string = "llvm_value_name"
external set_value_name : string -> llvalue -> unit = "llvm_set_value_name"
external dump_value : llvalue -> unit = "llvm_dump_value"

(*--... Operations on constants of (mostly) any type .......................--*)
external is_constant : llvalue -> bool = "llvm_is_constant"
external const_null : lltype -> llvalue = "LLVMConstNull"
external const_all_ones : (*int|vec*)lltype -> llvalue = "LLVMConstAllOnes"
external undef : lltype -> llvalue = "LLVMGetUndef"
external is_null : llvalue -> bool = "llvm_is_null"
external is_undef : llvalue -> bool = "llvm_is_undef"

(*--... Operations on scalar constants .....................................--*)
external const_int : lltype -> int -> llvalue = "llvm_const_int"
external const_of_int64 : lltype -> Int64.t -> bool -> llvalue
                        = "llvm_const_of_int64"
external const_float : lltype -> float -> llvalue = "llvm_const_float"

(*--... Operations on composite constants ..................................--*)
external const_string : string -> llvalue = "llvm_const_string"
external const_stringz : string -> llvalue = "llvm_const_stringz"
external const_array : lltype -> llvalue array -> llvalue = "llvm_const_array"
external const_struct : llvalue array -> llvalue = "llvm_const_struct"
external const_packed_struct : llvalue array -> llvalue
                             = "llvm_const_packed_struct"
external const_vector : llvalue array -> llvalue = "llvm_const_vector"

(*--... Constant expressions ...............................................--*)
external size_of : lltype -> llvalue = "LLVMSizeOf"
external const_neg : llvalue -> llvalue = "LLVMConstNeg"
external const_not : llvalue -> llvalue = "LLVMConstNot"
external const_add : llvalue -> llvalue -> llvalue = "LLVMConstAdd"
external const_sub : llvalue -> llvalue -> llvalue = "LLVMConstSub"
external const_mul : llvalue -> llvalue -> llvalue = "LLVMConstMul"
external const_udiv : llvalue -> llvalue -> llvalue = "LLVMConstUDiv"
external const_sdiv : llvalue -> llvalue -> llvalue = "LLVMConstSDiv"
external const_fdiv : llvalue -> llvalue -> llvalue = "LLVMConstFDiv"
external const_urem : llvalue -> llvalue -> llvalue = "LLVMConstURem"
external const_srem : llvalue -> llvalue -> llvalue = "LLVMConstSRem"
external const_frem : llvalue -> llvalue -> llvalue = "LLVMConstFRem"
external const_and : llvalue -> llvalue -> llvalue = "LLVMConstAnd"
external const_or : llvalue -> llvalue -> llvalue = "LLVMConstOr"
external const_xor : llvalue -> llvalue -> llvalue = "LLVMConstXor"
external const_icmp : Icmp.t -> llvalue -> llvalue -> llvalue
                    = "llvm_const_icmp"
external const_fcmp : Fcmp.t -> llvalue -> llvalue -> llvalue
                    = "llvm_const_fcmp"
external const_shl : llvalue -> llvalue -> llvalue = "LLVMConstShl"
external const_lshr : llvalue -> llvalue -> llvalue = "LLVMConstLShr"
external const_ashr : llvalue -> llvalue -> llvalue = "LLVMConstAShr"
external const_gep : llvalue -> llvalue array -> llvalue = "llvm_const_gep"
external const_trunc : llvalue -> lltype -> llvalue = "LLVMConstTrunc"
external const_sext : llvalue -> lltype -> llvalue = "LLVMConstSExt"
external const_zext : llvalue -> lltype -> llvalue = "LLVMConstZExt"
external const_fptrunc : llvalue -> lltype -> llvalue = "LLVMConstFPTrunc"
external const_fpext : llvalue -> lltype -> llvalue = "LLVMConstFPExt"
external const_uitofp : llvalue -> lltype -> llvalue = "LLVMConstUIToFP"
external const_sitofp : llvalue -> lltype -> llvalue = "LLVMConstSIToFP"
external const_fptoui : llvalue -> lltype -> llvalue = "LLVMConstFPToUI"
external const_fptosi : llvalue -> lltype -> llvalue = "LLVMConstFPToSI"
external const_ptrtoint : llvalue -> lltype -> llvalue = "LLVMConstPtrToInt"
external const_inttoptr : llvalue -> lltype -> llvalue = "LLVMConstIntToPtr"
external const_bitcast : llvalue -> lltype -> llvalue = "LLVMConstBitCast"
external const_select : llvalue -> llvalue -> llvalue -> llvalue
                      = "LLVMConstSelect"
external const_extractelement : llvalue -> llvalue -> llvalue
                              = "LLVMConstExtractElement"
external const_insertelement : llvalue -> llvalue -> llvalue -> llvalue
                             = "LLVMConstInsertElement"
external const_shufflevector : llvalue -> llvalue -> llvalue -> llvalue
                             = "LLVMConstShuffleVector"

(*--... Operations on global variables, functions, and aliases (globals) ...--*)
external global_parent : llvalue -> llmodule = "LLVMGetGlobalParent"
external is_declaration : llvalue -> bool = "llvm_is_declaration"
external linkage : llvalue -> Linkage.t = "llvm_linkage"
external set_linkage : Linkage.t -> llvalue -> unit = "llvm_set_linkage"
external section : llvalue -> string = "llvm_section"
external set_section : string -> llvalue -> unit = "llvm_set_section"
external visibility : llvalue -> Visibility.t = "llvm_visibility"
external set_visibility : Visibility.t -> llvalue -> unit = "llvm_set_visibility"
external alignment : llvalue -> int = "llvm_alignment"
external set_alignment : int -> llvalue -> unit = "llvm_set_alignment"
external is_global_constant : llvalue -> bool = "llvm_is_global_constant"
external set_global_constant : bool -> llvalue -> unit
                             = "llvm_set_global_constant"

(*--... Operations on global variables .....................................--*)
external declare_global : lltype -> string -> llmodule -> llvalue
                        = "llvm_declare_global"
external define_global : string -> llvalue -> llmodule -> llvalue
                       = "llvm_define_global"
external lookup_global : string -> llmodule -> llvalue option
                       = "llvm_lookup_global"
external delete_global : llvalue -> unit = "llvm_delete_global"
external global_initializer : llvalue -> llvalue = "LLVMGetInitializer"
external set_initializer : llvalue -> llvalue -> unit = "llvm_set_initializer"
external remove_initializer : llvalue -> unit = "llvm_remove_initializer"
external is_thread_local : llvalue -> bool = "llvm_is_thread_local"
external set_thread_local : bool -> llvalue -> unit = "llvm_set_thread_local"
external global_begin : llmodule -> (llmodule, llvalue) llpos
                      = "llvm_global_begin"
external global_succ : llvalue -> (llmodule, llvalue) llpos
                     = "llvm_global_succ"
external global_end : llmodule -> (llmodule, llvalue) llrev_pos
                    = "llvm_global_end"
external global_pred : llvalue -> (llmodule, llvalue) llrev_pos
                     = "llvm_global_pred"

let rec iter_global_range f i e =
  if i = e then () else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid global variable range.")
  | Before bb ->
      f bb;
      iter_global_range f (global_succ bb) e

let iter_globals f m =
  iter_global_range f (global_begin m) (At_end m)

let rec fold_left_global_range f init i e =
  if i = e then init else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid global variable range.")
  | Before bb -> fold_left_global_range f (f init bb) (global_succ bb) e

let fold_left_globals f init m =
  fold_left_global_range f init (global_begin m) (At_end m)

let rec rev_iter_global_range f i e =
  if i = e then () else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid global variable range.")
  | After bb ->
      f bb;
      rev_iter_global_range f (global_pred bb) e

let rev_iter_globals f m =
  rev_iter_global_range f (global_end m) (At_start m)

let rec fold_right_global_range f i e init =
  if i = e then init else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid global variable range.")
  | After bb -> fold_right_global_range f (global_pred bb) e (f bb init)

let fold_right_globals f m init =
  fold_right_global_range f (global_end m) (At_start m) init

(*--... Operations on functions ............................................--*)
external declare_function : string -> lltype -> llmodule -> llvalue
                          = "llvm_declare_function"
external define_function : string -> lltype -> llmodule -> llvalue
                         = "llvm_define_function"
external lookup_function : string -> llmodule -> llvalue option
                         = "llvm_lookup_function"
external delete_function : llvalue -> unit = "llvm_delete_function"
external is_intrinsic : llvalue -> bool = "llvm_is_intrinsic"
external function_call_conv : llvalue -> int = "llvm_function_call_conv"
external set_function_call_conv : int -> llvalue -> unit
                                = "llvm_set_function_call_conv"
external gc : llvalue -> string option = "llvm_gc"
external set_gc : string option -> llvalue -> unit = "llvm_set_gc"
external function_begin : llmodule -> (llmodule, llvalue) llpos
                        = "llvm_function_begin"
external function_succ : llvalue -> (llmodule, llvalue) llpos
                       = "llvm_function_succ"
external function_end : llmodule -> (llmodule, llvalue) llrev_pos
                      = "llvm_function_end"
external function_pred : llvalue -> (llmodule, llvalue) llrev_pos
                       = "llvm_function_pred"

let rec iter_function_range f i e =
  if i = e then () else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid function range.")
  | Before fn ->
      f fn;
      iter_function_range f (function_succ fn) e

let iter_functions f m =
  iter_function_range f (function_begin m) (At_end m)

let rec fold_left_function_range f init i e =
  if i = e then init else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid function range.")
  | Before fn -> fold_left_function_range f (f init fn) (function_succ fn) e

let fold_left_functions f init m =
  fold_left_function_range f init (function_begin m) (At_end m)

let rec rev_iter_function_range f i e =
  if i = e then () else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid function range.")
  | After fn ->
      f fn;
      rev_iter_function_range f (function_pred fn) e

let rev_iter_functions f m =
  rev_iter_function_range f (function_end m) (At_start m)

let rec fold_right_function_range f i e init =
  if i = e then init else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid function range.")
  | After fn -> fold_right_function_range f (function_pred fn) e (f fn init)

let fold_right_functions f m init =
  fold_right_function_range f (function_end m) (At_start m) init

(* TODO: param attrs *)

(*--... Operations on params ...............................................--*)
external params : llvalue -> llvalue array = "llvm_params"
external param : llvalue -> int -> llvalue = "llvm_param"
external param_parent : llvalue -> llvalue = "LLVMGetParamParent"
external param_begin : llvalue -> (llvalue, llvalue) llpos = "llvm_param_begin"
external param_succ : llvalue -> (llvalue, llvalue) llpos = "llvm_param_succ"
external param_end : llvalue -> (llvalue, llvalue) llrev_pos = "llvm_param_end"
external param_pred : llvalue -> (llvalue, llvalue) llrev_pos ="llvm_param_pred"

let rec iter_param_range f i e =
  if i = e then () else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid parameter range.")
  | Before p ->
      f p;
      iter_param_range f (param_succ p) e

let iter_params f fn =
  iter_param_range f (param_begin fn) (At_end fn)

let rec fold_left_param_range f init i e =
  if i = e then init else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid parameter range.")
  | Before p -> fold_left_param_range f (f init p) (param_succ p) e

let fold_left_params f init fn =
  fold_left_param_range f init (param_begin fn) (At_end fn)

let rec rev_iter_param_range f i e =
  if i = e then () else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid parameter range.")
  | After p ->
      f p;
      rev_iter_param_range f (param_pred p) e

let rev_iter_params f fn =
  rev_iter_param_range f (param_end fn) (At_start fn)

let rec fold_right_param_range f init i e =
  if i = e then init else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid parameter range.")
  | After p -> fold_right_param_range f (f p init) (param_pred p) e

let fold_right_params f fn init =
  fold_right_param_range f init (param_end fn) (At_start fn)

(*--... Operations on basic blocks .........................................--*)
external value_of_block : llbasicblock -> llvalue = "LLVMBasicBlockAsValue"
external value_is_block : llvalue -> bool = "llvm_value_is_block"
external block_of_value : llvalue -> llbasicblock = "LLVMValueAsBasicBlock"
external block_parent : llbasicblock -> llvalue = "LLVMGetBasicBlockParent"
external basic_blocks : llvalue -> llbasicblock array = "llvm_basic_blocks"
external entry_block : llvalue -> llbasicblock = "LLVMGetEntryBasicBlock"
external delete_block : llbasicblock -> unit = "llvm_delete_block"
external append_block : string -> llvalue -> llbasicblock = "llvm_append_block"
external insert_block : string -> llbasicblock -> llbasicblock
                      = "llvm_insert_block"
external block_begin : llvalue -> (llvalue, llbasicblock) llpos
                     = "llvm_block_begin"
external block_succ : llbasicblock -> (llvalue, llbasicblock) llpos
                    = "llvm_block_succ"
external block_end : llvalue -> (llvalue, llbasicblock) llrev_pos
                   = "llvm_block_end"
external block_pred : llbasicblock -> (llvalue, llbasicblock) llrev_pos
                    = "llvm_block_pred"

let rec iter_block_range f i e =
  if i = e then () else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid block range.")
  | Before bb ->
      f bb;
      iter_block_range f (block_succ bb) e

let iter_blocks f fn =
  iter_block_range f (block_begin fn) (At_end fn)

let rec fold_left_block_range f init i e =
  if i = e then init else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid block range.")
  | Before bb -> fold_left_block_range f (f init bb) (block_succ bb) e

let fold_left_blocks f init fn =
  fold_left_block_range f init (block_begin fn) (At_end fn)

let rec rev_iter_block_range f i e =
  if i = e then () else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid block range.")
  | After bb ->
      f bb;
      rev_iter_block_range f (block_pred bb) e

let rev_iter_blocks f fn =
  rev_iter_block_range f (block_end fn) (At_start fn)

let rec fold_right_block_range f init i e =
  if i = e then init else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid block range.")
  | After bb -> fold_right_block_range f (f bb init) (block_pred bb) e

let fold_right_blocks f fn init =
  fold_right_block_range f init (block_end fn) (At_start fn)

(*--... Operations on instructions .........................................--*)
external instr_parent : llvalue -> llbasicblock = "LLVMGetInstructionParent"
external instr_begin : llbasicblock -> (llbasicblock, llvalue) llpos
                     = "llvm_instr_begin"
external instr_succ : llvalue -> (llbasicblock, llvalue) llpos
                     = "llvm_instr_succ"
external instr_end : llbasicblock -> (llbasicblock, llvalue) llrev_pos
                     = "llvm_instr_end"
external instr_pred : llvalue -> (llbasicblock, llvalue) llrev_pos
                     = "llvm_instr_pred"

let rec iter_instrs_range f i e =
  if i = e then () else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid instruction range.")
  | Before i ->
      f i;
      iter_instrs_range f (instr_succ i) e

let iter_instrs f bb =
  iter_instrs_range f (instr_begin bb) (At_end bb)

let rec fold_left_instrs_range f init i e =
  if i = e then init else
  match i with
  | At_end _ -> raise (Invalid_argument "Invalid instruction range.")
  | Before i -> fold_left_instrs_range f (f init i) (instr_succ i) e

let fold_left_instrs f init bb =
  fold_left_instrs_range f init (instr_begin bb) (At_end bb)

let rec rev_iter_instrs_range f i e =
  if i = e then () else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid instruction range.")
  | After i ->
      f i;
      rev_iter_instrs_range f (instr_pred i) e

let rev_iter_instrs f bb =
  rev_iter_instrs_range f (instr_end bb) (At_start bb)

let rec fold_right_instr_range f i e init =
  if i = e then init else
  match i with
  | At_start _ -> raise (Invalid_argument "Invalid instruction range.")
  | After i -> fold_right_instr_range f (instr_pred i) e (f i init)

let fold_right_instrs f bb init =
  fold_right_instr_range f (instr_end bb) (At_start bb) init


(*--... Operations on call sites ...........................................--*)
external instruction_call_conv: llvalue -> int
                              = "llvm_instruction_call_conv"
external set_instruction_call_conv: int -> llvalue -> unit
                                  = "llvm_set_instruction_call_conv"

(*--... Operations on call instructions (only) .............................--*)
external is_tail_call : llvalue -> bool = "llvm_is_tail_call"
external set_tail_call : bool -> llvalue -> unit = "llvm_set_tail_call"

(*--... Operations on phi nodes ............................................--*)
external add_incoming : (llvalue * llbasicblock) -> llvalue -> unit
                      = "llvm_add_incoming"
external incoming : llvalue -> (llvalue * llbasicblock) list = "llvm_incoming"


(*===-- Instruction builders ----------------------------------------------===*)
external builder : unit -> llbuilder = "llvm_builder"
external position_builder : (llbasicblock, llvalue) llpos -> llbuilder -> unit
                          = "llvm_position_builder"
external insertion_block : llbuilder -> llbasicblock = "llvm_insertion_block"

let builder_at ip =
  let b = builder () in
  position_builder ip b;
  b

let builder_before i = builder_at (Before i)
let builder_at_end bb = builder_at (At_end bb)

let position_before i = position_builder (Before i)
let position_at_end bb = position_builder (At_end bb)


(*--... Terminators ........................................................--*)
external build_ret_void : llbuilder -> llvalue = "llvm_build_ret_void"
external build_ret : llvalue -> llbuilder -> llvalue = "llvm_build_ret"
external build_br : llbasicblock -> llbuilder -> llvalue = "llvm_build_br"
external build_cond_br : llvalue -> llbasicblock -> llbasicblock -> llbuilder ->
                         llvalue = "llvm_build_cond_br"
external build_switch : llvalue -> llbasicblock -> int -> llbuilder -> llvalue
                      = "llvm_build_switch"
external add_case : llvalue -> llvalue -> llbasicblock -> unit
                  = "llvm_add_case"
external build_invoke : llvalue -> llvalue array -> llbasicblock ->
                        llbasicblock -> string -> llbuilder -> llvalue
                      = "llvm_build_invoke_bc" "llvm_build_invoke_nat"
external build_unwind : llbuilder -> llvalue = "llvm_build_unwind"
external build_unreachable : llbuilder -> llvalue = "llvm_build_unreachable"

(*--... Arithmetic .........................................................--*)
external build_add : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_add"
external build_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_sub"
external build_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_mul"
external build_udiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_udiv"
external build_sdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_sdiv"
external build_fdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fdiv"
external build_urem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_urem"
external build_srem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_srem"
external build_frem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_frem"
external build_shl : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_shl"
external build_lshr : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_lshr"
external build_ashr : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_ashr"
external build_and : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_and"
external build_or : llvalue -> llvalue -> string -> llbuilder -> llvalue
                  = "llvm_build_or"
external build_xor : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_xor"
external build_neg : llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_neg"
external build_not : llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_not"

(*--... Memory .............................................................--*)
external build_malloc : lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_malloc"
external build_array_malloc : lltype -> llvalue -> string -> llbuilder ->
                              llvalue = "llvm_build_array_malloc"
external build_alloca : lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_alloca"
external build_array_alloca : lltype -> llvalue -> string -> llbuilder ->
                              llvalue = "llvm_build_array_alloca"
external build_free : llvalue -> llbuilder -> llvalue = "llvm_build_free"
external build_load : llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_load"
external build_store : llvalue -> llvalue -> llbuilder -> llvalue
                     = "llvm_build_store"
external build_gep : llvalue -> llvalue array -> string -> llbuilder -> llvalue
                   = "llvm_build_gep"

(*--... Casts ..............................................................--*)
external build_trunc : llvalue -> lltype -> string -> llbuilder -> llvalue
                     = "llvm_build_trunc"
external build_zext : llvalue -> lltype -> string -> llbuilder -> llvalue
                    = "llvm_build_zext"
external build_sext : llvalue -> lltype -> string -> llbuilder -> llvalue
                    = "llvm_build_sext"
external build_fptoui : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fptoui"
external build_fptosi : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fptosi"
external build_uitofp : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_uitofp"
external build_sitofp : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_sitofp"
external build_fptrunc : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_fptrunc"
external build_fpext : llvalue -> lltype -> string -> llbuilder -> llvalue
                     = "llvm_build_fpext"
external build_ptrtoint : llvalue -> lltype -> string -> llbuilder -> llvalue
                        = "llvm_build_prttoint"
external build_inttoptr : llvalue -> lltype -> string -> llbuilder -> llvalue
                        = "llvm_build_inttoptr"
external build_bitcast : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_bitcast"

(*--... Comparisons ........................................................--*)
external build_icmp : Icmp.t -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue = "llvm_build_icmp"
external build_fcmp : Fcmp.t -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue = "llvm_build_fcmp"

(*--... Miscellaneous instructions .........................................--*)
external build_phi : (llvalue * llbasicblock) list -> string -> llbuilder ->
                     llvalue = "llvm_build_phi"
external build_call : llvalue -> llvalue array -> string -> llbuilder -> llvalue
                    = "llvm_build_call"
external build_select : llvalue -> llvalue -> llvalue -> string -> llbuilder ->
                        llvalue = "llvm_build_select"
external build_va_arg : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_va_arg"
external build_extractelement : llvalue -> llvalue -> string -> llbuilder ->
                                llvalue = "llvm_build_extractelement"
external build_insertelement : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue = "llvm_build_insertelement"
external build_shufflevector : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue = "llvm_build_shufflevector"


(*===-- Module providers --------------------------------------------------===*)

module ModuleProvider = struct
  external create : llmodule -> llmoduleprovider
                  = "LLVMCreateModuleProviderForExistingModule"
  external dispose : llmoduleprovider -> unit = "llvm_dispose_module_provider"
end
  

(*===-- Memory buffers ----------------------------------------------------===*)

module MemoryBuffer = struct
  external of_file : string -> llmemorybuffer = "llvm_memorybuffer_of_file"
  external of_stdin : unit -> llmemorybuffer = "llvm_memorybuffer_of_stdin"
  external dispose : llmemorybuffer -> unit = "llvm_memorybuffer_dispose"
end


(*===-- Pass Manager ------------------------------------------------------===*)

module PassManager = struct
  type 'a t
  type any = [ `Module | `Function ]
  external create : unit -> [ `Module ] t = "llvm_passmanager_create"
  external create_function : llmoduleprovider -> [ `Function ] t
                           = "LLVMCreateFunctionPassManager"
  external run_module : llmodule -> [ `Module ] t -> bool
                      = "llvm_passmanager_run_module"
  external initialize : [ `Function ] t -> bool = "llvm_passmanager_initialize"
  external run_function : llvalue -> [ `Function ] t -> bool
                        = "llvm_passmanager_run_function"
  external finalize : [ `Function ] t -> bool = "llvm_passmanager_finalize"
  external dispose : [< any ] t -> unit = "llvm_passmanager_dispose"
end


(*===-- Non-Externs -------------------------------------------------------===*)
(* These functions are built using the externals, so must be declared late.   *)

let concat2 sep arr =
  let s = ref "" in
  if 0 < Array.length arr then begin
    s := !s ^ arr.(0);
    for i = 1 to (Array.length arr) - 1 do
      s := !s ^ sep ^ arr.(i)
    done
  end;
  !s

let rec string_of_lltype ty =
  (* FIXME: stop infinite recursion! :) *)
  match classify_type ty with
    TypeKind.Integer -> "i" ^ string_of_int (integer_bitwidth ty)
  | TypeKind.Pointer -> (string_of_lltype (element_type ty)) ^ "*"
  | TypeKind.Struct ->
      let s = "{ " ^ (concat2 ", " (
                Array.map string_of_lltype (element_types ty)
              )) ^ " }" in
      if is_packed ty
        then "<" ^ s ^ ">"
        else s
  | TypeKind.Array -> "["   ^ (string_of_int (array_length ty)) ^
                      " x " ^ (string_of_lltype (element_type ty)) ^ "]"
  | TypeKind.Vector -> "<"   ^ (string_of_int (vector_size ty)) ^
                       " x " ^ (string_of_lltype (element_type ty)) ^ ">"
  | TypeKind.Opaque -> "opaque"
  | TypeKind.Function -> string_of_lltype (return_type ty) ^
                         " (" ^ (concat2 ", " (
                           Array.map string_of_lltype (param_types ty)
                         )) ^ ")"
  | TypeKind.Label -> "label"
  | TypeKind.Ppc_fp128 -> "ppc_fp128"
  | TypeKind.Fp128 -> "fp128"
  | TypeKind.X86fp80 -> "x86_fp80"
  | TypeKind.Double -> "double"
  | TypeKind.Float -> "float"
  | TypeKind.Void -> "void"
