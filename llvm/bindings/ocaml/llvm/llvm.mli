(*===-- llvm/llvm.mli - LLVM Ocaml Interface -------------------------------===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Core API.

    This interface provides an ocaml API for the LLVM intermediate
    representation, the classes in the VMCore library. *)


(** {6 Abstract types}

    These abstract types correlate directly to the LLVM VMCore classes. *)

(** The top-level container for all LLVM global data. See the
    [llvm::LLVMContext] class. *)
type llcontext

(** The top-level container for all other LLVM Intermediate Representation (IR)
    objects. See the [llvm::Module] class. *)
type llmodule

(** Each value in the LLVM IR has a type, an instance of [lltype]. See the
    [llvm::Type] class. *)
type lltype

(** Any value in the LLVM IR. Functions, instructions, global variables,
    constants, and much more are all [llvalues]. See the [llvm::Value] class.
    This type covers a wide range of subclasses. *)
type llvalue

(** Used to store users and usees of values. See the [llvm::Use] class. *)
type lluse

(** A basic block in LLVM IR. See the [llvm::BasicBlock] class. *)
type llbasicblock

(** Used to generate instructions in the LLVM IR. See the [llvm::LLVMBuilder]
    class. *)
type llbuilder

(** Used to efficiently handle large buffers of read-only binary data.
    See the [llvm::MemoryBuffer] class. *)
type llmemorybuffer

(** The kind of an [lltype], the result of [classify_type ty]. See the
    [llvm::Type::TypeID] enumeration. *)
module TypeKind : sig
  type t =
    Void
  | Half
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
  | Vector
  | Metadata
end

(** The linkage of a global value, accessed with {!linkage} and
    {!set_linkage}. See [llvm::GlobalValue::LinkageTypes]. *)
module Linkage : sig
  type t =
    External
  | Available_externally
  | Link_once
  | Link_once_odr
  | Weak
  | Weak_odr
  | Appending
  | Internal
  | Private
  | Dllimport
  | Dllexport
  | External_weak
  | Ghost
  | Common
  | Linker_private
end

(** The linker visibility of a global value, accessed with {!visibility} and
    {!set_visibility}. See [llvm::GlobalValue::VisibilityTypes]. *)
module Visibility : sig
  type t =
    Default
  | Hidden
  | Protected
end

(** The following calling convention values may be accessed with
    {!function_call_conv} and {!set_function_call_conv}. Calling
    conventions are open-ended. *)
module CallConv : sig
  val c : int             (** [c] is the C calling convention. *)
  val fast : int          (** [fast] is the calling convention to allow LLVM
                              maximum optimization opportunities. Use only with
                              internal linkage. *)
  val cold : int          (** [cold] is the calling convention for
                              callee-save. *)
  val x86_stdcall : int   (** [x86_stdcall] is the familiar stdcall calling
                              convention from C. *)
  val x86_fastcall : int  (** [x86_fastcall] is the familiar fastcall calling
                              convention from C. *)
end

module Attribute : sig
  type t =
  | Zext
  | Sext
  | Noreturn
  | Inreg
  | Structret
  | Nounwind
  | Noalias
  | Byval
  | Nest
  | Readnone
  | Readonly
  | Noinline
  | Alwaysinline
  | Optsize
  | Ssp
  | Sspreq
  | Alignment of int
  | Nocapture
  | Noredzone
  | Noimplicitfloat
  | Naked
  | Inlinehint
  | Stackalignment of int
  | ReturnsTwice
  | UWTable
  | NonLazyBind
end

(** The predicate for an integer comparison ([icmp]) instruction.
    See the [llvm::ICmpInst::Predicate] enumeration. *)
module Icmp : sig
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

(** The predicate for a floating-point comparison ([fcmp]) instruction.
    See the [llvm::FCmpInst::Predicate] enumeration. *)
module Fcmp : sig
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

(** The opcodes for LLVM instructions and constant expressions. *)
module Opcode : sig
  type t =
  | Invalid (* not an instruction *)
  (* Terminator Instructions *)
  | Ret
  | Br
  | Switch
  | IndirectBr
  | Invoke
  | Invalid2
  | Unreachable
  (* Standard Binary Operators *)
  | Add
  | FAdd
  | Sub
  | FSub
  | Mul
  | FMul
  | UDiv
  | SDiv
  | FDiv
  | URem
  | SRem
  | FRem
  (* Logical Operators *)
  | Shl
  | LShr
  | AShr
  | And
  | Or
  | Xor
  (* Memory Operators *)
  | Alloca
  | Load
  | Store
  | GetElementPtr
  (* Cast Operators *)
  | Trunc
  | ZExt
  | SExt
  | FPToUI
  | FPToSI
  | UIToFP
  | SIToFP
  | FPTrunc
  | FPExt
  | PtrToInt
  | IntToPtr
  | BitCast
  (* Other Operators *)
  | ICmp
  | FCmp
  | PHI
  | Call
  | Select
  | UserOp1
  | UserOp2
  | VAArg
  | ExtractElement
  | InsertElement
  | ShuffleVector
  | ExtractValue
  | InsertValue
  | Fence
  | AtomicCmpXchg
  | AtomicRMW
  | Resume
  | LandingPad
  | Unwind
end

(** The kind of an [llvalue], the result of [classify_value v].
 * See the various [LLVMIsA*] functions. *)
module ValueKind : sig
  type t =
  | NullValue
  | Argument
  | BasicBlock
  | InlineAsm
  | MDNode
  | MDString
  | BlockAddress
  | ConstantAggregateZero
  | ConstantArray
  | ConstantExpr
  | ConstantFP
  | ConstantInt
  | ConstantPointerNull
  | ConstantStruct
  | ConstantVector
  | Function
  | GlobalAlias
  | GlobalVariable
  | UndefValue
  | Instruction of Opcode.t
end

(** {6 Iteration} *)

(** [Before b] and [At_end a] specify positions from the start of the ['b] list
    of [a]. [llpos] is used to specify positions in and for forward iteration
    through the various value lists maintained by the LLVM IR. *)
type ('a, 'b) llpos =
| At_end of 'a
| Before of 'b

(** [After b] and [At_start a] specify positions from the end of the ['b] list
    of [a]. [llrev_pos] is used for reverse iteration through the various value
    lists maintained by the LLVM IR. *)
type ('a, 'b) llrev_pos =
| At_start of 'a
| After of 'b


(** {6 Exceptions} *)

exception IoError of string


(** {6 Contexts} *)

(** [create_context ()] creates a context for storing the "global" state in
    LLVM. See the constructor [llvm::LLVMContext]. *)
val create_context : unit -> llcontext

(** [destroy_context ()] destroys a context. See the destructor
    [llvm::LLVMContext::~LLVMContext]. *)
val dispose_context : llcontext -> unit

(** See the function [llvm::getGlobalContext]. *)
val global_context : unit -> llcontext

(** [mdkind_id context name] returns the MDKind ID that corresponds to the
    name [name] in the context [context].  See the function
    [llvm::LLVMContext::getMDKindID]. *)
val mdkind_id : llcontext -> string -> int


(** {6 Modules} *)

(** [create_module context id] creates a module with the supplied module ID in
    the context [context].  Modules are not garbage collected; it is mandatory
    to call {!dispose_module} to free memory. See the constructor
    [llvm::Module::Module]. *)
val create_module : llcontext -> string -> llmodule

(** [dispose_module m] destroys a module [m] and all of the IR objects it
    contained. All references to subordinate objects are invalidated;
    referencing them will invoke undefined behavior. See the destructor
    [llvm::Module::~Module]. *)
val dispose_module : llmodule -> unit

(** [target_triple m] is the target specifier for the module [m], something like
    [i686-apple-darwin8]. See the method [llvm::Module::getTargetTriple]. *)
val target_triple: llmodule -> string


(** [target_triple triple m] changes the target specifier for the module [m] to
    the string [triple]. See the method [llvm::Module::setTargetTriple]. *)
val set_target_triple: string -> llmodule -> unit


(** [data_layout m] is the data layout specifier for the module [m], something
    like [e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-...-a0:0:64-f80:128:128]. See the
    method [llvm::Module::getDataLayout]. *)
val data_layout: llmodule -> string


(** [set_data_layout s m] changes the data layout specifier for the module [m]
    to the string [s]. See the method [llvm::Module::setDataLayout]. *)
val set_data_layout: string -> llmodule -> unit

(** [dump_module m] prints the .ll representation of the module [m] to standard
    error. See the method [llvm::Module::dump]. *)
val dump_module : llmodule -> unit

(** [set_module_inline_asm m asm] sets the inline assembler for the module. See
    the method [llvm::Module::setModuleInlineAsm]. *)
val set_module_inline_asm : llmodule -> string -> unit

(** [module_context m] returns the context of the specified module.
 * See the method [llvm::Module::getContext] *)
val module_context : llmodule -> llcontext

(** {6 Types} *)

(** [classify_type ty] returns the {!TypeKind.t} corresponding to the type [ty].
    See the method [llvm::Type::getTypeID]. *)
val classify_type : lltype -> TypeKind.t

(** [type_is_sized ty] returns whether the type has a size or not.
 * If it doesn't then it is not safe to call the [DataLayout::] methods on it.
 * *)
val type_is_sized : lltype -> bool

(** [type_context ty] returns the {!llcontext} corresponding to the type [ty].
    See the method [llvm::Type::getContext]. *)
val type_context : lltype -> llcontext

(** [string_of_lltype ty] returns a string describing the type [ty]. *)
val string_of_lltype : lltype -> string

(** {7 Operations on integer types} *)

(** [i1_type c] returns an integer type of bitwidth 1 in the context [c]. See
    [llvm::Type::Int1Ty]. *)
val i1_type : llcontext -> lltype

(** [i8_type c] returns an integer type of bitwidth 8 in the context [c]. See
    [llvm::Type::Int8Ty]. *)
val i8_type : llcontext -> lltype

(** [i16_type c] returns an integer type of bitwidth 16 in the context [c]. See
    [llvm::Type::Int16Ty]. *)
val i16_type : llcontext -> lltype

(** [i32_type c] returns an integer type of bitwidth 32 in the context [c]. See
    [llvm::Type::Int32Ty]. *)
val i32_type : llcontext -> lltype

(** [i64_type c] returns an integer type of bitwidth 64 in the context [c]. See
    [llvm::Type::Int64Ty]. *)
val i64_type : llcontext -> lltype

(** [integer_type c n] returns an integer type of bitwidth [n] in the context
    [c]. See the method [llvm::IntegerType::get]. *)
val integer_type : llcontext -> int -> lltype

(** [integer_bitwidth c ty] returns the number of bits in the integer type [ty]
    in the context [c].  See the method [llvm::IntegerType::getBitWidth]. *)
val integer_bitwidth : lltype -> int


(** {7 Operations on real types} *)

(** [float_type c] returns the IEEE 32-bit floating point type in the context
    [c]. See [llvm::Type::FloatTy]. *)
val float_type : llcontext -> lltype

(** [double_type c] returns the IEEE 64-bit floating point type in the context
    [c]. See [llvm::Type::DoubleTy]. *)
val double_type : llcontext -> lltype

(** [x86fp80_type c] returns the x87 80-bit floating point type in the context
    [c]. See [llvm::Type::X86_FP80Ty]. *)
val x86fp80_type : llcontext -> lltype

(** [fp128_type c] returns the IEEE 128-bit floating point type in the context
    [c]. See [llvm::Type::FP128Ty]. *)
val fp128_type : llcontext -> lltype

(** [ppc_fp128_type c] returns the PowerPC 128-bit floating point type in the
    context [c]. See [llvm::Type::PPC_FP128Ty]. *)
val ppc_fp128_type : llcontext -> lltype


(** {7 Operations on function types} *)

(** [function_type ret_ty param_tys] returns the function type returning
    [ret_ty] and taking [param_tys] as parameters.
    See the method [llvm::FunctionType::get]. *)
val function_type : lltype -> lltype array -> lltype

(** [var_arg_function_type ret_ty param_tys] is just like
    [function_type ret_ty param_tys] except that it returns the function type
    which also takes a variable number of arguments.
    See the method [llvm::FunctionType::get]. *)
val var_arg_function_type : lltype -> lltype array -> lltype


(** [is_var_arg fty] returns [true] if [fty] is a varargs function type, [false]
    otherwise. See the method [llvm::FunctionType::isVarArg]. *)
val is_var_arg : lltype -> bool

(** [return_type fty] gets the return type of the function type [fty].
    See the method [llvm::FunctionType::getReturnType]. *)
val return_type : lltype -> lltype

(** [param_types fty] gets the parameter types of the function type [fty].
    See the method [llvm::FunctionType::getParamType]. *)
val param_types : lltype -> lltype array


(** {7 Operations on struct types} *)

(** [struct_type context tys] returns the structure type in the context
    [context] containing in the types in the array [tys]. See the method
    [llvm::StructType::get]. *)
val struct_type : llcontext -> lltype array -> lltype


(** [packed_struct_type context ys] returns the packed structure type in the
    context [context] containing in the types in the array [tys]. See the method
    [llvm::StructType::get]. *)
val packed_struct_type : llcontext -> lltype array -> lltype

(** [struct_name ty] returns the name of the named structure type [ty],
 * or None if the structure type is not named *)
val struct_name : lltype -> string option

(** [named_struct_type context name] returns the named structure type [name]
 * in the context [context].
 * See the method [llvm::StructType::get]. *)
val named_struct_type : llcontext -> string -> lltype

(** [struct_set_body ty elts ispacked] sets the body of the named struct [ty]
 * to the [elts] elements.
 * See the moethd [llvm::StructType::setBody]. *)
val struct_set_body : lltype -> lltype array -> bool -> unit

(** [struct_element_types sty] returns the constituent types of the struct type
    [sty]. See the method [llvm::StructType::getElementType]. *)
val struct_element_types : lltype -> lltype array


(** [is_packed sty] returns [true] if the structure type [sty] is packed,
    [false] otherwise. See the method [llvm::StructType::isPacked]. *)
val is_packed : lltype -> bool

(** [is_opaque sty] returns [true] if the structure type [sty] is opaque.
    [false] otherwise. See the method [llvm::StructType::isOpaque]. *)
val is_opaque : lltype -> bool

(** {7 Operations on pointer, vector, and array types} *)

(** [array_type ty n] returns the array type containing [n] elements of type
    [ty]. See the method [llvm::ArrayType::get]. *)
val array_type : lltype -> int -> lltype

(** [pointer_type ty] returns the pointer type referencing objects of type
    [ty] in the default address space (0).
    See the method [llvm::PointerType::getUnqual]. *)
val pointer_type : lltype -> lltype

(** [qualified_pointer_type ty as] returns the pointer type referencing objects
    of type [ty] in address space [as].
    See the method [llvm::PointerType::get]. *)
val qualified_pointer_type : lltype -> int -> lltype


(** [vector_type ty n] returns the array type containing [n] elements of the
    primitive type [ty]. See the method [llvm::ArrayType::get]. *)
val vector_type : lltype -> int -> lltype

(** [element_type ty] returns the element type of the pointer, vector, or array
    type [ty]. See the method [llvm::SequentialType::get]. *)
val element_type : lltype -> lltype

(** [element_type aty] returns the element count of the array type [aty].
    See the method [llvm::ArrayType::getNumElements]. *)
val array_length : lltype -> int

(** [address_space pty] returns the address space qualifier of the pointer type
    [pty]. See the method [llvm::PointerType::getAddressSpace]. *)
val address_space : lltype -> int

(** [element_type ty] returns the element count of the vector type [ty].
    See the method [llvm::VectorType::getNumElements]. *)
val vector_size : lltype -> int


(** {7 Operations on other types} *)

(** [void_type c] creates a type of a function which does not return any
    value in the context [c]. See [llvm::Type::VoidTy]. *)
val void_type : llcontext -> lltype

(** [label_type c] creates a type of a basic block in the context [c]. See
    [llvm::Type::LabelTy]. *)
val label_type : llcontext -> lltype

(** [type_by_name m name] returns the specified type from the current module
 * if it exists.
 * See the method [llvm::Module::getTypeByName] *)
val type_by_name : llmodule -> string -> lltype option

(* {6 Values} *)

(** [type_of v] returns the type of the value [v].
    See the method [llvm::Value::getType]. *)
val type_of : llvalue -> lltype

val classify_value : llvalue -> ValueKind.t

(** [value_name v] returns the name of the value [v]. For global values, this is
    the symbol name. For instructions and basic blocks, it is the SSA register
    name. It is meaningless for constants.
    See the method [llvm::Value::getName]. *)
val value_name : llvalue -> string

(** [set_value_name n v] sets the name of the value [v] to [n]. See the method
    [llvm::Value::setName]. *)
val set_value_name : string -> llvalue -> unit

(** [dump_value v] prints the .ll representation of the value [v] to standard
    error. See the method [llvm::Value::dump]. *)
val dump_value : llvalue -> unit

(** [replace_all_uses_with old new] replaces all uses of the value [old]
 * with the value [new]. See the method [llvm::Value::replaceAllUsesWith]. *)
val replace_all_uses_with : llvalue -> llvalue -> unit



(* {6 Uses} *)

(** [use_begin v] returns the first position in the use list for the value [v].
    [use_begin] and [use_succ] can e used to iterate over the use list in order.
    See the method [llvm::Value::use_begin]. *)
val use_begin : llvalue -> lluse option

(** [use_succ u] returns the use list position succeeding [u].
    See the method [llvm::use_value_iterator::operator++]. *)
val use_succ : lluse -> lluse option

(** [user u] returns the user of the use [u].
    See the method [llvm::Use::getUser]. *)
val user : lluse -> llvalue

(** [used_value u] returns the usee of the use [u].
    See the method [llvm::Use::getUsedValue]. *)
val used_value : lluse -> llvalue

(** [iter_uses f v] applies function [f] to each of the users of the value [v]
    in order. Tail recursive. *)
val iter_uses : (lluse -> unit) -> llvalue -> unit

(** [fold_left_uses f init v] is [f (... (f init u1) ...) uN] where
    [u1,...,uN] are the users of the value [v]. Tail recursive. *)
val fold_left_uses : ('a -> lluse -> 'a) -> 'a -> llvalue -> 'a

(** [fold_right_uses f v init] is [f u1 (... (f uN init) ...)] where
    [u1,...,uN] are the users of the value [v]. Not tail recursive. *)
val fold_right_uses : (lluse -> 'a -> 'a) -> llvalue -> 'a -> 'a


(* {6 Users} *)

(** [operand v i] returns the operand at index [i] for the value [v]. See the
    method [llvm::User::getOperand]. *)
val operand : llvalue -> int -> llvalue

(** [set_operand v i o] sets the operand of the value [v] at the index [i] to
    the value [o].
    See the method [llvm::User::setOperand]. *)
val set_operand : llvalue -> int -> llvalue -> unit

(** [num_operands v] returns the number of operands for the value [v].
    See the method [llvm::User::getNumOperands]. *)
val num_operands : llvalue -> int

(** {7 Operations on constants of (mostly) any type} *)

(** [is_constant v] returns [true] if the value [v] is a constant, [false]
    otherwise. Similar to [llvm::isa<Constant>]. *)
val is_constant : llvalue -> bool

(** [const_null ty] returns the constant null (zero) of the type [ty].
    See the method [llvm::Constant::getNullValue]. *)
val const_null : lltype -> llvalue

(** [const_all_ones ty] returns the constant '-1' of the integer or vector type
    [ty]. See the method [llvm::Constant::getAllOnesValue]. *)
val const_all_ones : (*int|vec*)lltype -> llvalue

(** [const_pointer_null ty] returns the constant null (zero) pointer of the type
    [ty]. See the method [llvm::ConstantPointerNull::get]. *)
val const_pointer_null : lltype -> llvalue

(** [undef ty] returns the undefined value of the type [ty].
    See the method [llvm::UndefValue::get]. *)
val undef : lltype -> llvalue

(** [is_null v] returns [true] if the value [v] is the null (zero) value.
    See the method [llvm::Constant::isNullValue]. *)
val is_null : llvalue -> bool

(** [is_undef v] returns [true] if the value [v] is an undefined value, [false]
    otherwise. Similar to [llvm::isa<UndefValue>]. *)
val is_undef : llvalue -> bool

val constexpr_opcode : llvalue -> Opcode.t
(** {7 Operations on instructions} *)

(** [has_metadata i] returns whether or not the instruction [i] has any
    metadata attached to it. See the function
    [llvm::Instruction::hasMetadata]. *)
val has_metadata : llvalue -> bool

(** [metadata i kind] optionally returns the metadata associated with the
    kind [kind] in the instruction [i] See the function
    [llvm::Instruction::getMetadata]. *)
val metadata : llvalue -> int -> llvalue option

(** [set_metadata i kind md] sets the metadata [md] of kind [kind] in the
    instruction [i]. See the function [llvm::Instruction::setMetadata]. *)
val set_metadata : llvalue -> int -> llvalue -> unit

(** [clear_metadata i kind] clears the metadata of kind [kind] in the
    instruction [i]. See the function [llvm::Instruction::setMetadata]. *)
val clear_metadata : llvalue -> int -> unit


(** {7 Operations on metadata} *)

(** [mdstring c s] returns the MDString of the string [s] in the context [c].
    See the method [llvm::MDNode::get]. *)
val mdstring : llcontext -> string -> llvalue

(** [mdnode c elts] returns the MDNode containing the values [elts] in the
    context [c].
    See the method [llvm::MDNode::get]. *)
val mdnode : llcontext -> llvalue array -> llvalue

(** [get_mdstring v] returns the MDString.
 * See the method [llvm::MDString::getString] *)
val get_mdstring : llvalue -> string option

(** [get_named_metadata m name] return all the MDNodes belonging to the named
 * metadata (if any).
 * See the method [llvm::NamedMDNode::getOperand]. *)
val get_named_metadata : llmodule -> string -> llvalue array

(** {7 Operations on scalar constants} *)

(** [const_int ty i] returns the integer constant of type [ty] and value [i].
    See the method [llvm::ConstantInt::get]. *)
val const_int : lltype -> int -> llvalue

(** [const_of_int64 ty i] returns the integer constant of type [ty] and value
    [i]. See the method [llvm::ConstantInt::get]. *)
val const_of_int64 : lltype -> Int64.t -> bool -> llvalue

(** [int64_of_const c] returns the int64 value of the [c] constant integer.
 * None is returned if this is not an integer constant, or bitwidth exceeds 64.
 * See the method [llvm::ConstantInt::getSExtValue].*)
val int64_of_const : llvalue -> Int64.t option

(** [const_int_of_string ty s r] returns the integer constant of type [ty] and
 * value [s], with the radix [r]. See the method [llvm::ConstantInt::get]. *)
val const_int_of_string : lltype -> string -> int -> llvalue


(** [const_float ty n] returns the floating point constant of type [ty] and
    value [n]. See the method [llvm::ConstantFP::get]. *)
val const_float : lltype -> float -> llvalue

(** [const_float_of_string ty s] returns the floating point constant of type
    [ty] and value [n]. See the method [llvm::ConstantFP::get]. *)
val const_float_of_string : lltype -> string -> llvalue



(** {7 Operations on composite constants} *)

(** [const_string c s] returns the constant [i8] array with the values of the
    characters in the string [s] in the context [c]. The array is not 
    null-terminated (but see {!const_stringz}). This value can in turn be used
    as the initializer for a global variable. See the method
    [llvm::ConstantArray::get]. *)
val const_string : llcontext -> string -> llvalue

(** [const_stringz c s] returns the constant [i8] array with the values of the
    characters in the string [s] and a null terminator in the context [c]. This
    value can in turn be used as the initializer for a global variable.
    See the method [llvm::ConstantArray::get]. *)
val const_stringz : llcontext -> string -> llvalue

(** [const_array ty elts] returns the constant array of type
    [array_type ty (Array.length elts)] and containing the values [elts].
    This value can in turn be used as the initializer for a global variable.
    See the method [llvm::ConstantArray::get]. *)
val const_array : lltype -> llvalue array -> llvalue

(** [const_struct context elts] returns the structured constant of type
    [struct_type (Array.map type_of elts)] and containing the values [elts]
    in the context [context]. This value can in turn be used as the initializer
    for a global variable. See the method [llvm::ConstantStruct::getAnon]. *)
val const_struct : llcontext -> llvalue array -> llvalue

(** [const_named_struct namedty elts] returns the structured constant of type
    [namedty] (which must be a named structure type) and containing the values [elts].
    This value can in turn be used as the initializer
    for a global variable. See the method [llvm::ConstantStruct::get]. *)
val const_named_struct : lltype -> llvalue array -> llvalue

(** [const_packed_struct context elts] returns the structured constant of
    type {!packed_struct_type} [(Array.map type_of elts)] and containing the
    values [elts] in the context [context]. This value can in turn be used as
    the initializer for a global variable. See the method
    [llvm::ConstantStruct::get]. *)
val const_packed_struct : llcontext -> llvalue array -> llvalue


(** [const_vector elts] returns the vector constant of type
    [vector_type (type_of elts.(0)) (Array.length elts)] and containing the
    values [elts]. See the method [llvm::ConstantVector::get]. *)
val const_vector : llvalue array -> llvalue


(** {7 Constant expressions} *)

(** [align_of ty] returns the alignof constant for the type [ty]. This is
    equivalent to [const_ptrtoint (const_gep (const_null (pointer_type {i8,ty}))
    (const_int i32_type 0) (const_int i32_type 1)) i32_type], but considerably
    more readable.  See the method [llvm::ConstantExpr::getAlignOf]. *)
val align_of : lltype -> llvalue

(** [size_of ty] returns the sizeof constant for the type [ty]. This is
    equivalent to [const_ptrtoint (const_gep (const_null (pointer_type ty))
    (const_int i32_type 1)) i64_type], but considerably more readable.
    See the method [llvm::ConstantExpr::getSizeOf]. *)
val size_of : lltype -> llvalue

(** [const_neg c] returns the arithmetic negation of the constant [c].
    See the method [llvm::ConstantExpr::getNeg]. *)
val const_neg : llvalue -> llvalue

(** [const_nsw_neg c] returns the arithmetic negation of the constant [c] with
    no signed wrapping. The result is undefined if the negation overflows.
    See the method [llvm::ConstantExpr::getNSWNeg]. *)
val const_nsw_neg : llvalue -> llvalue

(** [const_nuw_neg c] returns the arithmetic negation of the constant [c] with
    no unsigned wrapping. The result is undefined if the negation overflows.
    See the method [llvm::ConstantExpr::getNUWNeg]. *)
val const_nuw_neg : llvalue -> llvalue

(** [const_fneg c] returns the arithmetic negation of the constant float [c].
    See the method [llvm::ConstantExpr::getFNeg]. *)
val const_fneg : llvalue -> llvalue

(** [const_not c] returns the bitwise inverse of the constant [c].
    See the method [llvm::ConstantExpr::getNot]. *)
val const_not : llvalue -> llvalue

(** [const_add c1 c2] returns the constant sum of two constants.
    See the method [llvm::ConstantExpr::getAdd]. *)
val const_add : llvalue -> llvalue -> llvalue

(** [const_nsw_add c1 c2] returns the constant sum of two constants with no
    signed wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWAdd]. *)
val const_nsw_add : llvalue -> llvalue -> llvalue

(** [const_nuw_add c1 c2] returns the constant sum of two constants with no
    unsigned wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWAdd]. *)
val const_nuw_add : llvalue -> llvalue -> llvalue

(** [const_fadd c1 c2] returns the constant sum of two constant floats.
    See the method [llvm::ConstantExpr::getFAdd]. *)
val const_fadd : llvalue -> llvalue -> llvalue

(** [const_sub c1 c2] returns the constant difference, [c1 - c2], of two
    constants. See the method [llvm::ConstantExpr::getSub]. *)
val const_sub : llvalue -> llvalue -> llvalue

(** [const_nsw_sub c1 c2] returns the constant difference of two constants with
    no signed wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWSub]. *)
val const_nsw_sub : llvalue -> llvalue -> llvalue

(** [const_nuw_sub c1 c2] returns the constant difference of two constants with
    no unsigned wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWSub]. *)
val const_nuw_sub : llvalue -> llvalue -> llvalue

(** [const_fsub c1 c2] returns the constant difference, [c1 - c2], of two
    constant floats. See the method [llvm::ConstantExpr::getFSub]. *)
val const_fsub : llvalue -> llvalue -> llvalue

(** [const_mul c1 c2] returns the constant product of two constants.
    See the method [llvm::ConstantExpr::getMul]. *)
val const_mul : llvalue -> llvalue -> llvalue

(** [const_nsw_mul c1 c2] returns the constant product of two constants with
    no signed wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWMul]. *)
val const_nsw_mul : llvalue -> llvalue -> llvalue

(** [const_nuw_mul c1 c2] returns the constant product of two constants with
    no unsigned wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWMul]. *)
val const_nuw_mul : llvalue -> llvalue -> llvalue

(** [const_fmul c1 c2] returns the constant product of two constants floats.
    See the method [llvm::ConstantExpr::getFMul]. *)
val const_fmul : llvalue -> llvalue -> llvalue

(** [const_udiv c1 c2] returns the constant quotient [c1 / c2] of two unsigned
    integer constants.
    See the method [llvm::ConstantExpr::getUDiv]. *)
val const_udiv : llvalue -> llvalue -> llvalue

(** [const_sdiv c1 c2] returns the constant quotient [c1 / c2] of two signed
    integer constants.
    See the method [llvm::ConstantExpr::getSDiv]. *)
val const_sdiv : llvalue -> llvalue -> llvalue

(** [const_exact_sdiv c1 c2] returns the constant quotient [c1 / c2] of two
    signed integer constants. The result is undefined if the result is rounded
    or overflows. See the method [llvm::ConstantExpr::getExactSDiv]. *)
val const_exact_sdiv : llvalue -> llvalue -> llvalue

(** [const_fdiv c1 c2] returns the constant quotient [c1 / c2] of two floating
    point constants.
    See the method [llvm::ConstantExpr::getFDiv]. *)
val const_fdiv : llvalue -> llvalue -> llvalue

(** [const_urem c1 c2] returns the constant remainder [c1 MOD c2] of two
    unsigned integer constants.
    See the method [llvm::ConstantExpr::getURem]. *)
val const_urem : llvalue -> llvalue -> llvalue

(** [const_srem c1 c2] returns the constant remainder [c1 MOD c2] of two
    signed integer constants.
    See the method [llvm::ConstantExpr::getSRem]. *)
val const_srem : llvalue -> llvalue -> llvalue

(** [const_frem c1 c2] returns the constant remainder [c1 MOD c2] of two
    signed floating point constants.
    See the method [llvm::ConstantExpr::getFRem]. *)
val const_frem : llvalue -> llvalue -> llvalue

(** [const_and c1 c2] returns the constant bitwise [AND] of two integer
    constants.
    See the method [llvm::ConstantExpr::getAnd]. *)
val const_and : llvalue -> llvalue -> llvalue

(** [const_or c1 c2] returns the constant bitwise [OR] of two integer
    constants.
    See the method [llvm::ConstantExpr::getOr]. *)
val const_or : llvalue -> llvalue -> llvalue

(** [const_xor c1 c2] returns the constant bitwise [XOR] of two integer
    constants.
    See the method [llvm::ConstantExpr::getXor]. *)
val const_xor : llvalue -> llvalue -> llvalue

(** [const_icmp pred c1 c2] returns the constant comparison of two integer
    constants, [c1 pred c2].
    See the method [llvm::ConstantExpr::getICmp]. *)
val const_icmp : Icmp.t -> llvalue -> llvalue -> llvalue


(** [const_fcmp pred c1 c2] returns the constant comparison of two floating
    point constants, [c1 pred c2].
    See the method [llvm::ConstantExpr::getFCmp]. *)
val const_fcmp : Fcmp.t -> llvalue -> llvalue -> llvalue


(** [const_shl c1 c2] returns the constant integer [c1] left-shifted by the
    constant integer [c2].
    See the method [llvm::ConstantExpr::getShl]. *)
val const_shl : llvalue -> llvalue -> llvalue

(** [const_lshr c1 c2] returns the constant integer [c1] right-shifted by the
    constant integer [c2] with zero extension.
    See the method [llvm::ConstantExpr::getLShr]. *)
val const_lshr : llvalue -> llvalue -> llvalue

(** [const_ashr c1 c2] returns the constant integer [c1] right-shifted by the
    constant integer [c2] with sign extension.
    See the method [llvm::ConstantExpr::getAShr]. *)
val const_ashr : llvalue -> llvalue -> llvalue

(** [const_gep pc indices] returns the constant [getElementPtr] of [p1] with the
    constant integers indices from the array [indices].
    See the method [llvm::ConstantExpr::getGetElementPtr]. *)
val const_gep : llvalue -> llvalue array -> llvalue

(** [const_in_bounds_gep pc indices] returns the constant [getElementPtr] of [p1]
    with the constant integers indices from the array [indices].
    See the method [llvm::ConstantExpr::getInBoundsGetElementPtr]. *)
val const_in_bounds_gep : llvalue -> llvalue array -> llvalue


(** [const_trunc c ty] returns the constant truncation of integer constant [c]
    to the smaller integer type [ty].
    See the method [llvm::ConstantExpr::getTrunc]. *)
val const_trunc : llvalue -> lltype -> llvalue

(** [const_sext c ty] returns the constant sign extension of integer constant
    [c] to the larger integer type [ty].
    See the method [llvm::ConstantExpr::getSExt]. *)
val const_sext : llvalue -> lltype -> llvalue

(** [const_zext c ty] returns the constant zero extension of integer constant
    [c] to the larger integer type [ty].
    See the method [llvm::ConstantExpr::getZExt]. *)
val const_zext : llvalue -> lltype -> llvalue

(** [const_fptrunc c ty] returns the constant truncation of floating point
    constant [c] to the smaller floating point type [ty].
    See the method [llvm::ConstantExpr::getFPTrunc]. *)
val const_fptrunc : llvalue -> lltype -> llvalue

(** [const_fpext c ty] returns the constant extension of floating point constant
    [c] to the larger floating point type [ty].
    See the method [llvm::ConstantExpr::getFPExt]. *)
val const_fpext : llvalue -> lltype -> llvalue

(** [const_uitofp c ty] returns the constant floating point conversion of
    unsigned integer constant [c] to the floating point type [ty].
    See the method [llvm::ConstantExpr::getUIToFP]. *)
val const_uitofp : llvalue -> lltype -> llvalue

(** [const_sitofp c ty] returns the constant floating point conversion of
    signed integer constant [c] to the floating point type [ty].
    See the method [llvm::ConstantExpr::getSIToFP]. *)
val const_sitofp : llvalue -> lltype -> llvalue

(** [const_fptoui c ty] returns the constant unsigned integer conversion of
    floating point constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getFPToUI]. *)
val const_fptoui : llvalue -> lltype -> llvalue

(** [const_fptoui c ty] returns the constant unsigned integer conversion of
    floating point constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getFPToSI]. *)
val const_fptosi : llvalue -> lltype -> llvalue

(** [const_ptrtoint c ty] returns the constant integer conversion of
    pointer constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getPtrToInt]. *)
val const_ptrtoint : llvalue -> lltype -> llvalue

(** [const_inttoptr c ty] returns the constant pointer conversion of
    integer constant [c] to pointer type [ty].
    See the method [llvm::ConstantExpr::getIntToPtr]. *)
val const_inttoptr : llvalue -> lltype -> llvalue

(** [const_bitcast c ty] returns the constant bitwise conversion of constant [c]
    to type [ty] of equal size.
    See the method [llvm::ConstantExpr::getBitCast]. *)
val const_bitcast : llvalue -> lltype -> llvalue

(** [const_zext_or_bitcast c ty] returns a constant zext or bitwise cast
    conversion of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getZExtOrBitCast]. *)
val const_zext_or_bitcast : llvalue -> lltype -> llvalue


(** [const_sext_or_bitcast c ty] returns a constant sext or bitwise cast
    conversion of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getSExtOrBitCast]. *)
val const_sext_or_bitcast : llvalue -> lltype -> llvalue


(** [const_trunc_or_bitcast c ty] returns a constant trunc or bitwise cast
    conversion of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getTruncOrBitCast]. *)
val const_trunc_or_bitcast : llvalue -> lltype -> llvalue


(** [const_pointercast c ty] returns a constant bitcast or a pointer-to-int
    cast conversion of constant [c] to type [ty] of equal size.
    See the method [llvm::ConstantExpr::getPointerCast]. *)
val const_pointercast : llvalue -> lltype -> llvalue


(** [const_intcast c ty] returns a constant zext, bitcast, or trunc for integer
    -> integer casts of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getIntCast]. *)
val const_intcast : llvalue -> lltype -> llvalue


(** [const_fpcast c ty] returns a constant fpext, bitcast, or fptrunc for fp ->
    fp casts of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getFPCast]. *)
val const_fpcast : llvalue -> lltype -> llvalue


(** [const_select cond t f] returns the constant conditional which returns value
    [t] if the boolean constant [cond] is true and the value [f] otherwise.
    See the method [llvm::ConstantExpr::getSelect]. *)
val const_select : llvalue -> llvalue -> llvalue -> llvalue


(** [const_extractelement vec i] returns the constant [i]th element of
    constant vector [vec]. [i] must be a constant [i32] value unsigned less than
    the size of the vector.
    See the method [llvm::ConstantExpr::getExtractElement]. *)
val const_extractelement : llvalue -> llvalue -> llvalue


(** [const_insertelement vec v i] returns the constant vector with the same
    elements as constant vector [v] but the [i]th element replaced by the
    constant [v]. [v] must be a constant value with the type of the vector
    elements. [i] must be a constant [i32] value unsigned less than the size
    of the vector.
    See the method [llvm::ConstantExpr::getInsertElement]. *)
val const_insertelement : llvalue -> llvalue -> llvalue -> llvalue


(** [const_shufflevector a b mask] returns a constant [shufflevector].
    See the LLVM Language Reference for details on the [shufflevector]
    instruction.
    See the method [llvm::ConstantExpr::getShuffleVector]. *)
val const_shufflevector : llvalue -> llvalue -> llvalue -> llvalue


(** [const_extractvalue agg idxs] returns the constant [idxs]th value of
    constant aggregate [agg]. Each [idxs] must be less than the size of the
    aggregate.  See the method [llvm::ConstantExpr::getExtractValue]. *)
val const_extractvalue : llvalue -> int array -> llvalue


(** [const_insertvalue agg val idxs] inserts the value [val] in the specified
    indexs [idxs] in the aggegate [agg]. Each [idxs] must be less than the size
    of the aggregate. See the method [llvm::ConstantExpr::getInsertValue]. *)
val const_insertvalue : llvalue -> llvalue -> int array -> llvalue


(** [const_inline_asm ty asm con side align] inserts a inline assembly string.
    See the method [llvm::InlineAsm::get]. *)
val const_inline_asm : lltype -> string -> string -> bool -> bool ->
                            llvalue


(** [block_address f bb] returns the address of the basic block [bb] in the
    function [f]. See the method [llvm::BasicBlock::get]. *)
val block_address : llvalue -> llbasicblock -> llvalue


(** {7 Operations on global variables, functions, and aliases (globals)} *)

(** [global_parent g] is the enclosing module of the global value [g].
    See the method [llvm::GlobalValue::getParent]. *)
val global_parent : llvalue -> llmodule

(** [is_declaration g] returns [true] if the global value [g] is a declaration
    only. Returns [false] otherwise.
    See the method [llvm::GlobalValue::isDeclaration]. *)
val is_declaration : llvalue -> bool

(** [linkage g] returns the linkage of the global value [g].
    See the method [llvm::GlobalValue::getLinkage]. *)
val linkage : llvalue -> Linkage.t

(** [set_linkage l g] sets the linkage of the global value [g] to [l].
    See the method [llvm::GlobalValue::setLinkage]. *)
val set_linkage : Linkage.t -> llvalue -> unit

(** [section g] returns the linker section of the global value [g].
    See the method [llvm::GlobalValue::getSection]. *)
val section : llvalue -> string

(** [set_section s g] sets the linker section of the global value [g] to [s].
    See the method [llvm::GlobalValue::setSection]. *)
val set_section : string -> llvalue -> unit

(** [visibility g] returns the linker visibility of the global value [g].
    See the method [llvm::GlobalValue::getVisibility]. *)
val visibility : llvalue -> Visibility.t

(** [set_visibility v g] sets the linker visibility of the global value [g] to
    [v]. See the method [llvm::GlobalValue::setVisibility]. *)
val set_visibility : Visibility.t -> llvalue -> unit


(** [alignment g] returns the required alignment of the global value [g].
    See the method [llvm::GlobalValue::getAlignment]. *)
val alignment : llvalue -> int

(** [set_alignment n g] sets the required alignment of the global value [g] to
    [n] bytes. See the method [llvm::GlobalValue::setAlignment]. *)
val set_alignment : int -> llvalue -> unit


(** {7 Operations on global variables} *)

(** [declare_global ty name m] returns a new global variable of type [ty] and
    with name [name] in module [m] in the default address space (0). If such a
    global variable already exists, it is returned. If the type of the existing
    global differs, then a bitcast to [ty] is returned. *)
val declare_global : lltype -> string -> llmodule -> llvalue


(** [declare_qualified_global ty name addrspace m] returns a new global variable
    of type [ty] and with name [name] in module [m] in the address space
    [addrspace]. If such a global variable already exists, it is returned. If
    the type of the existing global differs, then a bitcast to [ty] is
    returned. *)
val declare_qualified_global : lltype -> string -> int -> llmodule ->
                                    llvalue


(** [define_global name init m] returns a new global with name [name] and
    initializer [init] in module [m] in the default address space (0). If the
    named global already exists, it is renamed.
    See the constructor of [llvm::GlobalVariable]. *)
val define_global : string -> llvalue -> llmodule -> llvalue


(** [define_qualified_global name init addrspace m] returns a new global with
    name [name] and initializer [init] in module [m] in the address space
    [addrspace]. If the named global already exists, it is renamed.
    See the constructor of [llvm::GlobalVariable]. *)
val define_qualified_global : string -> llvalue -> int -> llmodule ->
                                   llvalue


(** [lookup_global name m] returns [Some g] if a global variable with name
    [name] exists in module [m]. If no such global exists, returns [None].
    See the [llvm::GlobalVariable] constructor. *)
val lookup_global : string -> llmodule -> llvalue option


(** [delete_global gv] destroys the global variable [gv].
    See the method [llvm::GlobalVariable::eraseFromParent]. *)
val delete_global : llvalue -> unit

(** [global_begin m] returns the first position in the global variable list of
    the module [m]. [global_begin] and [global_succ] can be used to iterate
    over the global list in order.
    See the method [llvm::Module::global_begin]. *)
val global_begin : llmodule -> (llmodule, llvalue) llpos


(** [global_succ gv] returns the global variable list position succeeding
    [Before gv].
    See the method [llvm::Module::global_iterator::operator++]. *)
val global_succ : llvalue -> (llmodule, llvalue) llpos


(** [iter_globals f m] applies function [f] to each of the global variables of
    module [m] in order. Tail recursive. *)
val iter_globals : (llvalue -> unit) -> llmodule -> unit

(** [fold_left_globals f init m] is [f (... (f init g1) ...) gN] where
    [g1,...,gN] are the global variables of module [m]. Tail recursive. *)
val fold_left_globals : ('a -> llvalue -> 'a) -> 'a -> llmodule -> 'a

(** [global_end m] returns the last position in the global variable list of the
    module [m]. [global_end] and [global_pred] can be used to iterate over the
    global list in reverse.
    See the method [llvm::Module::global_end]. *)
val global_end : llmodule -> (llmodule, llvalue) llrev_pos


(** [global_pred gv] returns the global variable list position preceding
    [After gv].
    See the method [llvm::Module::global_iterator::operator--]. *)
val global_pred : llvalue -> (llmodule, llvalue) llrev_pos


(** [rev_iter_globals f m] applies function [f] to each of the global variables
    of module [m] in reverse order. Tail recursive. *)
val rev_iter_globals : (llvalue -> unit) -> llmodule -> unit

(** [fold_right_globals f m init] is [f g1 (... (f gN init) ...)] where
    [g1,...,gN] are the global variables of module [m]. Tail recursive. *)
val fold_right_globals : (llvalue -> 'a -> 'a) -> llmodule -> 'a -> 'a

(** [is_global_constant gv] returns [true] if the global variabile [gv] is a
    constant. Returns [false] otherwise.
    See the method [llvm::GlobalVariable::isConstant]. *)
val is_global_constant : llvalue -> bool

(** [set_global_constant c gv] sets the global variable [gv] to be a constant if
    [c] is [true] and not if [c] is [false].
    See the method [llvm::GlobalVariable::setConstant]. *)
val set_global_constant : bool -> llvalue -> unit


(** [global_initializer gv] returns the initializer for the global variable
    [gv]. See the method [llvm::GlobalVariable::getInitializer]. *)
val global_initializer : llvalue -> llvalue

(** [set_initializer c gv] sets the initializer for the global variable
    [gv] to the constant [c].
    See the method [llvm::GlobalVariable::setInitializer]. *)
val set_initializer : llvalue -> llvalue -> unit

(** [remove_initializer gv] unsets the initializer for the global variable
    [gv].
    See the method [llvm::GlobalVariable::setInitializer]. *)
val remove_initializer : llvalue -> unit

(** [is_thread_local gv] returns [true] if the global variable [gv] is
    thread-local and [false] otherwise.
    See the method [llvm::GlobalVariable::isThreadLocal]. *)
val is_thread_local : llvalue -> bool

(** [set_thread_local c gv] sets the global variable [gv] to be thread local if
    [c] is [true] and not otherwise.
    See the method [llvm::GlobalVariable::setThreadLocal]. *)
val set_thread_local : bool -> llvalue -> unit


(** {7 Operations on aliases} *)

(** [add_alias m t a n] inserts an alias in the module [m] with the type [t] and
    the aliasee [a] with the name [n].
    See the constructor for [llvm::GlobalAlias]. *)
val add_alias : llmodule -> lltype -> llvalue -> string -> llvalue



(** {7 Operations on functions} *)

(** [declare_function name ty m] returns a new function of type [ty] and
    with name [name] in module [m]. If such a function already exists,
    it is returned. If the type of the existing function differs, then a bitcast
    to [ty] is returned. *)
val declare_function : string -> lltype -> llmodule -> llvalue


(** [define_function name ty m] creates a new function with name [name] and
    type [ty] in module [m]. If the named function already exists, it is
    renamed. An entry basic block is created in the function.
    See the constructor of [llvm::GlobalVariable]. *)
val define_function : string -> lltype -> llmodule -> llvalue


(** [lookup_function name m] returns [Some f] if a function with name
    [name] exists in module [m]. If no such function exists, returns [None].
    See the method [llvm::Module] constructor. *)
val lookup_function : string -> llmodule -> llvalue option


(** [delete_function f] destroys the function [f].
    See the method [llvm::Function::eraseFromParent]. *)
val delete_function : llvalue -> unit

(** [function_begin m] returns the first position in the function list of the
    module [m]. [function_begin] and [function_succ] can be used to iterate over
    the function list in order.
    See the method [llvm::Module::begin]. *)
val function_begin : llmodule -> (llmodule, llvalue) llpos


(** [function_succ gv] returns the function list position succeeding
    [Before gv].
    See the method [llvm::Module::iterator::operator++]. *)
val function_succ : llvalue -> (llmodule, llvalue) llpos


(** [iter_functions f m] applies function [f] to each of the functions of module
    [m] in order. Tail recursive. *)
val iter_functions : (llvalue -> unit) -> llmodule -> unit

(** [fold_left_function f init m] is [f (... (f init f1) ...) fN] where
    [f1,...,fN] are the functions of module [m]. Tail recursive. *)
val fold_left_functions : ('a -> llvalue -> 'a) -> 'a -> llmodule -> 'a

(** [function_end m] returns the last position in the function list of
    the module [m]. [function_end] and [function_pred] can be used to iterate
    over the function list in reverse.
    See the method [llvm::Module::end]. *)
val function_end : llmodule -> (llmodule, llvalue) llrev_pos


(** [function_pred gv] returns the function list position preceding [After gv].
    See the method [llvm::Module::iterator::operator--]. *)
val function_pred : llvalue -> (llmodule, llvalue) llrev_pos


(** [rev_iter_functions f fn] applies function [f] to each of the functions of
    module [m] in reverse order. Tail recursive. *)
val rev_iter_functions : (llvalue -> unit) -> llmodule -> unit

(** [fold_right_functions f m init] is [f (... (f init fN) ...) f1] where
    [f1,...,fN] are the functions of module [m]. Tail recursive. *)
val fold_right_functions : (llvalue -> 'a -> 'a) -> llmodule -> 'a -> 'a

(** [is_intrinsic f] returns true if the function [f] is an intrinsic.
    See the method [llvm::Function::isIntrinsic]. *)
val is_intrinsic : llvalue -> bool

(** [function_call_conv f] returns the calling convention of the function [f].
    See the method [llvm::Function::getCallingConv]. *)
val function_call_conv : llvalue -> int

(** [set_function_call_conv cc f] sets the calling convention of the function
    [f] to the calling convention numbered [cc].
    See the method [llvm::Function::setCallingConv]. *)
val set_function_call_conv : int -> llvalue -> unit


(** [gc f] returns [Some name] if the function [f] has a garbage
    collection algorithm specified and [None] otherwise.
    See the method [llvm::Function::getGC]. *)
val gc : llvalue -> string option

(** [set_gc gc f] sets the collection algorithm for the function [f] to
    [gc]. See the method [llvm::Function::setGC]. *)
val set_gc : string option -> llvalue -> unit

(** [add_function_attr f a] adds attribute [a] to the return type of function
    [f]. *)
val add_function_attr : llvalue -> Attribute.t -> unit

(** [function_attr f] returns the function attribute for the function [f].
 * See the method [llvm::Function::getAttributes] *)
val function_attr : llvalue -> Attribute.t list

(** [remove_function_attr f a] removes attribute [a] from the return type of
    function [f]. *)
val remove_function_attr : llvalue -> Attribute.t -> unit

(** {7 Operations on params} *)

(** [params f] returns the parameters of function [f].
    See the method [llvm::Function::getArgumentList]. *)
val params : llvalue -> llvalue array

(** [param f n] returns the [n]th parameter of function [f].
    See the method [llvm::Function::getArgumentList]. *)
val param : llvalue -> int -> llvalue

(** [param_attr p] returns the attributes of parameter [p].
 * See the methods [llvm::Function::getAttributes] and
 * [llvm::Attributes::getParamAttributes] *)
val param_attr : llvalue -> Attribute.t list

(** [param_parent p] returns the parent function that owns the parameter.
    See the method [llvm::Argument::getParent]. *)
val param_parent : llvalue -> llvalue

(** [param_begin f] returns the first position in the parameter list of the
    function [f]. [param_begin] and [param_succ] can be used to iterate over
    the parameter list in order.
    See the method [llvm::Function::arg_begin]. *)
val param_begin : llvalue -> (llvalue, llvalue) llpos

(** [param_succ bb] returns the parameter list position succeeding
    [Before bb].
    See the method [llvm::Function::arg_iterator::operator++]. *)
val param_succ : llvalue -> (llvalue, llvalue) llpos

(** [iter_params f fn] applies function [f] to each of the parameters
    of function [fn] in order. Tail recursive. *)
val iter_params : (llvalue -> unit) -> llvalue -> unit

(** [fold_left_params f init fn] is [f (... (f init b1) ...) bN] where
    [b1,...,bN] are the parameters of function [fn]. Tail recursive. *)
val fold_left_params : ('a -> llvalue -> 'a) -> 'a -> llvalue -> 'a

(** [param_end f] returns the last position in the parameter list of
    the function [f]. [param_end] and [param_pred] can be used to iterate
    over the parameter list in reverse.
    See the method [llvm::Function::arg_end]. *)
val param_end : llvalue -> (llvalue, llvalue) llrev_pos

(** [param_pred gv] returns the function list position preceding [After gv].
    See the method [llvm::Function::arg_iterator::operator--]. *)
val param_pred : llvalue -> (llvalue, llvalue) llrev_pos


(** [rev_iter_params f fn] applies function [f] to each of the parameters
    of function [fn] in reverse order. Tail recursive. *)
val rev_iter_params : (llvalue -> unit) -> llvalue -> unit

(** [fold_right_params f fn init] is [f (... (f init bN) ...) b1] where
    [b1,...,bN] are the parameters of function [fn]. Tail recursive. *)
val fold_right_params : (llvalue -> 'a -> 'a) -> llvalue -> 'a -> 'a

(** [add_param p a] adds attribute [a] to parameter [p]. *)
val add_param_attr : llvalue -> Attribute.t -> unit

(** [remove_param_attr p a] removes attribute [a] from parameter [p]. *)
val remove_param_attr : llvalue -> Attribute.t -> unit

(** [set_param_alignment p a] set the alignment of parameter [p] to [a]. *)
val set_param_alignment : llvalue -> int -> unit


(** {7 Operations on basic blocks} *)

(** [basic_blocks fn] returns the basic blocks of the function [f].
    See the method [llvm::Function::getBasicBlockList]. *)
val basic_blocks : llvalue -> llbasicblock array

(** [entry_block fn] returns the entry basic block of the function [f].
    See the method [llvm::Function::getEntryBlock]. *)
val entry_block : llvalue -> llbasicblock

(** [delete_block bb] deletes the basic block [bb].
    See the method [llvm::BasicBlock::eraseFromParent]. *)
val delete_block : llbasicblock -> unit

(** [append_block c name f] creates a new basic block named [name] at the end of
    function [f] in the context [c].
    See the constructor of [llvm::BasicBlock]. *)
val append_block : llcontext -> string -> llvalue -> llbasicblock


(** [insert_block c name bb] creates a new basic block named [name] before the
    basic block [bb] in the context [c].
    See the constructor of [llvm::BasicBlock]. *)
val insert_block : llcontext -> string -> llbasicblock -> llbasicblock


(** [block_parent bb] returns the parent function that owns the basic block.
    See the method [llvm::BasicBlock::getParent]. *)
val block_parent : llbasicblock -> llvalue

(** [block_begin f] returns the first position in the basic block list of the
    function [f]. [block_begin] and [block_succ] can be used to iterate over
    the basic block list in order.
    See the method [llvm::Function::begin]. *)
val block_begin : llvalue -> (llvalue, llbasicblock) llpos


(** [block_succ bb] returns the basic block list position succeeding
    [Before bb].
    See the method [llvm::Function::iterator::operator++]. *)
val block_succ : llbasicblock -> (llvalue, llbasicblock) llpos


(** [iter_blocks f fn] applies function [f] to each of the basic blocks
    of function [fn] in order. Tail recursive. *)
val iter_blocks : (llbasicblock -> unit) -> llvalue -> unit

(** [fold_left_blocks f init fn] is [f (... (f init b1) ...) bN] where
    [b1,...,bN] are the basic blocks of function [fn]. Tail recursive. *)
val fold_left_blocks : ('a -> llbasicblock -> 'a) -> 'a -> llvalue -> 'a

(** [block_end f] returns the last position in the basic block list of
    the function [f]. [block_end] and [block_pred] can be used to iterate
    over the basic block list in reverse.
    See the method [llvm::Function::end]. *)
val block_end : llvalue -> (llvalue, llbasicblock) llrev_pos


(** [block_pred gv] returns the function list position preceding [After gv].
    See the method [llvm::Function::iterator::operator--]. *)
val block_pred : llbasicblock -> (llvalue, llbasicblock) llrev_pos

val block_terminator : llbasicblock -> llvalue option

(** [rev_iter_blocks f fn] applies function [f] to each of the basic blocks
    of function [fn] in reverse order. Tail recursive. *)
val rev_iter_blocks : (llbasicblock -> unit) -> llvalue -> unit

(** [fold_right_blocks f fn init] is [f (... (f init bN) ...) b1] where
    [b1,...,bN] are the basic blocks of function [fn]. Tail recursive. *)
val fold_right_blocks : (llbasicblock -> 'a -> 'a) -> llvalue -> 'a -> 'a

(** [value_of_block bb] losslessly casts [bb] to an [llvalue]. *)
val value_of_block : llbasicblock -> llvalue

(** [value_is_block v] returns [true] if the value [v] is a basic block and
    [false] otherwise.
    Similar to [llvm::isa<BasicBlock>]. *)
val value_is_block : llvalue -> bool

(** [block_of_value v] losslessly casts [v] to an [llbasicblock]. *)
val block_of_value : llvalue -> llbasicblock


(** {7 Operations on instructions} *)

(** [instr_parent i] is the enclosing basic block of the instruction [i].
    See the method [llvm::Instruction::getParent]. *)
val instr_parent : llvalue -> llbasicblock

(** [instr_begin bb] returns the first position in the instruction list of the
    basic block [bb]. [instr_begin] and [instr_succ] can be used to iterate over
    the instruction list in order.
    See the method [llvm::BasicBlock::begin]. *)
val instr_begin : llbasicblock -> (llbasicblock, llvalue) llpos


(** [instr_succ i] returns the instruction list position succeeding [Before i].
    See the method [llvm::BasicBlock::iterator::operator++]. *)
val instr_succ : llvalue -> (llbasicblock, llvalue) llpos


(** [iter_instrs f bb] applies function [f] to each of the instructions of basic
    block [bb] in order. Tail recursive. *)
val iter_instrs: (llvalue -> unit) -> llbasicblock -> unit

(** [fold_left_instrs f init bb] is [f (... (f init g1) ...) gN] where
    [g1,...,gN] are the instructions of basic block [bb]. Tail recursive. *)
val fold_left_instrs: ('a -> llvalue -> 'a) -> 'a -> llbasicblock -> 'a

(** [instr_end bb] returns the last position in the instruction list of the
    basic block [bb]. [instr_end] and [instr_pred] can be used to iterate over
    the instruction list in reverse.
    See the method [llvm::BasicBlock::end]. *)
val instr_end : llbasicblock -> (llbasicblock, llvalue) llrev_pos


(** [instr_pred i] returns the instruction list position preceding [After i].
    See the method [llvm::BasicBlock::iterator::operator--]. *)
val instr_pred : llvalue -> (llbasicblock, llvalue) llrev_pos


(** [fold_right_instrs f bb init] is [f (... (f init fN) ...) f1] where
    [f1,...,fN] are the instructions of basic block [bb]. Tail recursive. *)
val fold_right_instrs: (llvalue -> 'a -> 'a) -> llbasicblock -> 'a -> 'a

val instr_opcode : llvalue -> Opcode.t

val icmp_predicate : llvalue -> Icmp.t option

(** {7 Operations on call sites} *)

(** [instruction_call_conv ci] is the calling convention for the call or invoke
    instruction [ci], which may be one of the values from the module
    {!CallConv}. See the method [llvm::CallInst::getCallingConv] and
    [llvm::InvokeInst::getCallingConv]. *)
val instruction_call_conv: llvalue -> int


(** [set_instruction_call_conv cc ci] sets the calling convention for the call
    or invoke instruction [ci] to the integer [cc], which can be one of the
    values from the module {!CallConv}.
    See the method [llvm::CallInst::setCallingConv]
    and [llvm::InvokeInst::setCallingConv]. *)
val set_instruction_call_conv: int -> llvalue -> unit


(** [add_instruction_param_attr ci i a] adds attribute [a] to the [i]th
    parameter of the call or invoke instruction [ci]. [i]=0 denotes the return
    value. *)
val add_instruction_param_attr : llvalue -> int -> Attribute.t -> unit

(** [remove_instruction_param_attr ci i a] removes attribute [a] from the
    [i]th parameter of the call or invoke instruction [ci]. [i]=0 denotes the
    return value. *)
val remove_instruction_param_attr : llvalue -> int -> Attribute.t -> unit

(** {Operations on call instructions (only)} *)

(** [is_tail_call ci] is [true] if the call instruction [ci] is flagged as
    eligible for tail call optimization, [false] otherwise.
    See the method [llvm::CallInst::isTailCall]. *)
val is_tail_call : llvalue -> bool

(** [set_tail_call tc ci] flags the call instruction [ci] as eligible for tail
    call optimization if [tc] is [true], clears otherwise.
    See the method [llvm::CallInst::setTailCall]. *)
val set_tail_call : bool -> llvalue -> unit

(** {7 Operations on phi nodes} *)

(** [add_incoming (v, bb) pn] adds the value [v] to the phi node [pn] for use
    with branches from [bb]. See the method [llvm::PHINode::addIncoming]. *)
val add_incoming : (llvalue * llbasicblock) -> llvalue -> unit


(** [incoming pn] returns the list of value-block pairs for phi node [pn].
    See the method [llvm::PHINode::getIncomingValue]. *)
val incoming : llvalue -> (llvalue * llbasicblock) list

(** [delete_instruction i] deletes the instruction [i].
 * See the method [llvm::Instruction::eraseFromParent]. *)
val delete_instruction : llvalue -> unit

(** {6 Instruction builders} *)

(** [builder context] creates an instruction builder with no position in
    the context [context]. It is invalid to use this builder until its position
    is set with {!position_before} or {!position_at_end}. See the constructor
    for [llvm::LLVMBuilder]. *)
val builder : llcontext -> llbuilder

(** [builder_at ip] creates an instruction builder positioned at [ip].
    See the constructor for [llvm::LLVMBuilder]. *)
val builder_at : llcontext -> (llbasicblock, llvalue) llpos -> llbuilder

(** [builder_before ins] creates an instruction builder positioned before the
    instruction [isn]. See the constructor for [llvm::LLVMBuilder]. *)
val builder_before : llcontext -> llvalue -> llbuilder

(** [builder_at_end bb] creates an instruction builder positioned at the end of
    the basic block [bb]. See the constructor for [llvm::LLVMBuilder]. *)
val builder_at_end : llcontext -> llbasicblock -> llbuilder

(** [position_builder ip bb] moves the instruction builder [bb] to the position
    [ip].
    See the constructor for [llvm::LLVMBuilder]. *)
val position_builder : (llbasicblock, llvalue) llpos -> llbuilder -> unit


(** [position_before ins b] moves the instruction builder [b] to before the
    instruction [isn]. See the method [llvm::LLVMBuilder::SetInsertPoint]. *)
val position_before : llvalue -> llbuilder -> unit

(** [position_at_end bb b] moves the instruction builder [b] to the end of the
    basic block [bb]. See the method [llvm::LLVMBuilder::SetInsertPoint]. *)
val position_at_end : llbasicblock -> llbuilder -> unit

(** [insertion_block b] returns the basic block that the builder [b] is
    positioned to insert into. Raises [Not_Found] if the instruction builder is
    uninitialized.
    See the method [llvm::LLVMBuilder::GetInsertBlock]. *)
val insertion_block : llbuilder -> llbasicblock

(** [insert_into_builder i name b] inserts the specified instruction [i] at the
    position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::Insert]. *)
val insert_into_builder : llvalue -> string -> llbuilder -> unit


(** {7 Metadata} *)

(** [set_current_debug_location b md] sets the current debug location [md] in
    the builder [b].
    See the method [llvm::IRBuilder::SetDebugLocation]. *)
val set_current_debug_location : llbuilder -> llvalue -> unit


(** [clear_current_debug_location b] clears the current debug location in the
    builder [b]. *)
val clear_current_debug_location : llbuilder -> unit


(** [current_debug_location b] returns the current debug location, or None
    if none is currently set.
    See the method [llvm::IRBuilder::GetDebugLocation]. *)
val current_debug_location : llbuilder -> llvalue option


(** [set_inst_debug_location b i] sets the current debug location of the builder
    [b] to the instruction [i].
    See the method [llvm::IRBuilder::SetInstDebugLocation]. *)
val set_inst_debug_location : llbuilder -> llvalue -> unit


(** {7 Terminators} *)

(** [build_ret_void b] creates a
    [ret void]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateRetVoid]. *)
val build_ret_void : llbuilder -> llvalue

(** [build_ret v b] creates a
    [ret %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateRet]. *)
val build_ret : llvalue -> llbuilder -> llvalue

(** [build_aggregate_ret vs b] creates a
    [ret {...} { %v1, %v2, ... } ]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAggregateRet]. *)
val build_aggregate_ret : llvalue array -> llbuilder -> llvalue


(** [build_br bb b] creates a
    [br %bb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateBr]. *)
val build_br : llbasicblock -> llbuilder -> llvalue

(** [build_cond_br cond tbb fbb b] creates a
    [br %cond, %tbb, %fbb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateCondBr]. *)
val build_cond_br : llvalue -> llbasicblock -> llbasicblock -> llbuilder ->
                         llvalue

(** [build_switch case elsebb count b] creates an empty
    [switch %case, %elsebb]
    instruction at the position specified by the instruction builder [b] with
    space reserved for [count] cases.
    See the method [llvm::LLVMBuilder::CreateSwitch]. *)
val build_switch : llvalue -> llbasicblock -> int -> llbuilder -> llvalue

(** [build_malloc ty name b] creates an [malloc]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::CallInst::CreateMalloc]. *)
val build_malloc : lltype -> string -> llbuilder -> llvalue

(** [build_array_malloc ty val name b] creates an [array malloc]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::CallInst::CreateArrayMalloc]. *)
val build_array_malloc : lltype -> llvalue -> string -> llbuilder -> llvalue

(** [build_free p b] creates a [free]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFree]. *)
val build_free : llvalue -> llbuilder -> llvalue

(** [add_case sw onval bb] causes switch instruction [sw] to branch to [bb]
    when its input matches the constant [onval].
    See the method [llvm::SwitchInst::addCase]. **)
val add_case : llvalue -> llvalue -> llbasicblock -> unit

(** [switch_default_dest sw] returns the default destination of the [switch]
 * instruction.
 * See the method [llvm:;SwitchInst::getDefaultDest]. **)
val switch_default_dest : llvalue -> llbasicblock

(** [build_indirect_br addr count b] creates a
    [indirectbr %addr]
    instruction at the position specified by the instruction builder [b] with
    space reserved for [count] destinations.
    See the method [llvm::LLVMBuilder::CreateIndirectBr]. *)
val build_indirect_br : llvalue -> int -> llbuilder -> llvalue


(** [add_destination br bb] adds the basic block [bb] as a possible branch
    location for the indirectbr instruction [br].
    See the method [llvm::IndirectBrInst::addDestination]. **)
val add_destination : llvalue -> llbasicblock -> unit


(** [build_invoke fn args tobb unwindbb name b] creates an
    [%name = invoke %fn(args) to %tobb unwind %unwindbb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInvoke]. *)
val build_invoke : llvalue -> llvalue array -> llbasicblock ->
                        llbasicblock -> string -> llbuilder -> llvalue

(** [build_landingpad ty persfn numclauses name b] creates an
    [landingpad]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLandingPad]. *)
val build_landingpad : lltype -> llvalue -> int -> string -> llbuilder ->
                         llvalue

(** [set_cleanup lp] sets the cleanup flag in the [landingpad]instruction.
    See the method [llvm::LandingPadInst::setCleanup]. *)
val set_cleanup : llvalue -> bool -> unit

(** [add_clause lp clause] adds the clause to the [landingpad]instruction.
    See the method [llvm::LandingPadInst::addClause]. *)
val add_clause : llvalue -> llvalue -> unit

(* [build_resume exn b] builds a [resume exn] instruction
 * at the position specified by the instruction builder [b].
 * See the method [llvm::LLVMBuilder::CreateResume] *)
val build_resume : llvalue -> llbuilder -> llvalue

(** [build_unreachable b] creates an
    [unreachable]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUnwind]. *)
val build_unreachable : llbuilder -> llvalue


(** {7 Arithmetic} *)

(** [build_add x y name b] creates a
    [%name = add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAdd]. *)
val build_add : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_nsw_add x y name b] creates a
    [%name = nsw add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNSWAdd]. *)
val build_nsw_add : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_nuw_add x y name b] creates a
    [%name = nuw add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNUWAdd]. *)
val build_nuw_add : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_fadd x y name b] creates a
    [%name = fadd %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFAdd]. *)
val build_fadd : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_sub x y name b] creates a
    [%name = sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSub]. *)
val build_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_nsw_sub x y name b] creates a
    [%name = nsw sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNSWSub]. *)
val build_nsw_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_nuw_sub x y name b] creates a
    [%name = nuw sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNUWSub]. *)
val build_nuw_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_fsub x y name b] creates a
    [%name = fsub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFSub]. *)
val build_fsub : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_mul x y name b] creates a
    [%name = mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateMul]. *)
val build_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_nsw_mul x y name b] creates a
    [%name = nsw mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNSWMul]. *)
val build_nsw_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_nuw_mul x y name b] creates a
    [%name = nuw mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNUWMul]. *)
val build_nuw_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_fmul x y name b] creates a
    [%name = fmul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFMul]. *)
val build_fmul : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_udiv x y name b] creates a
    [%name = udiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUDiv]. *)
val build_udiv : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_sdiv x y name b] creates a
    [%name = sdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSDiv]. *)
val build_sdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_exact_sdiv x y name b] creates a
    [%name = exact sdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExactSDiv]. *)
val build_exact_sdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_fdiv x y name b] creates a
    [%name = fdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFDiv]. *)
val build_fdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_urem x y name b] creates a
    [%name = urem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateURem]. *)
val build_urem : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_SRem x y name b] creates a
    [%name = srem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSRem]. *)
val build_srem : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_frem x y name b] creates a
    [%name = frem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFRem]. *)
val build_frem : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_shl x y name b] creates a
    [%name = shl %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateShl]. *)
val build_shl : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_lshr x y name b] creates a
    [%name = lshr %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLShr]. *)
val build_lshr : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_ashr x y name b] creates a
    [%name = ashr %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAShr]. *)
val build_ashr : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_and x y name b] creates a
    [%name = and %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAnd]. *)
val build_and : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_or x y name b] creates a
    [%name = or %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateOr]. *)
val build_or : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_xor x y name b] creates a
    [%name = xor %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateXor]. *)
val build_xor : llvalue -> llvalue -> string -> llbuilder -> llvalue


(** [build_neg x name b] creates a
    [%name = sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. *)
val build_neg : llvalue -> string -> llbuilder -> llvalue


(** [build_nsw_neg x name b] creates a
    [%name = nsw sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. *)
val build_nsw_neg : llvalue -> string -> llbuilder -> llvalue


(** [build_nuw_neg x name b] creates a
    [%name = nuw sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. *)
val build_nuw_neg : llvalue -> string -> llbuilder -> llvalue


(** [build_fneg x name b] creates a
    [%name = fsub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateFNeg]. *)
val build_fneg : llvalue -> string -> llbuilder -> llvalue


(** [build_xor x name b] creates a
    [%name = xor %x, -1]
    instruction at the position specified by the instruction builder [b].
    [-1] is the correct "all ones" value for the type of [x].
    See the method [llvm::LLVMBuilder::CreateXor]. *)
val build_not : llvalue -> string -> llbuilder -> llvalue



(** {7 Memory} *)

(** [build_alloca ty name b] creates a
    [%name = alloca %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. *)
val build_alloca : lltype -> string -> llbuilder -> llvalue


(** [build_array_alloca ty n name b] creates a
    [%name = alloca %ty, %n]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. *)
val build_array_alloca : lltype -> llvalue -> string -> llbuilder ->
                              llvalue

(** [build_load v name b] creates a
    [%name = load %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLoad]. *)
val build_load : llvalue -> string -> llbuilder -> llvalue


(** [build_store v p b] creates a
    [store %v, %p]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateStore]. *)
val build_store : llvalue -> llvalue -> llbuilder -> llvalue


(** [build_gep p indices name b] creates a
    [%name = getelementptr %p, indices...]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateGetElementPtr]. *)
val build_gep : llvalue -> llvalue array -> string -> llbuilder -> llvalue


(** [build_in_bounds_gep p indices name b] creates a
    [%name = gelementptr inbounds %p, indices...]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInBoundsGetElementPtr]. *)
val build_in_bounds_gep : llvalue -> llvalue array -> string -> llbuilder ->
                               llvalue

(** [build_struct_gep p idx name b] creates a
    [%name = getelementptr %p, 0, idx]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateStructGetElementPtr]. *)
val build_struct_gep : llvalue -> int -> string -> llbuilder ->
                            llvalue

(** [build_global_string str name b] creates a series of instructions that adds
    a global string at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateGlobalString]. *)
val build_global_string : string -> string -> llbuilder -> llvalue


(** [build_global_stringptr str name b] creates a series of instructions that
    adds a global string pointer at the position specified by the instruction
    builder [b].
    See the method [llvm::LLVMBuilder::CreateGlobalStringPtr]. *)
val build_global_stringptr : string -> string -> llbuilder -> llvalue



(** {7 Casts} *)

(** [build_trunc v ty name b] creates a
    [%name = trunc %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateTrunc]. *)
val build_trunc : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_zext v ty name b] creates a
    [%name = zext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExt]. *)
val build_zext : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_sext v ty name b] creates a
    [%name = sext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSExt]. *)
val build_sext : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_fptoui v ty name b] creates a
    [%name = fptoui %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPToUI]. *)
val build_fptoui : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_fptosi v ty name b] creates a
    [%name = fptosi %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPToSI]. *)
val build_fptosi : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_uitofp v ty name b] creates a
    [%name = uitofp %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUIToFP]. *)
val build_uitofp : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_sitofp v ty name b] creates a
    [%name = sitofp %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSIToFP]. *)
val build_sitofp : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_fptrunc v ty name b] creates a
    [%name = fptrunc %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPTrunc]. *)
val build_fptrunc : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_fpext v ty name b] creates a
    [%name = fpext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPExt]. *)
val build_fpext : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_ptrtoint v ty name b] creates a
    [%name = prtotint %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePtrToInt]. *)
val build_ptrtoint : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_inttoptr v ty name b] creates a
    [%name = inttoptr %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIntToPtr]. *)
val build_inttoptr : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_bitcast v ty name b] creates a
    [%name = bitcast %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateBitCast]. *)
val build_bitcast : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_zext_or_bitcast v ty name b] creates a zext or bitcast
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExtOrBitCast]. *)
val build_zext_or_bitcast : llvalue -> lltype -> string -> llbuilder ->
                                 llvalue

(** [build_sext_or_bitcast v ty name b] creates a sext or bitcast
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSExtOrBitCast]. *)
val build_sext_or_bitcast : llvalue -> lltype -> string -> llbuilder ->
                                 llvalue

(** [build_trunc_or_bitcast v ty name b] creates a trunc or bitcast
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExtOrBitCast]. *)
val build_trunc_or_bitcast : llvalue -> lltype -> string -> llbuilder ->
                                  llvalue

(** [build_pointercast v ty name b] creates a bitcast or pointer-to-int
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePointerCast]. *)
val build_pointercast : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_intcast v ty name b] creates a zext, bitcast, or trunc
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIntCast]. *)
val build_intcast : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_fpcast v ty name b] creates a fpext, bitcast, or fptrunc
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPCast]. *)
val build_fpcast : llvalue -> lltype -> string -> llbuilder -> llvalue



(** {7 Comparisons} *)

(** [build_icmp pred x y name b] creates a
    [%name = icmp %pred %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateICmp]. *)
val build_icmp : Icmp.t -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue

(** [build_fcmp pred x y name b] creates a
    [%name = fcmp %pred %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFCmp]. *)
val build_fcmp : Fcmp.t -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue


(** {7 Miscellaneous instructions} *)

(** [build_phi incoming name b] creates a
    [%name = phi %incoming]
    instruction at the position specified by the instruction builder [b].
    [incoming] is a list of [(llvalue, llbasicblock)] tuples.
    See the method [llvm::LLVMBuilder::CreatePHI]. *)
val build_phi : (llvalue * llbasicblock) list -> string -> llbuilder ->
                     llvalue

(** [build_call fn args name b] creates a
    [%name = call %fn(args...)]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateCall]. *)
val build_call : llvalue -> llvalue array -> string -> llbuilder -> llvalue


(** [build_select cond thenv elsev name b] creates a
    [%name = select %cond, %thenv, %elsev]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSelect]. *)
val build_select : llvalue -> llvalue -> llvalue -> string -> llbuilder ->
                        llvalue

(** [build_va_arg valist argty name b] creates a
    [%name = va_arg %valist, %argty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateVAArg]. *)
val build_va_arg : llvalue -> lltype -> string -> llbuilder -> llvalue


(** [build_extractelement vec i name b] creates a
    [%name = extractelement %vec, %i]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExtractElement]. *)
val build_extractelement : llvalue -> llvalue -> string -> llbuilder ->
                                llvalue

(** [build_insertelement vec elt i name b] creates a
    [%name = insertelement %vec, %elt, %i]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInsertElement]. *)
val build_insertelement : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue

(** [build_shufflevector veca vecb mask name b] creates a
    [%name = shufflevector %veca, %vecb, %mask]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateShuffleVector]. *)
val build_shufflevector : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue

(** [build_insertvalue agg idx name b] creates a
    [%name = extractvalue %agg, %idx]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExtractValue]. *)
val build_extractvalue : llvalue -> int -> string -> llbuilder -> llvalue


(** [build_insertvalue agg val idx name b] creates a
    [%name = insertvalue %agg, %val, %idx]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInsertValue]. *)
val build_insertvalue : llvalue -> llvalue -> int -> string -> llbuilder ->
                             llvalue

(** [build_is_null val name b] creates a
    [%name = icmp eq %val, null]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIsNull]. *)
val build_is_null : llvalue -> string -> llbuilder -> llvalue


(** [build_is_not_null val name b] creates a
    [%name = icmp ne %val, null]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIsNotNull]. *)
val build_is_not_null : llvalue -> string -> llbuilder -> llvalue


(** [build_ptrdiff lhs rhs name b] creates a series of instructions that measure
    the difference between two pointer values at the position specified by the
    instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePtrDiff]. *)
val build_ptrdiff : llvalue -> llvalue -> string -> llbuilder -> llvalue



(** {6 Memory buffers} *)

module MemoryBuffer : sig
  (** [of_file p] is the memory buffer containing the contents of the file at
      path [p]. If the file could not be read, then [IoError msg] is
      raised. *)
  val of_file : string -> llmemorybuffer
  
  (** [stdin ()] is the memory buffer containing the contents of standard input.
      If standard input is empty, then [IoError msg] is raised. *)
  val of_stdin : unit -> llmemorybuffer
  
  (** Disposes of a memory buffer. *)
  val dispose : llmemorybuffer -> unit
end


(** {6 Pass Managers} *)

module PassManager : sig
  (**  *)
  type 'a t
  type any = [ `Module | `Function ]
  
  (** [PassManager.create ()] constructs a new whole-module pass pipeline. This
      type of pipeline is suitable for link-time optimization and whole-module
      transformations.
      See the constructor of [llvm::PassManager]. *)
  val create : unit -> [ `Module ] t
  
  (** [PassManager.create_function m] constructs a new function-by-function
      pass pipeline over the module [m]. It does not take ownership of [m].
      This type of pipeline is suitable for code generation and JIT compilation
      tasks.
      See the constructor of [llvm::FunctionPassManager]. *)
  val create_function : llmodule -> [ `Function ] t

  
  (** [run_module m pm] initializes, executes on the module [m], and finalizes
      all of the passes scheduled in the pass manager [pm]. Returns [true] if
      any of the passes modified the module, [false] otherwise.
      See the [llvm::PassManager::run] method. *)
  val run_module : llmodule -> [ `Module ] t -> bool

  
  (** [initialize fpm] initializes all of the function passes scheduled in the
      function pass manager [fpm]. Returns [true] if any of the passes modified
      the module, [false] otherwise.
      See the [llvm::FunctionPassManager::doInitialization] method. *)
  val initialize : [ `Function ] t -> bool
  
  (** [run_function f fpm] executes all of the function passes scheduled in the
      function pass manager [fpm] over the function [f]. Returns [true] if any
      of the passes modified [f], [false] otherwise.
      See the [llvm::FunctionPassManager::run] method. *)
  val run_function : llvalue -> [ `Function ] t -> bool

  
  (** [finalize fpm] finalizes all of the function passes scheduled in in the
      function pass manager [fpm]. Returns [true] if any of the passes
      modified the module, [false] otherwise.
      See the [llvm::FunctionPassManager::doFinalization] method. *)
  val finalize : [ `Function ] t -> bool
  
  (** Frees the memory of a pass pipeline. For function pipelines, does not free
      the module.
      See the destructor of [llvm::BasePassManager]. *)
  val dispose : [< any ] t -> unit
end
