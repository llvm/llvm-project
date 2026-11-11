//===-- llvm/RuntimeLibcalls.td - File that describes libcalls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/IR/Attributes.td"
include "llvm/TableGen/SetTheory.td"

// Dag operators for composing predicates.
def all_of;
def any_of;
def not;

// Composable predicate leaf expressed as a C++ boolean over an
// `Triple TT`. Generally should check a single triple property.
class LibcallPredicate<code cond> {
  code Cond = cond;
}

// Predicate for whether a libcall exists for the target ABI. This is
// a module level property that should only be computed based on the
// triple.
//
// The condition is expressed as a `dag` of LibcallPredicates combined
// with the all_of/any_of/not operators, which the emitter lowers to
// the a boolean function of a Triple named TT (and other constructor
// arguments of RuntimeLibcallsInfo).
class RuntimeLibcallAvailability<dag cond = (?)> {
  // Composable form: (all_of/any_of/not <LibcallPredicate>...). The `(?)`
  // placeholder default means the libcall is always available.
  dag CondDag = cond;
}

def AlwaysAvailable : RuntimeLibcallAvailability;

class LibcallCallingConv<code CC = [{}]> {
  // Enum value for the calling convention. Empty string defaults to
  // CallingConv::C.
  code CallingConv = CC;
}

def DefaultCC : LibcallCallingConv;
def FASTCC : LibcallCallingConv<[{CallingConv::Fast}]>;
def MSP430_BUILTIN : LibcallCallingConv<[{CallingConv::MSP430_BUILTIN}]>;
def ARM_AAPCS : LibcallCallingConv<[{CallingConv::ARM_AAPCS}]>;
def ARM_AAPCS_VFP : LibcallCallingConv<[{CallingConv::ARM_AAPCS_VFP}]>;
def X86_STDCALL : LibcallCallingConv<[{CallingConv::X86_StdCall}]>;
def AVR_BUILTIN : LibcallCallingConv<[{CallingConv::AVR_BUILTIN}]>;
def SMEABI_PreserveMost_From_X0 : LibcallCallingConv<[{CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0}]>;
def SMEABI_PreserveMost_From_X1 : LibcallCallingConv<[{CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1}]>;
def SMEABI_PreserveMost_From_X2 : LibcallCallingConv<[{CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2}]>;
def AARCH64_VECTORCALL : LibcallCallingConv<[{CallingConv::AArch64_VectorCall}]>;


// Return type or argument type of library calls.
class FuncArgType {
  string Name = NAME;
}

// Recognized types of library call arguments and return types.
def NoneType : FuncArgType; // dummy
def Void : FuncArgType;
def Bool : FuncArgType;     // 8 bits on all targets
def Int8 : FuncArgType;
def Int16 : FuncArgType;
def Int32 : FuncArgType;
def Int : FuncArgType;
def IntPlus : FuncArgType;  // Int or bigger.
def Long : FuncArgType;     // Either 32 or 64 bits.
def IntX : FuncArgType;     // Any integer type.
def Int64 : FuncArgType;
def Int128 : FuncArgType;
def LLong : FuncArgType;    // 64 bits on all targets.
def UInt8 : FuncArgType;
def UInt16 : FuncArgType;
def UInt32 : FuncArgType;
def UInt : FuncArgType;
def UIntPlus : FuncArgType;
def ULong : FuncArgType;
def UIntX : FuncArgType;
def UInt64 : FuncArgType;
def ULLong : FuncArgType;
def UInt128 : FuncArgType;
def SizeT : FuncArgType;    // size_t.
def SSizeT : FuncArgType;   // POSIX ssize_t.
defvar AlignValT = SizeT;  // std::align_val_t.
def CmpResult : FuncArgType; // libgcc_cmp_return.
def Half : FuncArgType;     // IEEE half.
def BFloat : FuncArgType;   // bfloat16.
def Flt : FuncArgType;      // IEEE float.
def Dbl : FuncArgType;      // IEEE double.
def LDbl : FuncArgType;     // Any floating type (TODO: tighten this up).
def X86FP80 : FuncArgType;
def FP128 : FuncArgType;
def PPCFP128 : FuncArgType;
def Floating : FuncArgType; // Any floating type.
def Ptr : FuncArgType;      // Any pointer type.
def AllocaPtr : FuncArgType; // Pointer in the alloca address space.
def Struct : FuncArgType;   // Any struct type.
def Ellip : FuncArgType;    // The ellipsis (...).
def Same : FuncArgType;     // Same argument type as the previous one.
def NXV2I1 : FuncArgType;
def NXV4I1 : FuncArgType;
def V2I32 : FuncArgType;
def V4I32 : FuncArgType;
def V8I32 : FuncArgType;
def V16I32 : FuncArgType;
def NXV2I32 : FuncArgType;
def NXV4I32 : FuncArgType;

// Signature markers for libcalls that are global variables rather than
// functions. The variable's type follows the marker.
def Variable : FuncArgType;
def ThreadLocalVariable : FuncArgType;

class StructFuncArgType<list<FuncArgType> elementTypes> : FuncArgType {
  list<FuncArgType> ElementTypes = elementTypes;
  bit InReg = false;
}

class VectorFuncArgType<FuncArgType elementType, int numElements,
                        bit scalable = false> : FuncArgType {
  FuncArgType ElementType = elementType;
  int NumElements = numElements;
  bit Scalable = scalable;

  defvar Prefix = !if(scalable, "NXV", "V") # numElements;
  FuncArgType IntType = !cast<FuncArgType>(Prefix # "I32");
  FuncArgType MaskType = !if(scalable, !cast<FuncArgType>(Prefix # "I1"), ?);
}

def V2F32 : VectorFuncArgType<Flt, 2>;
def V4F32 : VectorFuncArgType<Flt, 4>;
def V8F32 : VectorFuncArgType<Flt, 8>;
def V16F32 : VectorFuncArgType<Flt, 16>;
def NXV4F32 : VectorFuncArgType<Flt, 4, scalable = true>;
def V2F64 : VectorFuncArgType<Dbl, 2>;
def V4F64 : VectorFuncArgType<Dbl, 4>;
def V8F64 : VectorFuncArgType<Dbl, 8>;
def NXV2F64 : VectorFuncArgType<Dbl, 2, scalable = true>;

class ModRefInfo;
def NoModRef : ModRefInfo;
def Ref : ModRefInfo;
def Mod : ModRefInfo;
def ModRef : ModRefInfo;

class MemoryEffects<ModRefInfo mr = ModRef> {
  ModRefInfo ArgMem = mr;
  ModRefInfo InaccessibleMem = mr;
  ModRefInfo ErrnoMem = mr;
  ModRefInfo Other = mr;
  ModRefInfo TargetMem0 = mr;
  ModRefInfo TargetMem1 = mr;
}

def UnknownMemory : MemoryEffects;
def ArgMemWriteOnly : MemoryEffects<NoModRef> { let ArgMem = Mod; }

class ValueAttr<list<AttrProperty> props, string expr>
    : Attr<"", IntersectPreserve, props> {
  string Expr = expr;
}

class AllocKindAttr<string kind>
    : ValueAttr<[FnAttr], "Attribute::getWithAllocKind(C, " # kind # ")">;

class AllocSizeAttr<int elemSizeArg, string numElemsArg = "std::nullopt">
    : ValueAttr<[FnAttr], "Attribute::getWithAllocSizeArgs(C, " # elemSizeArg #
                          ", " # numElemsArg # ")">;

class AllocFamilyAttr<string family>
    : ValueAttr<[FnAttr],
                "Attribute::get(C, \"alloc-family\", \"" # family # "\")">;

def CapturesNone
    : ValueAttr<[ParamAttr],
                "Attribute::getWithCaptureInfo(C, CaptureInfo::none())">;

/// Abstract definition for functionality the compiler may need to
/// emit a call to. Emits the RTLIB::Libcall enum - This enum defines
/// all of the runtime library calls the backend can emit. The various
/// long double types cannot be merged, because 80-bit library
/// functions use "xf" and 128-bit use "tf".
///
/// When adding PPCF128 functions here, note that their names generally need
/// to be overridden for Darwin with the xxx$LDBL128 form.  See
/// PPCISelLowering.cpp.
class RuntimeLibcall<FuncArgType RetType = NoneType,
                     list<FuncArgType> ArgTypes = []> {
  string Name = NAME;
  FuncArgType ReturnType = RetType;
  list<FuncArgType> ArgumentTypes = ArgTypes;

  list<Attr> FnAttrs = [];
  MemoryEffects Memory = UnknownMemory;
  list<Attr> RetAttrs = [];
  list<list<Attr>> ParamAttrs = [];
  list<FuncArgType> ParamPointeeTypes = [];
}

class RuntimeLibcallVariable<FuncArgType Ty, bit threadLocal = false>
    : RuntimeLibcall<!if(threadLocal, ThreadLocalVariable, Variable), [Ty]>;

class MaskedRuntimeLibcall<RuntimeLibcall unmasked, FuncArgType mask>
    : RuntimeLibcall<unmasked.ReturnType,
                     !if(!empty(unmasked.ArgumentTypes), [],
                         !listconcat(unmasked.ArgumentTypes, [mask]))> {
  RuntimeLibcall Unmasked = unmasked;
  let FnAttrs = unmasked.FnAttrs;
  let Memory = unmasked.Memory;
  let RetAttrs = unmasked.RetAttrs;
  let ParamAttrs = unmasked.ParamAttrs;
  let ParamPointeeTypes = unmasked.ParamPointeeTypes;
}

// TODO: NoCallback probably unsafe in general
defvar VectorMathFnAttrs = [NoCallback, NoFree, NoSync, NoUnwind, WillReturn];

multiclass VectorMathLibcall<string suffix, FuncArgType RetType = NoneType,
                             list<FuncArgType> ArgTypes = [],
                             MemoryEffects memory = UnknownMemory,
                             list<list<Attr>> paramAttrs = [],
                             list<FuncArgType> paramPointeeTypes = []> {
  defvar VT = !cast<VectorFuncArgType>(suffix);
  def _#suffix : RuntimeLibcall<RetType, ArgTypes> {
    let FnAttrs = VectorMathFnAttrs;
    let Memory = memory;
    let ParamAttrs = paramAttrs;
    let ParamPointeeTypes = paramPointeeTypes;
  }
  if !initialized(VT.MaskType) then
    def _MASKED_#suffix
        : MaskedRuntimeLibcall<!cast<RuntimeLibcall>(NAME#"_"#suffix),
                               VT.MaskType>;
}

// Concrete implementation of a libcall, which may have a different
// name and only be valid on some subset of targets.

class RuntimeLibcallImpl<RuntimeLibcall P, string Name = NAME> {
  RuntimeLibcall Provides = P;
  string LibCallFuncName = Name;
  FuncArgType ReturnType = P.ReturnType;
  list<FuncArgType> ArgumentTypes = P.ArgumentTypes;
  list<Attr> FnAttrs = P.FnAttrs;
  MemoryEffects Memory = P.Memory;
  list<Attr> RetAttrs = P.RetAttrs;
  list<list<Attr>> ParamAttrs = P.ParamAttrs;
  list<FuncArgType> ParamPointeeTypes = P.ParamPointeeTypes;
  bit AlignPointeesToElementType = true;
}

// Implementation of a long double function, which provides the libcall
// matching the module's long double format.
class LongDoubleLibcallImpl<string libcall_basename, string Name = NAME>
    : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_F128"),
                         Name> {
  let Provides = ?;
  RuntimeLibcall ProvidesF80 = !cast<RuntimeLibcall>(libcall_basename#"_F80");
  RuntimeLibcall ProvidesF128 = !cast<RuntimeLibcall>(libcall_basename#"_F128");
  RuntimeLibcall ProvidesPPCF128 =
      !cast<RuntimeLibcall>(libcall_basename#"_PPCF128");

  let ReturnType = !if(!eq(ProvidesF128.ReturnType, FP128), LDbl,
                       ProvidesF128.ReturnType);
  let ArgumentTypes = !foreach(Ty, ProvidesF128.ArgumentTypes,
                               !if(!eq(Ty, FP128), LDbl, Ty));
  let ParamPointeeTypes = !foreach(Ty, ProvidesF128.ParamPointeeTypes,
                                   !if(!eq(Ty, FP128), LDbl, Ty));
}

/// A family of runtime libcalls implementing one floating-point operation, used
/// to generate both the backend selector RTLIB::get<base>(EVT) and the IR-level
/// getLibcallForIntrinsic() mapping. The libcall for a type is `<base>_<suffix>`
/// (e.g. SIN_F64).
class RuntimeLibcallFamily<string base, list<string> intrinsics = [],
                           list<string> vectorSuffixes = []> {
  string LibcallBase = base;
  list<string> Intrinsics = intrinsics;
  list<string> VectorSuffixes = vectorSuffixes;
}

class LibcallImpls<dag funcList,
                   RuntimeLibcallAvailability Pred = AlwaysAvailable> {
  // Function of the triple where this applies
  RuntimeLibcallAvailability AvailabilityPredicate = Pred;
  dag MemberList = funcList;

  // If set, applies calling convention to every entry instead of
  // taking the system default.
  LibcallCallingConv CallingConv = ?;
}

/// Convenience wrapper around LibcallImplSet to make a single libcall
/// implementation conditionally conditionally available.
class AvailableIf<RuntimeLibcallImpl Impl, RuntimeLibcallAvailability Pred>
    : LibcallImpls<(add Impl), Pred>;

class LibcallsWithCC<dag funcList, LibcallCallingConv CC,
                     RuntimeLibcallAvailability Pred = AlwaysAvailable>
    : LibcallImpls<funcList, Pred> {
  LibcallCallingConv CallingConv = CC;
}

// A named runtime library whose Impls are made available as a unit. Impls is a
// set of RuntimeLibcallImpls, optionally grouped with LibcallImpls.
class LibcallLibrary<string name, dag impls,
                     RuntimeLibcallAvailability pred = AlwaysAvailable> {
  string LibraryName = name;
  dag Impls = impls;

  // Availability guard for the library. Libraries sharing a LibraryName merge
  // into one function, each variant under its own Pred.
  RuntimeLibcallAvailability Pred = pred;

  // When set, emit this library as its own setup function instead of merging it
  // with others sharing its LibraryName, so a consumer gets exactly these members
  // rather than the merged core. For a target providing a strict subset or a
  // name-decorated variant of a library. LibraryName stays the
  // isLibraryAvailable() key.
  bit Isolated = false;
}

// Drop the impls in `exclude` from the library named `name` on triples matching
// `pred`, so a target can use a provider (e.g. compiler-rt) but keep its own
// versions of a few entries.
class LibraryExclusion<string name, RuntimeLibcallAvailability pred,
                       list<RuntimeLibcallImpl> exclude> {
  string LibraryName = name;
  RuntimeLibcallAvailability Pred = pred;
  list<RuntimeLibcallImpl> Exclude = exclude;
}

/// Define the default runtime libraries for a target, by LibraryName.
class SystemRuntimeLibrary<RuntimeLibcallAvailability Pred,
                           list<string> libraries> {
  /// Set the default calling convention assumed for library members.
  LibcallCallingConv DefaultLibcallCallingConv = DefaultCC;
  RuntimeLibcallAvailability TriplePred = Pred;
  list<string> DefaultLibraries = libraries;
}
