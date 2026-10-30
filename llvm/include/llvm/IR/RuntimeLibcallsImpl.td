//===-- llvm/RuntimeLibcalls.td - File that describes libcalls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
def UInt16 : FuncArgType;
def UInt32 : FuncArgType;
def UInt : FuncArgType;
def UIntPlus : FuncArgType;
def ULong : FuncArgType;
def UIntX : FuncArgType;
def UInt64 : FuncArgType;
def ULLong : FuncArgType;
def SizeT : FuncArgType;    // size_t.
def SSizeT : FuncArgType;   // POSIX ssize_t.
def Flt : FuncArgType;      // IEEE float.
def Dbl : FuncArgType;      // IEEE double.
def LDbl : FuncArgType;     // Any floating type (TODO: tighten this up).
def Floating : FuncArgType; // Any floating type.
def Ptr : FuncArgType;      // Any pointer type.
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

class VectorFuncArgType<FuncArgType intType> : FuncArgType {
  FuncArgType IntType = intType;
  FuncArgType MaskType = ?;
}

def V2F32 : VectorFuncArgType<V2I32>;
def V4F32 : VectorFuncArgType<V4I32>;
def V8F32 : VectorFuncArgType<V8I32>;
def V16F32 : VectorFuncArgType<V16I32>;
def NXV4F32 : VectorFuncArgType<NXV4I32> { let MaskType = NXV4I1; }
def V2F64 : VectorFuncArgType<V2I32>;
def V4F64 : VectorFuncArgType<V4I32>;
def V8F64 : VectorFuncArgType<V8I32>;
def NXV2F64 : VectorFuncArgType<NXV2I32> { let MaskType = NXV2I1; }

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

  // TODO: Record attributes
}

class MaskedRuntimeLibcall<RuntimeLibcall unmasked, FuncArgType mask>
    : RuntimeLibcall<unmasked.ReturnType,
                     !if(!empty(unmasked.ArgumentTypes), [],
                         !listconcat(unmasked.ArgumentTypes, [mask]))> {
  RuntimeLibcall Unmasked = unmasked;
}

multiclass VectorMathLibcall<string suffix, FuncArgType RetType = NoneType,
                             list<FuncArgType> ArgTypes = []> {
  defvar VT = !cast<VectorFuncArgType>(suffix);
  def _#suffix : RuntimeLibcall<RetType, ArgTypes>;
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
