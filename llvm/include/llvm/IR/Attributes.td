//===- Attributes.td - Defines all LLVM attributes ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all the LLVM attributes.
//
//===----------------------------------------------------------------------===//

/// Attribute property base class.
class AttrProperty;

/// Can be used as function attribute.
def FnAttr : AttrProperty;

/// Can be used as parameter attribute.
def ParamAttr : AttrProperty;

/// Can be used as return attribute.
def RetAttr : AttrProperty;

/// Attribute base class.
class Attr<string S, list<AttrProperty> P> {
  // String representation of this attribute in the IR.
  string AttrString = S;
  list<AttrProperty> Properties = P;
}

/// Enum attribute.
class EnumAttr<string S, list<AttrProperty> P> : Attr<S, P>;

/// Int attribute.
class IntAttr<string S, list<AttrProperty> P> : Attr<S, P>;

/// Type attribute.
class TypeAttr<string S, list<AttrProperty> P> : Attr<S, P>;

/// StringBool attribute.
class StrBoolAttr<string S> : Attr<S, []>;

/// Target-independent enum attributes.

/// Alignment of parameter (5 bits) stored as log2 of alignment with +1 bias.
/// 0 means unaligned (different from align(1)).
def Alignment : IntAttr<"align", [ParamAttr, RetAttr]>;

/// The result of the function is guaranteed to point to a number of bytes that
/// we can determine if we know the value of the function's arguments.
def AllocSize : IntAttr<"allocsize", [FnAttr]>;

/// inline=always.
def AlwaysInline : EnumAttr<"alwaysinline", [FnAttr]>;

/// Function can access memory only using pointers based on its arguments.
def ArgMemOnly : EnumAttr<"argmemonly", [FnAttr]>;

/// Callee is recognized as a builtin, despite nobuiltin attribute on its
/// declaration.
def Builtin : EnumAttr<"builtin", [FnAttr]>;

/// Pass structure by value.
def ByVal : TypeAttr<"byval", [ParamAttr]>;

/// Mark in-memory ABI type.
def ByRef : TypeAttr<"byref", [ParamAttr]>;

/// Parameter or return value may not contain uninitialized or poison bits.
def NoUndef : EnumAttr<"noundef", [ParamAttr, RetAttr]>;

/// Marks function as being in a cold path.
def Cold : EnumAttr<"cold", [FnAttr]>;

/// Can only be moved to control-equivalent blocks.
def Convergent : EnumAttr<"convergent", [FnAttr]>;

/// Marks function as being in a hot path and frequently called.
def Hot: EnumAttr<"hot", [FnAttr]>;

/// Pointer is known to be dereferenceable.
def Dereferenceable : IntAttr<"dereferenceable", [ParamAttr, RetAttr]>;

/// Pointer is either null or dereferenceable.
def DereferenceableOrNull : IntAttr<"dereferenceable_or_null",
                                    [ParamAttr, RetAttr]>;

/// Do not instrument function with sanitizers.
def DisableSanitizerInstrumentation: EnumAttr<"disable_sanitizer_instrumentation", [FnAttr]>;

/// Provide pointer element type to intrinsic.
def ElementType : TypeAttr<"elementtype", [ParamAttr]>;

/// Function may only access memory that is inaccessible from IR.
def InaccessibleMemOnly : EnumAttr<"inaccessiblememonly", [FnAttr]>;

/// Function may only access memory that is either inaccessible from the IR,
/// or pointed to by its pointer arguments.
def InaccessibleMemOrArgMemOnly : EnumAttr<"inaccessiblemem_or_argmemonly",
                                           [FnAttr]>;

/// Pass structure in an alloca.
def InAlloca : TypeAttr<"inalloca", [ParamAttr]>;

/// Source said inlining was desirable.
def InlineHint : EnumAttr<"inlinehint", [FnAttr]>;

/// Force argument to be passed in register.
def InReg : EnumAttr<"inreg", [ParamAttr, RetAttr]>;

/// Build jump-instruction tables and replace refs.
def JumpTable : EnumAttr<"jumptable", [FnAttr]>;

/// Function must be optimized for size first.
def MinSize : EnumAttr<"minsize", [FnAttr]>;

/// Naked function.
def Naked : EnumAttr<"naked", [FnAttr]>;

/// Nested function static chain.
def Nest : EnumAttr<"nest", [ParamAttr]>;

/// Considered to not alias after call.
def NoAlias : EnumAttr<"noalias", [ParamAttr, RetAttr]>;

/// Callee isn't recognized as a builtin.
def NoBuiltin : EnumAttr<"nobuiltin", [FnAttr]>;

/// Function cannot enter into caller's translation unit.
def NoCallback : EnumAttr<"nocallback", [FnAttr]>;

/// Function creates no aliases of pointer.
def NoCapture : EnumAttr<"nocapture", [ParamAttr]>;

/// Call cannot be duplicated.
def NoDuplicate : EnumAttr<"noduplicate", [FnAttr]>;

/// Function does not deallocate memory.
def NoFree : EnumAttr<"nofree", [FnAttr, ParamAttr]>;

/// Disable implicit floating point insts.
def NoImplicitFloat : EnumAttr<"noimplicitfloat", [FnAttr]>;

/// inline=never.
def NoInline : EnumAttr<"noinline", [FnAttr]>;

/// Function is called early and/or often, so lazy binding isn't worthwhile.
def NonLazyBind : EnumAttr<"nonlazybind", [FnAttr]>;

/// Disable merging for specified functions or call sites.
def NoMerge : EnumAttr<"nomerge", [FnAttr]>;

/// Pointer is known to be not null.
def NonNull : EnumAttr<"nonnull", [ParamAttr, RetAttr]>;

/// The function does not recurse.
def NoRecurse : EnumAttr<"norecurse", [FnAttr]>;

/// Disable redzone.
def NoRedZone : EnumAttr<"noredzone", [FnAttr]>;

/// Mark the function as not returning.
def NoReturn : EnumAttr<"noreturn", [FnAttr]>;

/// Function does not synchronize.
def NoSync : EnumAttr<"nosync", [FnAttr]>;

/// Disable Indirect Branch Tracking.
def NoCfCheck : EnumAttr<"nocf_check", [FnAttr]>;

/// Function should not be instrumented.
def NoProfile : EnumAttr<"noprofile", [FnAttr]>;

/// Function doesn't unwind stack.
def NoUnwind : EnumAttr<"nounwind", [FnAttr]>;

/// No SanitizeCoverage instrumentation.
def NoSanitizeCoverage : EnumAttr<"nosanitize_coverage", [FnAttr]>;

/// Null pointer in address space zero is valid.
def NullPointerIsValid : EnumAttr<"null_pointer_is_valid", [FnAttr]>;

/// Select optimizations for best fuzzing signal.
def OptForFuzzing : EnumAttr<"optforfuzzing", [FnAttr]>;

/// opt_size.
def OptimizeForSize : EnumAttr<"optsize", [FnAttr]>;

/// Function must not be optimized.
def OptimizeNone : EnumAttr<"optnone", [FnAttr]>;

/// Similar to byval but without a copy.
def Preallocated : TypeAttr<"preallocated", [FnAttr, ParamAttr]>;

/// Function does not access memory.
def ReadNone : EnumAttr<"readnone", [FnAttr, ParamAttr]>;

/// Function only reads from memory.
def ReadOnly : EnumAttr<"readonly", [FnAttr, ParamAttr]>;

/// Return value is always equal to this argument.
def Returned : EnumAttr<"returned", [ParamAttr]>;

/// Parameter is required to be a trivial constant.
def ImmArg : EnumAttr<"immarg", [ParamAttr]>;

/// Function can return twice.
def ReturnsTwice : EnumAttr<"returns_twice", [FnAttr]>;

/// Safe Stack protection.
def SafeStack : EnumAttr<"safestack", [FnAttr]>;

/// Shadow Call Stack protection.
def ShadowCallStack : EnumAttr<"shadowcallstack", [FnAttr]>;

/// Sign extended before/after call.
def SExt : EnumAttr<"signext", [ParamAttr, RetAttr]>;

/// Alignment of stack for function (3 bits)  stored as log2 of alignment with
/// +1 bias 0 means unaligned (different from alignstack=(1)).
def StackAlignment : IntAttr<"alignstack", [FnAttr, ParamAttr]>;

/// Function can be speculated.
def Speculatable : EnumAttr<"speculatable", [FnAttr]>;

/// Stack protection.
def StackProtect : EnumAttr<"ssp", [FnAttr]>;

/// Stack protection required.
def StackProtectReq : EnumAttr<"sspreq", [FnAttr]>;

/// Strong Stack protection.
def StackProtectStrong : EnumAttr<"sspstrong", [FnAttr]>;

/// Function was called in a scope requiring strict floating point semantics.
def StrictFP : EnumAttr<"strictfp", [FnAttr]>;

/// Hidden pointer to structure to return.
def StructRet : TypeAttr<"sret", [ParamAttr]>;

/// AddressSanitizer is on.
def SanitizeAddress : EnumAttr<"sanitize_address", [FnAttr]>;

/// ThreadSanitizer is on.
def SanitizeThread : EnumAttr<"sanitize_thread", [FnAttr]>;

/// MemorySanitizer is on.
def SanitizeMemory : EnumAttr<"sanitize_memory", [FnAttr]>;

/// HWAddressSanitizer is on.
def SanitizeHWAddress : EnumAttr<"sanitize_hwaddress", [FnAttr]>;

/// MemTagSanitizer is on.
def SanitizeMemTag : EnumAttr<"sanitize_memtag", [FnAttr]>;

/// Speculative Load Hardening is enabled.
///
/// Note that this uses the default compatibility (always compatible during
/// inlining) and a conservative merge strategy where inlining an attributed
/// body will add the attribute to the caller. This ensures that code carrying
/// this attribute will always be lowered with hardening enabled.
def SpeculativeLoadHardening : EnumAttr<"speculative_load_hardening",
                                        [FnAttr]>;

/// Argument is swift error.
def SwiftError : EnumAttr<"swifterror", [ParamAttr]>;

/// Argument is swift self/context.
def SwiftSelf : EnumAttr<"swiftself", [ParamAttr]>;

/// Argument is swift async context.
def SwiftAsync : EnumAttr<"swiftasync", [ParamAttr]>;

/// Function must be in a unwind table.
def UWTable : EnumAttr<"uwtable", [FnAttr]>;

/// Minimum/Maximum vscale value for function.
def VScaleRange : IntAttr<"vscale_range", [FnAttr]>;

/// Function always comes back to callsite.
def WillReturn : EnumAttr<"willreturn", [FnAttr]>;

/// Function only writes to memory.
def WriteOnly : EnumAttr<"writeonly", [FnAttr, ParamAttr]>;

/// Zero extended before/after call.
def ZExt : EnumAttr<"zeroext", [ParamAttr, RetAttr]>;

/// Function is required to make Forward Progress.
def MustProgress : EnumAttr<"mustprogress", [FnAttr]>;

/// Target-independent string attributes.
def LessPreciseFPMAD : StrBoolAttr<"less-precise-fpmad">;
def NoInfsFPMath : StrBoolAttr<"no-infs-fp-math">;
def NoNansFPMath : StrBoolAttr<"no-nans-fp-math">;
def NoSignedZerosFPMath : StrBoolAttr<"no-signed-zeros-fp-math">;
def UnsafeFPMath : StrBoolAttr<"unsafe-fp-math">;
def NoJumpTables : StrBoolAttr<"no-jump-tables">;
def NoInlineLineTables : StrBoolAttr<"no-inline-line-tables">;
def ProfileSampleAccurate : StrBoolAttr<"profile-sample-accurate">;
def UseSampleProfile : StrBoolAttr<"use-sample-profile">;

class CompatRule<string F> {
  // The name of the function called to check the attribute of the caller and
  // callee and decide whether inlining should be allowed. The function's
  // signature must match "bool(const Function&, const Function &)", where the
  // first parameter is the reference to the caller and the second parameter is
  // the reference to the callee. It must return false if the attributes of the
  // caller and callee are incompatible, and true otherwise.
  string CompatFunc = F;
}

def : CompatRule<"isEqual<SanitizeAddressAttr>">;
def : CompatRule<"isEqual<SanitizeThreadAttr>">;
def : CompatRule<"isEqual<SanitizeMemoryAttr>">;
def : CompatRule<"isEqual<SanitizeHWAddressAttr>">;
def : CompatRule<"isEqual<SanitizeMemTagAttr>">;
def : CompatRule<"isEqual<SafeStackAttr>">;
def : CompatRule<"isEqual<ShadowCallStackAttr>">;
def : CompatRule<"isEqual<UseSampleProfileAttr>">;
def : CompatRule<"isEqual<NoProfileAttr>">;

class MergeRule<string F> {
  // The name of the function called to merge the attributes of the caller and
  // callee. The function's signature must match
  // "void(Function&, const Function &)", where the first parameter is the
  // reference to the caller and the second parameter is the reference to the
  // callee.
  string MergeFunc = F;
}

def : MergeRule<"setAND<LessPreciseFPMADAttr>">;
def : MergeRule<"setAND<NoInfsFPMathAttr>">;
def : MergeRule<"setAND<NoNansFPMathAttr>">;
def : MergeRule<"setAND<NoSignedZerosFPMathAttr>">;
def : MergeRule<"setAND<UnsafeFPMathAttr>">;
def : MergeRule<"setOR<NoImplicitFloatAttr>">;
def : MergeRule<"setOR<NoJumpTablesAttr>">;
def : MergeRule<"setOR<ProfileSampleAccurateAttr>">;
def : MergeRule<"setOR<SpeculativeLoadHardeningAttr>">;
def : MergeRule<"adjustCallerSSPLevel">;
def : MergeRule<"adjustCallerStackProbes">;
def : MergeRule<"adjustCallerStackProbeSize">;
def : MergeRule<"adjustMinLegalVectorWidth">;
def : MergeRule<"adjustNullPointerValidAttr">;
def : MergeRule<"setAND<MustProgressAttr>">;

// P2 specific attributes
def Cogtext : EnumAttr<"cogtext", [FnAttr]>;
def Cogmain : EnumAttr<"cogmain", [FnAttr]>;