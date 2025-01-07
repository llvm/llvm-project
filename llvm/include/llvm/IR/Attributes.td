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



/// Intersection rules. Used for example in sinking/hoisting two
/// callbases to find a set of attributes that apply to both.
/// Note, there are some attributes we can (probably) legally drop
/// but are intentionally excluded as of now.
///
/// When intersecting the attributes both must be present and equal.
/// Use this for attributes it is not safe to drop at any time. E.g.
/// `byval(Ty)` on a parameter.
def IntersectPreserve : AttrProperty;

/// When intersecting take the AND of the two attrs.
/// Only valid for Enum attrs.
def IntersectAnd : AttrProperty;

/// When intersecting take the min value of the two attrs.
/// Only valid for Int attrs.
def IntersectMin : AttrProperty;

/// When intersecting rely on some specially defined code.
def IntersectCustom : AttrProperty;



/// Attribute base class.
class Attr<string S, AttrProperty I, list<AttrProperty> P> {
  // String representation of this attribute in the IR.
  string AttrString = S;
  list<AttrProperty> Properties = P # [I];
}

/// Enum attribute.
class EnumAttr<string S, AttrProperty I, list<AttrProperty> P> : Attr<S, I, P>;

/// Int attribute.
class IntAttr<string S, AttrProperty I, list<AttrProperty> P> : Attr<S, I, P>;

/// Type attribute.
class TypeAttr<string S, AttrProperty I, list<AttrProperty> P> : Attr<S, I, P>;

/// StringBool attribute.
class StrBoolAttr<string S> : Attr<S, IntersectPreserve, []>;

/// Arbitrary string attribute.
class ComplexStrAttr<string S, list<AttrProperty> P> : Attr<S, IntersectPreserve, P>;

/// ConstantRange attribute.
class ConstantRangeAttr<string S, AttrProperty I, list<AttrProperty> P> : Attr<S, I, P>;

/// ConstantRangeList attribute.
class ConstantRangeListAttr<string S, AttrProperty I, list<AttrProperty> P> : Attr<S, I, P>;

/// Target-independent enum attributes.

/// Alignment of parameter (5 bits) stored as log2 of alignment with +1 bias.
/// 0 means unaligned (different from align(1)).
def Alignment : IntAttr<"align", IntersectCustom, [ParamAttr, RetAttr]>;

/// Parameter of a function that tells us the alignment of an allocation, as in
/// aligned_alloc and aligned ::operator::new.
def AllocAlign: EnumAttr<"allocalign", IntersectAnd, [ParamAttr]>;

/// Describes behavior of an allocator function in terms of known properties.
def AllocKind: IntAttr<"allockind", IntersectPreserve, [FnAttr]>;

/// Parameter is the pointer to be manipulated by the allocator function.
def AllocatedPointer : EnumAttr<"allocptr", IntersectAnd, [ParamAttr]>;

/// The result of the function is guaranteed to point to a number of bytes that
/// we can determine if we know the value of the function's arguments.
def AllocSize : IntAttr<"allocsize", IntersectPreserve, [FnAttr]>;

/// inline=always.
def AlwaysInline : EnumAttr<"alwaysinline", IntersectPreserve, [FnAttr]>;

/// Callee is recognized as a builtin, despite nobuiltin attribute on its
/// declaration.
def Builtin : EnumAttr<"builtin", IntersectPreserve, [FnAttr]>;

/// Pass structure by value.
def ByVal : TypeAttr<"byval", IntersectPreserve, [ParamAttr]>;

/// Mark in-memory ABI type.
def ByRef : TypeAttr<"byref", IntersectPreserve, [ParamAttr]>;

/// Parameter or return value may not contain uninitialized or poison bits.
def NoUndef : EnumAttr<"noundef", IntersectAnd, [ParamAttr, RetAttr]>;

/// Marks function as being in a cold path.
def Cold : EnumAttr<"cold", IntersectAnd, [FnAttr]>;

/// Can only be moved to control-equivalent blocks.
/// NB: Could be IntersectCustom with "or" handling.
def Convergent : EnumAttr<"convergent", IntersectPreserve, [FnAttr]>;

/// Marks function as being in a hot path and frequently called.
def Hot: EnumAttr<"hot", IntersectAnd, [FnAttr]>;

/// Pointer is known to be dereferenceable.
def Dereferenceable : IntAttr<"dereferenceable", IntersectMin, [ParamAttr, RetAttr]>;

/// Pointer is either null or dereferenceable.
def DereferenceableOrNull : IntAttr<"dereferenceable_or_null", IntersectMin,
                                    [ParamAttr, RetAttr]>;

/// Do not instrument function with sanitizers.
def DisableSanitizerInstrumentation: EnumAttr<"disable_sanitizer_instrumentation", IntersectPreserve, [FnAttr]>;

/// Provide pointer element type to intrinsic.
def ElementType : TypeAttr<"elementtype", IntersectPreserve, [ParamAttr]>;

/// Whether to keep return instructions, or replace with a jump to an external
/// symbol.
def FnRetThunkExtern : EnumAttr<"fn_ret_thunk_extern", IntersectPreserve, [FnAttr]>;

/// Function has a hybrid patchable thunk.
def HybridPatchable : EnumAttr<"hybrid_patchable", IntersectPreserve, [FnAttr]>;

/// Pass structure in an alloca.
def InAlloca : TypeAttr<"inalloca", IntersectPreserve, [ParamAttr]>;

/// Pointer argument memory is initialized.
def Initializes : ConstantRangeListAttr<"initializes", IntersectPreserve, [ParamAttr]>;

/// Source said inlining was desirable.
def InlineHint : EnumAttr<"inlinehint", IntersectAnd, [FnAttr]>;

/// Force argument to be passed in register.
def InReg : EnumAttr<"inreg", IntersectPreserve, [ParamAttr, RetAttr]>;

/// Build jump-instruction tables and replace refs.
def JumpTable : EnumAttr<"jumptable", IntersectPreserve, [FnAttr]>;

/// Memory effects of the function.
def Memory : IntAttr<"memory", IntersectCustom, [FnAttr]>;

/// Forbidden floating-point classes.
def NoFPClass : IntAttr<"nofpclass", IntersectCustom, [ParamAttr, RetAttr]>;

/// Function must be optimized for size first.
def MinSize : EnumAttr<"minsize", IntersectPreserve, [FnAttr]>;

/// Naked function.
def Naked : EnumAttr<"naked", IntersectPreserve, [FnAttr]>;

/// Nested function static chain.
def Nest : EnumAttr<"nest", IntersectPreserve, [ParamAttr]>;

/// Considered to not alias after call.
def NoAlias : EnumAttr<"noalias", IntersectAnd, [ParamAttr, RetAttr]>;

/// Callee isn't recognized as a builtin.
def NoBuiltin : EnumAttr<"nobuiltin", IntersectPreserve, [FnAttr]>;

/// Function cannot enter into caller's translation unit.
def NoCallback : EnumAttr<"nocallback", IntersectAnd, [FnAttr]>;

/// Function creates no aliases of pointer.
def NoCapture : EnumAttr<"nocapture", IntersectAnd, [ParamAttr]>;

/// Function is not a source of divergence.
def NoDivergenceSource : EnumAttr<"nodivergencesource", IntersectAnd, [FnAttr]>;

/// Call cannot be duplicated.
def NoDuplicate : EnumAttr<"noduplicate", IntersectPreserve, [FnAttr]>;

/// No extension needed before/after call (high bits are undefined).
def NoExt : EnumAttr<"noext", IntersectPreserve, [ParamAttr, RetAttr]>;

/// Function does not deallocate memory.
def NoFree : EnumAttr<"nofree", IntersectAnd, [FnAttr, ParamAttr]>;

/// Argument is dead if the call unwinds.
def DeadOnUnwind : EnumAttr<"dead_on_unwind", IntersectAnd, [ParamAttr]>;

/// Disable implicit floating point insts.
def NoImplicitFloat : EnumAttr<"noimplicitfloat", IntersectPreserve, [FnAttr]>;

/// inline=never.
def NoInline : EnumAttr<"noinline", IntersectPreserve, [FnAttr]>;

/// Function is called early and/or often, so lazy binding isn't worthwhile.
def NonLazyBind : EnumAttr<"nonlazybind", IntersectPreserve, [FnAttr]>;

/// Disable merging for specified functions or call sites.
def NoMerge : EnumAttr<"nomerge", IntersectPreserve, [FnAttr]>;

/// Pointer is known to be not null.
def NonNull : EnumAttr<"nonnull", IntersectAnd, [ParamAttr, RetAttr]>;

/// The function does not recurse.
def NoRecurse : EnumAttr<"norecurse", IntersectAnd, [FnAttr]>;

/// Disable redzone.
def NoRedZone : EnumAttr<"noredzone", IntersectPreserve, [FnAttr]>;

/// Mark the function as not returning.
def NoReturn : EnumAttr<"noreturn", IntersectAnd, [FnAttr]>;

/// Function does not synchronize.
def NoSync : EnumAttr<"nosync", IntersectAnd, [FnAttr]>;

/// Disable Indirect Branch Tracking.
def NoCfCheck : EnumAttr<"nocf_check", IntersectPreserve, [FnAttr]>;

/// Function should not be instrumented.
def NoProfile : EnumAttr<"noprofile", IntersectPreserve, [FnAttr]>;

/// This function should not be instrumented but it is ok to inline profiled
// functions into it.
def SkipProfile : EnumAttr<"skipprofile", IntersectPreserve, [FnAttr]>;

/// Function doesn't unwind stack.
def NoUnwind : EnumAttr<"nounwind", IntersectAnd, [FnAttr]>;

/// No SanitizeBounds instrumentation.
def NoSanitizeBounds : EnumAttr<"nosanitize_bounds", IntersectPreserve, [FnAttr]>;

/// No SanitizeCoverage instrumentation.
def NoSanitizeCoverage : EnumAttr<"nosanitize_coverage", IntersectPreserve, [FnAttr]>;

/// Null pointer in address space zero is valid.
def NullPointerIsValid : EnumAttr<"null_pointer_is_valid", IntersectPreserve, [FnAttr]>;

/// Select optimizations that give decent debug info.
def OptimizeForDebugging : EnumAttr<"optdebug", IntersectPreserve, [FnAttr]>;

/// Select optimizations for best fuzzing signal.
def OptForFuzzing : EnumAttr<"optforfuzzing", IntersectPreserve, [FnAttr]>;

/// opt_size.
def OptimizeForSize : EnumAttr<"optsize", IntersectPreserve, [FnAttr]>;

/// Function must not be optimized.
def OptimizeNone : EnumAttr<"optnone", IntersectPreserve, [FnAttr]>;

/// Similar to byval but without a copy.
def Preallocated : TypeAttr<"preallocated", IntersectPreserve, [FnAttr, ParamAttr]>;

/// Parameter or return value is within the specified range.
def Range : ConstantRangeAttr<"range", IntersectCustom, [ParamAttr, RetAttr]>;

/// Function does not access memory.
def ReadNone : EnumAttr<"readnone", IntersectAnd, [ParamAttr]>;

/// Function only reads from memory.
def ReadOnly : EnumAttr<"readonly", IntersectAnd, [ParamAttr]>;

/// Return value is always equal to this argument.
def Returned : EnumAttr<"returned", IntersectAnd, [ParamAttr]>;

/// Parameter is required to be a trivial constant.
def ImmArg : EnumAttr<"immarg", IntersectPreserve, [ParamAttr]>;

/// Function can return twice.
def ReturnsTwice : EnumAttr<"returns_twice", IntersectPreserve, [FnAttr]>;

/// Safe Stack protection.
def SafeStack : EnumAttr<"safestack", IntersectPreserve, [FnAttr]>;

/// Shadow Call Stack protection.
def ShadowCallStack : EnumAttr<"shadowcallstack", IntersectPreserve, [FnAttr]>;

/// Sign extended before/after call.
def SExt : EnumAttr<"signext", IntersectPreserve, [ParamAttr, RetAttr]>;

/// Alignment of stack for function (3 bits)  stored as log2 of alignment with
/// +1 bias 0 means unaligned (different from alignstack=(1)).
def StackAlignment : IntAttr<"alignstack", IntersectPreserve, [FnAttr, ParamAttr]>;

/// Function can be speculated.
def Speculatable : EnumAttr<"speculatable", IntersectAnd, [FnAttr]>;

/// Stack protection.
def StackProtect : EnumAttr<"ssp", IntersectPreserve, [FnAttr]>;

/// Stack protection required.
def StackProtectReq : EnumAttr<"sspreq", IntersectPreserve, [FnAttr]>;

/// Strong Stack protection.
def StackProtectStrong : EnumAttr<"sspstrong", IntersectPreserve, [FnAttr]>;

/// Function was called in a scope requiring strict floating point semantics.
def StrictFP : EnumAttr<"strictfp", IntersectPreserve, [FnAttr]>;

/// Hidden pointer to structure to return.
def StructRet : TypeAttr<"sret", IntersectPreserve, [ParamAttr]>;

/// AddressSanitizer is on.
def SanitizeAddress : EnumAttr<"sanitize_address", IntersectPreserve, [FnAttr]>;

/// ThreadSanitizer is on.
def SanitizeThread : EnumAttr<"sanitize_thread", IntersectPreserve, [FnAttr]>;

/// TypeSanitizer is on.
def SanitizeType : EnumAttr<"sanitize_type", IntersectPreserve, [FnAttr]>;

/// MemorySanitizer is on.
def SanitizeMemory : EnumAttr<"sanitize_memory", IntersectPreserve, [FnAttr]>;

/// HWAddressSanitizer is on.
def SanitizeHWAddress : EnumAttr<"sanitize_hwaddress", IntersectPreserve, [FnAttr]>;

/// MemTagSanitizer is on.
def SanitizeMemTag : EnumAttr<"sanitize_memtag", IntersectPreserve, [FnAttr]>;

/// NumericalStabilitySanitizer is on.
def SanitizeNumericalStability : EnumAttr<"sanitize_numerical_stability", IntersectPreserve, [FnAttr]>;

/// RealtimeSanitizer is on.
def SanitizeRealtime : EnumAttr<"sanitize_realtime", IntersectPreserve, [FnAttr]>;

/// RealtimeSanitizer should error if a real-time unsafe function is invoked
/// during a real-time sanitized function (see `sanitize_realtime`).
def SanitizeRealtimeBlocking : EnumAttr<"sanitize_realtime_blocking", IntersectPreserve, [FnAttr]>;

/// Speculative Load Hardening is enabled.
///
/// Note that this uses the default compatibility (always compatible during
/// inlining) and a conservative merge strategy where inlining an attributed
/// body will add the attribute to the caller. This ensures that code carrying
/// this attribute will always be lowered with hardening enabled.
def SpeculativeLoadHardening : EnumAttr<"speculative_load_hardening",
                                        IntersectPreserve,
                                        [FnAttr]>;

/// Argument is swift error.
def SwiftError : EnumAttr<"swifterror", IntersectPreserve, [ParamAttr]>;

/// Argument is swift self/context.
def SwiftSelf : EnumAttr<"swiftself", IntersectPreserve, [ParamAttr]>;

/// Argument is swift async context.
def SwiftAsync : EnumAttr<"swiftasync", IntersectPreserve, [ParamAttr]>;

/// Function must be in a unwind table.
def UWTable : IntAttr<"uwtable", IntersectPreserve, [FnAttr]>;

/// Minimum/Maximum vscale value for function.
def VScaleRange : IntAttr<"vscale_range", IntersectPreserve, [FnAttr]>;

/// Function always comes back to callsite.
def WillReturn : EnumAttr<"willreturn", IntersectAnd, [FnAttr]>;

/// Pointer argument is writable.
def Writable : EnumAttr<"writable", IntersectAnd, [ParamAttr]>;

/// Function only writes to memory.
def WriteOnly : EnumAttr<"writeonly", IntersectAnd, [ParamAttr]>;

/// Zero extended before/after call.
def ZExt : EnumAttr<"zeroext", IntersectPreserve, [ParamAttr, RetAttr]>;

/// Function is required to make Forward Progress.
def MustProgress : EnumAttr<"mustprogress", IntersectAnd, [FnAttr]>;

/// Function is a presplit coroutine.
def PresplitCoroutine : EnumAttr<"presplitcoroutine", IntersectPreserve, [FnAttr]>;

/// The coroutine would only be destroyed when it is complete.
def CoroDestroyOnlyWhenComplete : EnumAttr<"coro_only_destroy_when_complete", IntersectPreserve, [FnAttr]>;

/// The coroutine call meets the elide requirement. Hint the optimization
/// pipeline to perform elide on the call or invoke instruction.
def CoroElideSafe : EnumAttr<"coro_elide_safe", IntersectPreserve, [FnAttr]>;

/// Target-independent string attributes.
def LessPreciseFPMAD : StrBoolAttr<"less-precise-fpmad">;
def NoInfsFPMath : StrBoolAttr<"no-infs-fp-math">;
def NoNansFPMath : StrBoolAttr<"no-nans-fp-math">;
def ApproxFuncFPMath : StrBoolAttr<"approx-func-fp-math">;
def NoSignedZerosFPMath : StrBoolAttr<"no-signed-zeros-fp-math">;
def UnsafeFPMath : StrBoolAttr<"unsafe-fp-math">;
def NoJumpTables : StrBoolAttr<"no-jump-tables">;
def NoInlineLineTables : StrBoolAttr<"no-inline-line-tables">;
def ProfileSampleAccurate : StrBoolAttr<"profile-sample-accurate">;
def UseSampleProfile : StrBoolAttr<"use-sample-profile">;

def DenormalFPMath : ComplexStrAttr<"denormal-fp-math", [FnAttr]>;
def DenormalFPMathF32 : ComplexStrAttr<"denormal-fp-math-f32", [FnAttr]>;

// Attribute compatiblity rules are generated to check the attribute of the
// caller and callee and decide whether inlining should be allowed. CompatRule
// and child classes are used for the rule generation. CompatRule takes only a
// compare function which could be templated with the attribute type.
// CompatRuleStrAttr takes the compare function and the string attribute for
// checking compatibility for inline substitution.
class CompatRule<string F> {
  // The function's signature must match "bool(const Function&, const
  // Function&)", where the first parameter is the reference to the caller and
  // the second parameter is the reference to the callee. It must return false
  // if the attributes of the caller and callee are incompatible, and true
  // otherwise.
  string CompatFunc = F;
  string AttrName = "";
}

class CompatRuleStrAttr<string F, string Attr> : CompatRule<F> {
  // The checker function is extended with an third argument as the function
  // attribute string "bool(const Function&, const Function&, const StringRef&)".
  string AttrName = Attr;
}

def : CompatRule<"isEqual<SanitizeAddressAttr>">;
def : CompatRule<"isEqual<SanitizeThreadAttr>">;
def : CompatRule<"isEqual<SanitizeTypeAttr>">;
def : CompatRule<"isEqual<SanitizeMemoryAttr>">;
def : CompatRule<"isEqual<SanitizeHWAddressAttr>">;
def : CompatRule<"isEqual<SanitizeMemTagAttr>">;
def : CompatRule<"isEqual<SanitizeNumericalStabilityAttr>">;
def : CompatRule<"isEqual<SanitizeRealtimeAttr>">;
def : CompatRule<"isEqual<SanitizeRealtimeBlockingAttr>">;
def : CompatRule<"isEqual<SafeStackAttr>">;
def : CompatRule<"isEqual<ShadowCallStackAttr>">;
def : CompatRule<"isEqual<UseSampleProfileAttr>">;
def : CompatRule<"isEqual<NoProfileAttr>">;
def : CompatRule<"checkDenormMode">;
def : CompatRule<"checkStrictFP">;
def : CompatRuleStrAttr<"isEqual", "sign-return-address">;
def : CompatRuleStrAttr<"isEqual", "sign-return-address-key">;
def : CompatRuleStrAttr<"isEqual", "branch-protection-pauth-lr">;

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
def : MergeRule<"setAND<ApproxFuncFPMathAttr>">;
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
