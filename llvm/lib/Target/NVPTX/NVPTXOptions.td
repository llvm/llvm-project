//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def NVPTXOptions : OptionsStruct<prefix = "nvptx-">;

// NVPTXAliasAnalysis.cpp
defm : ValueField<"nvptx-traverse-address-aliasing-limit", "unsigned", "6",
  "Depth limit for finding address space through traversal">;

// NVPTXCodeGenPassBuilder.cpp
// byval arguments in NVPTX are special. We're only allowed to read from them
// using a special instruction, and if we ever need to write to them or take an
// address, we must make a local copy and use it, instead.
//
// The problem is that local copies are very expensive, and we create them very
// late in the compilation pipeline, so LLVM does not have much of a chance to
// eliminate them, if they turn out to be unnecessary.
//
// One way around that is to create such copies early on, and let them percolate
// through the optimizations. The copying itself will never trigger creation of
// another copy later on, as the reads are allowed. If LLVM can eliminate it,
// it's a win. If the full optimization pipeline can't remove the copy, that's
// as good as it gets in terms of the effort we could've done, and it's
// certainly a much better effort than what we do now.
//
// This early injection of the copies has potential to create undesirable
// side-effects, so it's disabled by default, for now, until it sees more
// testing.
defm : BoolField<"nvptx-early-byval-copy", "false",
  "Create a copy of byval function arguments early.">;

// NVPTXCtorDtorLowering.cpp
defm : BoolField<"nvptx-emit-init-fini-kernel", "true",
  "Emit kernels to call ctor/dtor globals.">;
defm : ValueField<"nvptx-lower-global-ctor-dtor-id", "StringRef", "",
  "Override unique ID of ctor/dtor globals.">;

// NVPTXDwarfDebug.cpp
// Command line option to control inlined_at enhancement to lineinfo support.
// Valid only when debuginfo emissionkind is DebugDirectivesOnly or
// LineTablesOnly.
defm : BoolField<"line-info-inlined-at", "true",
  "Emit line with inlined_at enhancement for NVPTX">;

// NVPTXISelDAGToDAG.cpp
// FIXME: This is a WAR to recover lost performance from #155024.
// We still need to investigate the regression and find a more permanent
// solution.
defm : BoolField<"nvptx-mad-wide-opt", "false",
  "Enable MAD wide optimization">;
defm : BoolField<"nvptx-rsqrt-approx-opt", "true",
  "Enable reciprocal sqrt optimization">;

// NVPTXISelLowering.cpp
// PTX atom.add.f32 has fixed FTZ behavior that may not match the function's
// (see shouldExpandAtomicRMWInIR), so by default we fall back to a CAS loop
// when they disagree (strict mode), unless declared insignificant via
// !atomic.ignore.denormal.mode metadata or this option. This option allows
// using atom.add anyway, trading correct denormal handling for the speed of the
// native instruction.
defm : BoolField<"nvptx-allow-ftz-atomics", "false",
  "NVPTX Specific: Lower atomicrmw fadd to atom.add even when its FTZ behavior does not match the function's denormal mode.">;
/// Whereas CUDA's implementation (see libdevice) uses ex2.approx for exp2(), it
/// does NOT use lg2.approx for log2, so this is disabled by default.
defm : BoolField<"nvptx-approx-log2f32", "false",
  "NVPTX Specific: whether to use lg2.approx for log2">;
defm : OptionalValueField<"nvptx-fma-level", "unsigned",
  "NVPTX Specific: FMA contraction (0: don't do it, 1: do it, 2: do it aggressively)">;
defm : OptionalEnumField<"nvptx-prec-divf32", "NVPTX::DivPrecisionLevel",
  "0,1,2,3", ["NVPTX::DivPrecisionLevel::Approx", "NVPTX::DivPrecisionLevel::Full", "NVPTX::DivPrecisionLevel::IEEE754", "NVPTX::DivPrecisionLevel::IEEE754_NoFTZ"],
  "NVPTX Specific: Override the precision of the lowering for f32 fdiv">;
defm : OptionalBoolField<"nvptx-prec-sqrtf32",
  "NVPTX Specific: 0 use sqrt.approx, 1 use sqrt.rn.">;
defm : BoolField<"nvptx-sched4reg", "false",
  "NVPTX Specific: schedule for register pressure">;

// NVPTXSubtarget.cpp
defm : BoolField<"nvptx-no-f16-math", "false",
  "NVPTX Specific: Disable generation of f16 math ops.">;
defm : BoolField<"nvptx-no-f32x2", "false",
  "NVPTX Specific: Disable generation of f32x2 instructions and registers.">;

// NVPTXTargetMachine.cpp
// NVPTX IR Peephole is a new pass; this option will let us turn it off in case
// we encounter some issues.
defm : BoolField<"disable-nvptx-ir-peephole", "false",
  "Disable NVPTX IR Peephole">;
// LSV is still relatively new; this switch lets us turn it off in case we
// encounter (or suspect) a bug.
defm : BoolField<"disable-nvptx-load-store-vectorizer", "false",
  "Disable load/store vectorizer">;
// TODO: Remove this flag when we are confident with no regressions.
defm : BoolField<"disable-nvptx-require-structured-cfg", "false",
  "Transitional flag to turn off NVPTX's requirement on preserving structured CFG. The requirement should be disabled only when unexpected regressions happen.">;

// NVPTXUtilities.cpp
defm : BoolField<"nvptx-force-min-byval-param-align", "false",
  "NVPTX Specific: force 4-byte minimal alignment for byval params of device functions.">;

// NVVMReflect.cpp
// Allow users to specify additional key/value pairs to reflect. These key/value
// pairs are the last to be added to the ReflectMap, and therefore will take
// precedence over initial values (i.e. __CUDA_FTZ from module metadata and
// __CUDA_ARCH from SmVersion).
defm : ListField<"nvvm-reflect-add", "StringRef",
  "A key=value pair. Replace __nvvm_reflect(name) with value.">,
  MetaVarName<"name=<int>">;
defm : BoolField<"nvvm-reflect-enable", "true",
  "NVVM reflection, enabled by default">;
