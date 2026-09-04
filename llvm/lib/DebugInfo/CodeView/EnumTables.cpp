//===- EnumTables.cpp - Enum to string conversion tables ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/ADT/Enum.h"
#include <type_traits>

using namespace llvm;
using namespace codeview;

#define CV_ENUM_CLASS_ENT(enum_class, enum)                                    \
  {{#enum}, std::underlying_type_t<enum_class>(enum_class::enum)}

#define CV_ENUM_ENT(ns, enum) {{#enum}, ns::enum}

namespace llvm {
namespace codeview {

EnumStrings<SymbolKind> getSymbolTypeNames() {
  constexpr EnumStringDef<SymbolKind> SymbolTypeNameDefs[] = {
#define CV_SYMBOL(enum, val) {{#enum}, enum},
#include "llvm/DebugInfo/CodeView/CodeViewSymbols.def"
#undef CV_SYMBOL
  };
  static constexpr auto SymbolTypeNames =
      BUILD_ENUM_STRINGS(SymbolTypeNameDefs);
  return SymbolTypeNames;
}

EnumStrings<TypeLeafKind> getTypeLeafNames() {
  constexpr EnumStringDef<TypeLeafKind> TypeLeafNameDefs[] = {
#define CV_TYPE(name, val) {{#name}, name},
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"
#undef CV_TYPE
  };
  static constexpr auto TypeLeafNames = BUILD_ENUM_STRINGS(TypeLeafNameDefs);
  return TypeLeafNames;
}

EnumStrings<uint16_t> getRegisterNames(CPUType Cpu) {
  constexpr EnumStringDef<uint16_t> RegisterNameDefs_X86[] = {
#define CV_REGISTERS_X86
#define CV_REGISTER(name, val) CV_ENUM_CLASS_ENT(RegisterId, name),
#include "llvm/DebugInfo/CodeView/CodeViewRegisters.def"
#undef CV_REGISTER
#undef CV_REGISTERS_X86
  };
  static constexpr auto RegisterNames_X86 =
      BUILD_ENUM_STRINGS(RegisterNameDefs_X86);

  constexpr EnumStringDef<uint16_t> RegisterNameDefs_ARM[] = {
#define CV_REGISTERS_ARM
#define CV_REGISTER(name, val) CV_ENUM_CLASS_ENT(RegisterId, name),
#include "llvm/DebugInfo/CodeView/CodeViewRegisters.def"
#undef CV_REGISTER
#undef CV_REGISTERS_ARM
  };
  static constexpr auto RegisterNames_ARM =
      BUILD_ENUM_STRINGS(RegisterNameDefs_ARM);

  constexpr EnumStringDef<uint16_t> RegisterNameDefs_ARM64[] = {
#define CV_REGISTERS_ARM64
#define CV_REGISTER(name, val) CV_ENUM_CLASS_ENT(RegisterId, name),
#include "llvm/DebugInfo/CodeView/CodeViewRegisters.def"
#undef CV_REGISTER
#undef CV_REGISTERS_ARM64
  };
  static constexpr auto RegisterNames_ARM64 =
      BUILD_ENUM_STRINGS(RegisterNameDefs_ARM64);

  if (Cpu == CPUType::ARMNT) {
    return RegisterNames_ARM;
  } else if (Cpu == CPUType::ARM64) {
    return RegisterNames_ARM64;
  }
  return RegisterNames_X86;
}

EnumStrings<uint32_t> getPublicSymFlagNames() {
  constexpr EnumStringDef<uint32_t> PublicSymFlagNameDefs[] = {
      CV_ENUM_CLASS_ENT(PublicSymFlags, Code),
      CV_ENUM_CLASS_ENT(PublicSymFlags, Function),
      CV_ENUM_CLASS_ENT(PublicSymFlags, Managed),
      CV_ENUM_CLASS_ENT(PublicSymFlags, MSIL),
  };
  static constexpr auto PublicSymFlagNames =
      BUILD_ENUM_STRINGS(PublicSymFlagNameDefs);
  return PublicSymFlagNames;
}

EnumStrings<uint8_t> getProcSymFlagNames() {
  constexpr EnumStringDef<uint8_t> ProcSymFlagNameDefs[] = {
      CV_ENUM_CLASS_ENT(ProcSymFlags, HasFP),
      CV_ENUM_CLASS_ENT(ProcSymFlags, HasIRET),
      CV_ENUM_CLASS_ENT(ProcSymFlags, HasFRET),
      CV_ENUM_CLASS_ENT(ProcSymFlags, IsNoReturn),
      CV_ENUM_CLASS_ENT(ProcSymFlags, IsUnreachable),
      CV_ENUM_CLASS_ENT(ProcSymFlags, HasCustomCallingConv),
      CV_ENUM_CLASS_ENT(ProcSymFlags, IsNoInline),
      CV_ENUM_CLASS_ENT(ProcSymFlags, HasOptimizedDebugInfo),
  };
  static constexpr auto ProcSymFlagNames =
      BUILD_ENUM_STRINGS(ProcSymFlagNameDefs);
  return ProcSymFlagNames;
}

EnumStrings<uint16_t> getLocalFlagNames() {
  constexpr EnumStringDef<uint16_t> LocalFlagDefs[] = {
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsParameter),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsAddressTaken),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsCompilerGenerated),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsAggregate),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsAggregated),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsAliased),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsAlias),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsReturnValue),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsOptimizedOut),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsEnregisteredGlobal),
      CV_ENUM_CLASS_ENT(LocalSymFlags, IsEnregisteredStatic),
  };
  static constexpr auto LocalFlags = BUILD_ENUM_STRINGS(LocalFlagDefs);
  return LocalFlags;
}

EnumStrings<uint8_t> getFrameCookieKindNames() {
  constexpr EnumStringDef<uint8_t> FrameCookieKindDefs[] = {
      CV_ENUM_CLASS_ENT(FrameCookieKind, Copy),
      CV_ENUM_CLASS_ENT(FrameCookieKind, XorStackPointer),
      CV_ENUM_CLASS_ENT(FrameCookieKind, XorFramePointer),
      CV_ENUM_CLASS_ENT(FrameCookieKind, XorR13),
  };
  static constexpr auto FrameCookieKinds =
      BUILD_ENUM_STRINGS(FrameCookieKindDefs);
  return FrameCookieKinds;
}

EnumStrings<SourceLanguage> getSourceLanguageNames() {
  constexpr EnumStringDef<codeview::SourceLanguage> SourceLanguageDefs[] = {
      CV_ENUM_ENT(SourceLanguage, C),
      CV_ENUM_ENT(SourceLanguage, Cpp),
      CV_ENUM_ENT(SourceLanguage, Fortran),
      CV_ENUM_ENT(SourceLanguage, Masm),
      CV_ENUM_ENT(SourceLanguage, Pascal),
      CV_ENUM_ENT(SourceLanguage, Basic),
      CV_ENUM_ENT(SourceLanguage, Cobol),
      CV_ENUM_ENT(SourceLanguage, Link),
      CV_ENUM_ENT(SourceLanguage, Cvtres),
      CV_ENUM_ENT(SourceLanguage, Cvtpgd),
      CV_ENUM_ENT(SourceLanguage, CSharp),
      CV_ENUM_ENT(SourceLanguage, VB),
      CV_ENUM_ENT(SourceLanguage, ILAsm),
      CV_ENUM_ENT(SourceLanguage, Java),
      CV_ENUM_ENT(SourceLanguage, JScript),
      CV_ENUM_ENT(SourceLanguage, MSIL),
      CV_ENUM_ENT(SourceLanguage, HLSL),
      CV_ENUM_ENT(SourceLanguage, D),
      CV_ENUM_ENT(SourceLanguage, Swift),
      CV_ENUM_ENT(SourceLanguage, Rust),
      CV_ENUM_ENT(SourceLanguage, ObjC),
      CV_ENUM_ENT(SourceLanguage, ObjCpp),
      CV_ENUM_ENT(SourceLanguage, AliasObj),
      CV_ENUM_ENT(SourceLanguage, Go),
      {{"Swift"}, SourceLanguage::OldSwift},
  };
  static constexpr auto SourceLanguages =
      BUILD_ENUM_STRINGS(SourceLanguageDefs);
  return SourceLanguages;
}

EnumStrings<uint32_t> getCompileSym2FlagNames() {
  constexpr EnumStringDef<uint32_t> CompileSym2FlagNameDefs[] = {
      CV_ENUM_CLASS_ENT(CompileSym2Flags, EC),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, NoDbgInfo),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, LTCG),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, NoDataAlign),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, ManagedPresent),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, SecurityChecks),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, HotPatch),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, CVTCIL),
      CV_ENUM_CLASS_ENT(CompileSym2Flags, MSILModule),
  };
  static constexpr auto CompileSym2FlagNames =
      BUILD_ENUM_STRINGS(CompileSym2FlagNameDefs);
  return CompileSym2FlagNames;
}

EnumStrings<uint32_t> getCompileSym3FlagNames() {
  constexpr EnumStringDef<uint32_t> CompileSym3FlagNameDefs[] = {
      CV_ENUM_CLASS_ENT(CompileSym3Flags, EC),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, NoDbgInfo),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, LTCG),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, NoDataAlign),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, ManagedPresent),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, SecurityChecks),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, HotPatch),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, CVTCIL),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, MSILModule),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, Sdl),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, PGO),
      CV_ENUM_CLASS_ENT(CompileSym3Flags, Exp),
  };
  static constexpr auto CompileSym3FlagNames =
      BUILD_ENUM_STRINGS(CompileSym3FlagNameDefs);
  return CompileSym3FlagNames;
}

EnumStrings<uint32_t> getFileChecksumNames() {
  constexpr EnumStringDef<uint32_t> FileChecksumNameDefs[] = {
      CV_ENUM_CLASS_ENT(FileChecksumKind, None),
      CV_ENUM_CLASS_ENT(FileChecksumKind, MD5),
      CV_ENUM_CLASS_ENT(FileChecksumKind, SHA1),
      CV_ENUM_CLASS_ENT(FileChecksumKind, SHA256),
  };
  static constexpr auto FileChecksumNames =
      BUILD_ENUM_STRINGS(FileChecksumNameDefs);
  return FileChecksumNames;
}

EnumStrings<unsigned> getCPUTypeNames() {
  constexpr EnumStringDef<unsigned> CPUTypeNameDefs[] = {
      CV_ENUM_CLASS_ENT(CPUType, Intel8080),
      CV_ENUM_CLASS_ENT(CPUType, Intel8086),
      CV_ENUM_CLASS_ENT(CPUType, Intel80286),
      CV_ENUM_CLASS_ENT(CPUType, Intel80386),
      CV_ENUM_CLASS_ENT(CPUType, Intel80486),
      CV_ENUM_CLASS_ENT(CPUType, Pentium),
      CV_ENUM_CLASS_ENT(CPUType, PentiumPro),
      CV_ENUM_CLASS_ENT(CPUType, Pentium3),
      CV_ENUM_CLASS_ENT(CPUType, MIPS),
      CV_ENUM_CLASS_ENT(CPUType, MIPS16),
      CV_ENUM_CLASS_ENT(CPUType, MIPS32),
      CV_ENUM_CLASS_ENT(CPUType, MIPS64),
      CV_ENUM_CLASS_ENT(CPUType, MIPSI),
      CV_ENUM_CLASS_ENT(CPUType, MIPSII),
      CV_ENUM_CLASS_ENT(CPUType, MIPSIII),
      CV_ENUM_CLASS_ENT(CPUType, MIPSIV),
      CV_ENUM_CLASS_ENT(CPUType, MIPSV),
      CV_ENUM_CLASS_ENT(CPUType, M68000),
      CV_ENUM_CLASS_ENT(CPUType, M68010),
      CV_ENUM_CLASS_ENT(CPUType, M68020),
      CV_ENUM_CLASS_ENT(CPUType, M68030),
      CV_ENUM_CLASS_ENT(CPUType, M68040),
      CV_ENUM_CLASS_ENT(CPUType, Alpha),
      CV_ENUM_CLASS_ENT(CPUType, Alpha21164),
      CV_ENUM_CLASS_ENT(CPUType, Alpha21164A),
      CV_ENUM_CLASS_ENT(CPUType, Alpha21264),
      CV_ENUM_CLASS_ENT(CPUType, Alpha21364),
      CV_ENUM_CLASS_ENT(CPUType, PPC601),
      CV_ENUM_CLASS_ENT(CPUType, PPC603),
      CV_ENUM_CLASS_ENT(CPUType, PPC604),
      CV_ENUM_CLASS_ENT(CPUType, PPC620),
      CV_ENUM_CLASS_ENT(CPUType, PPCFP),
      CV_ENUM_CLASS_ENT(CPUType, PPCBE),
      CV_ENUM_CLASS_ENT(CPUType, SH3),
      CV_ENUM_CLASS_ENT(CPUType, SH3E),
      CV_ENUM_CLASS_ENT(CPUType, SH3DSP),
      CV_ENUM_CLASS_ENT(CPUType, SH4),
      CV_ENUM_CLASS_ENT(CPUType, SHMedia),
      CV_ENUM_CLASS_ENT(CPUType, ARM3),
      CV_ENUM_CLASS_ENT(CPUType, ARM4),
      CV_ENUM_CLASS_ENT(CPUType, ARM4T),
      CV_ENUM_CLASS_ENT(CPUType, ARM5),
      CV_ENUM_CLASS_ENT(CPUType, ARM5T),
      CV_ENUM_CLASS_ENT(CPUType, ARM6),
      CV_ENUM_CLASS_ENT(CPUType, ARM_XMAC),
      CV_ENUM_CLASS_ENT(CPUType, ARM_WMMX),
      CV_ENUM_CLASS_ENT(CPUType, ARM7),
      CV_ENUM_CLASS_ENT(CPUType, Omni),
      CV_ENUM_CLASS_ENT(CPUType, Ia64),
      CV_ENUM_CLASS_ENT(CPUType, Ia64_2),
      CV_ENUM_CLASS_ENT(CPUType, CEE),
      CV_ENUM_CLASS_ENT(CPUType, AM33),
      CV_ENUM_CLASS_ENT(CPUType, M32R),
      CV_ENUM_CLASS_ENT(CPUType, TriCore),
      CV_ENUM_CLASS_ENT(CPUType, X64),
      CV_ENUM_CLASS_ENT(CPUType, EBC),
      CV_ENUM_CLASS_ENT(CPUType, Thumb),
      CV_ENUM_CLASS_ENT(CPUType, ARMNT),
      CV_ENUM_CLASS_ENT(CPUType, ARM64),
      CV_ENUM_CLASS_ENT(CPUType, HybridX86ARM64),
      CV_ENUM_CLASS_ENT(CPUType, ARM64EC),
      CV_ENUM_CLASS_ENT(CPUType, ARM64X),
      CV_ENUM_CLASS_ENT(CPUType, Unknown),
      CV_ENUM_CLASS_ENT(CPUType, D3D11_Shader),
  };
  static constexpr auto CPUTypeNames = BUILD_ENUM_STRINGS(CPUTypeNameDefs);
  return CPUTypeNames;
}

EnumStrings<uint32_t> getFrameProcSymFlagNames() {
  constexpr EnumStringDef<uint32_t> FrameProcSymFlagNameDefs[] = {
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, HasAlloca),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, HasSetJmp),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, HasLongJmp),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, HasInlineAssembly),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, HasExceptionHandling),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, MarkedInline),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, HasStructuredExceptionHandling),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, Naked),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, SecurityChecks),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, AsynchronousExceptionHandling),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions,
                        NoStackOrderingForSecurityChecks),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, Inlined),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, StrictSecurityChecks),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, SafeBuffers),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, EncodedLocalBasePointerMask),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, EncodedParamBasePointerMask),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, ProfileGuidedOptimization),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, ValidProfileCounts),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, OptimizedForSpeed),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, GuardCfg),
      CV_ENUM_CLASS_ENT(FrameProcedureOptions, GuardCfw),
  };
  static constexpr auto FrameProcSymFlagNames =
      BUILD_ENUM_STRINGS(FrameProcSymFlagNameDefs);
  return FrameProcSymFlagNames;
}

EnumStrings<uint16_t> getExportSymFlagNames() {
  constexpr EnumStringDef<uint16_t> ExportSymFlagNameDefs[] = {
      CV_ENUM_CLASS_ENT(ExportFlags, IsConstant),
      CV_ENUM_CLASS_ENT(ExportFlags, IsData),
      CV_ENUM_CLASS_ENT(ExportFlags, IsPrivate),
      CV_ENUM_CLASS_ENT(ExportFlags, HasNoName),
      CV_ENUM_CLASS_ENT(ExportFlags, HasExplicitOrdinal),
      CV_ENUM_CLASS_ENT(ExportFlags, IsForwarder),
  };
  static constexpr auto ExportSymFlagNames =
      BUILD_ENUM_STRINGS(ExportSymFlagNameDefs);
  return ExportSymFlagNames;
}

EnumStrings<uint32_t> getModuleSubstreamKindNames() {
  constexpr EnumStringDef<uint32_t> ModuleSubstreamKindNameDefs[] = {
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, None),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, Symbols),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, Lines),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, StringTable),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, FileChecksums),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, FrameData),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, InlineeLines),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, CrossScopeImports),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, CrossScopeExports),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, ILLines),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, FuncMDTokenMap),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, TypeMDTokenMap),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, MergedAssemblyInput),
      CV_ENUM_CLASS_ENT(DebugSubsectionKind, CoffSymbolRVA),
  };
  static constexpr auto ModuleSubstreamKindNames =
      BUILD_ENUM_STRINGS(ModuleSubstreamKindNameDefs);
  return ModuleSubstreamKindNames;
}

EnumStrings<uint8_t> getThunkOrdinalNames() {
  constexpr EnumStringDef<uint8_t> ThunkOrdinalNameDefs[] = {
      CV_ENUM_CLASS_ENT(ThunkOrdinal, Standard),
      CV_ENUM_CLASS_ENT(ThunkOrdinal, ThisAdjustor),
      CV_ENUM_CLASS_ENT(ThunkOrdinal, Vcall),
      CV_ENUM_CLASS_ENT(ThunkOrdinal, Pcode),
      CV_ENUM_CLASS_ENT(ThunkOrdinal, UnknownLoad),
      CV_ENUM_CLASS_ENT(ThunkOrdinal, TrampIncremental),
      CV_ENUM_CLASS_ENT(ThunkOrdinal, BranchIsland),
  };
  static constexpr auto ThunkOrdinalNames =
      BUILD_ENUM_STRINGS(ThunkOrdinalNameDefs);
  return ThunkOrdinalNames;
}

EnumStrings<uint16_t> getTrampolineNames() {
  constexpr EnumStringDef<uint16_t> TrampolineNameDefs[] = {
      CV_ENUM_CLASS_ENT(TrampolineType, TrampIncremental),
      CV_ENUM_CLASS_ENT(TrampolineType, BranchIsland),
  };
  static constexpr auto TrampolineNames =
      BUILD_ENUM_STRINGS(TrampolineNameDefs);
  return TrampolineNames;
}

EnumStrings<COFF::SectionCharacteristics> getImageSectionCharacteristicNames() {
  constexpr EnumStringDef<COFF::SectionCharacteristics>
      ImageSectionCharacteristicNameDefs[] = {
          CV_ENUM_ENT(COFF, IMAGE_SCN_TYPE_NOLOAD),
          CV_ENUM_ENT(COFF, IMAGE_SCN_TYPE_NO_PAD),
          CV_ENUM_ENT(COFF, IMAGE_SCN_CNT_CODE),
          CV_ENUM_ENT(COFF, IMAGE_SCN_CNT_INITIALIZED_DATA),
          CV_ENUM_ENT(COFF, IMAGE_SCN_CNT_UNINITIALIZED_DATA),
          CV_ENUM_ENT(COFF, IMAGE_SCN_LNK_OTHER),
          CV_ENUM_ENT(COFF, IMAGE_SCN_LNK_INFO),
          CV_ENUM_ENT(COFF, IMAGE_SCN_LNK_REMOVE),
          CV_ENUM_ENT(COFF, IMAGE_SCN_LNK_COMDAT),
          CV_ENUM_ENT(COFF, IMAGE_SCN_GPREL),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_PURGEABLE),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_16BIT),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_LOCKED),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_PRELOAD),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_1BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_2BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_4BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_8BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_16BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_32BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_64BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_128BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_256BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_512BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_1024BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_2048BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_4096BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_ALIGN_8192BYTES),
          CV_ENUM_ENT(COFF, IMAGE_SCN_LNK_NRELOC_OVFL),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_DISCARDABLE),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_NOT_CACHED),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_NOT_PAGED),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_SHARED),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_EXECUTE),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_READ),
          CV_ENUM_ENT(COFF, IMAGE_SCN_MEM_WRITE)};
  static constexpr auto ImageSectionCharacteristicNames =
      BUILD_ENUM_STRINGS(ImageSectionCharacteristicNameDefs);
  return ImageSectionCharacteristicNames;
}

EnumStrings<uint16_t> getClassOptionNames() {
  constexpr EnumStringDef<uint16_t> ClassOptionNameDefs[] = {
      CV_ENUM_CLASS_ENT(ClassOptions, Packed),
      CV_ENUM_CLASS_ENT(ClassOptions, HasConstructorOrDestructor),
      CV_ENUM_CLASS_ENT(ClassOptions, HasOverloadedOperator),
      CV_ENUM_CLASS_ENT(ClassOptions, Nested),
      CV_ENUM_CLASS_ENT(ClassOptions, ContainsNestedClass),
      CV_ENUM_CLASS_ENT(ClassOptions, HasOverloadedAssignmentOperator),
      CV_ENUM_CLASS_ENT(ClassOptions, HasConversionOperator),
      CV_ENUM_CLASS_ENT(ClassOptions, ForwardReference),
      CV_ENUM_CLASS_ENT(ClassOptions, Scoped),
      CV_ENUM_CLASS_ENT(ClassOptions, HasUniqueName),
      CV_ENUM_CLASS_ENT(ClassOptions, Sealed),
      CV_ENUM_CLASS_ENT(ClassOptions, Intrinsic),
  };
  static constexpr auto ClassOptionNames =
      BUILD_ENUM_STRINGS(ClassOptionNameDefs);
  return ClassOptionNames;
}

EnumStrings<uint8_t> getMemberAccessNames() {
  constexpr EnumStringDef<uint8_t> MemberAccessNameDefs[] = {
      CV_ENUM_CLASS_ENT(MemberAccess, None),
      CV_ENUM_CLASS_ENT(MemberAccess, Private),
      CV_ENUM_CLASS_ENT(MemberAccess, Protected),
      CV_ENUM_CLASS_ENT(MemberAccess, Public),
  };
  static constexpr auto MemberAccessNames =
      BUILD_ENUM_STRINGS(MemberAccessNameDefs);
  return MemberAccessNames;
}

EnumStrings<uint16_t> getMethodOptionNames() {
  constexpr EnumStringDef<uint16_t> MethodOptionNameDefs[] = {
      CV_ENUM_CLASS_ENT(MethodOptions, Pseudo),
      CV_ENUM_CLASS_ENT(MethodOptions, NoInherit),
      CV_ENUM_CLASS_ENT(MethodOptions, NoConstruct),
      CV_ENUM_CLASS_ENT(MethodOptions, CompilerGenerated),
      CV_ENUM_CLASS_ENT(MethodOptions, Sealed),
  };
  static constexpr auto MethodOptionNames =
      BUILD_ENUM_STRINGS(MethodOptionNameDefs);
  return MethodOptionNames;
}

EnumStrings<uint16_t> getMemberKindNames() {
  constexpr EnumStringDef<uint16_t> MemberKindNameDefs[] = {
      CV_ENUM_CLASS_ENT(MethodKind, Vanilla),
      CV_ENUM_CLASS_ENT(MethodKind, Virtual),
      CV_ENUM_CLASS_ENT(MethodKind, Static),
      CV_ENUM_CLASS_ENT(MethodKind, Friend),
      CV_ENUM_CLASS_ENT(MethodKind, IntroducingVirtual),
      CV_ENUM_CLASS_ENT(MethodKind, PureVirtual),
      CV_ENUM_CLASS_ENT(MethodKind, PureIntroducingVirtual),
  };
  static constexpr auto MemberKindNames =
      BUILD_ENUM_STRINGS(MemberKindNameDefs);
  return MemberKindNames;
}

EnumStrings<uint8_t> getPtrKindNames() {
  constexpr EnumStringDef<uint8_t> PtrKindNameDefs[] = {
      CV_ENUM_CLASS_ENT(PointerKind, Near16),
      CV_ENUM_CLASS_ENT(PointerKind, Far16),
      CV_ENUM_CLASS_ENT(PointerKind, Huge16),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnSegment),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnValue),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnSegmentValue),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnAddress),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnSegmentAddress),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnType),
      CV_ENUM_CLASS_ENT(PointerKind, BasedOnSelf),
      CV_ENUM_CLASS_ENT(PointerKind, Near32),
      CV_ENUM_CLASS_ENT(PointerKind, Far32),
      CV_ENUM_CLASS_ENT(PointerKind, Near64),
  };
  static constexpr auto PtrKindNames = BUILD_ENUM_STRINGS(PtrKindNameDefs);
  return PtrKindNames;
}

EnumStrings<uint8_t> getPtrModeNames() {
  constexpr EnumStringDef<uint8_t> PtrModeNameDefs[] = {
      CV_ENUM_CLASS_ENT(PointerMode, Pointer),
      CV_ENUM_CLASS_ENT(PointerMode, LValueReference),
      CV_ENUM_CLASS_ENT(PointerMode, PointerToDataMember),
      CV_ENUM_CLASS_ENT(PointerMode, PointerToMemberFunction),
      CV_ENUM_CLASS_ENT(PointerMode, RValueReference),
  };
  static constexpr auto PtrModeNames = BUILD_ENUM_STRINGS(PtrModeNameDefs);
  return PtrModeNames;
}

EnumStrings<uint16_t> getPtrMemberRepNames() {
  constexpr EnumStringDef<uint16_t> PtrMemberRepNameDefs[] = {
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation, Unknown),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation, SingleInheritanceData),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation, MultipleInheritanceData),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation, VirtualInheritanceData),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation, GeneralData),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                        SingleInheritanceFunction),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                        MultipleInheritanceFunction),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation,
                        VirtualInheritanceFunction),
      CV_ENUM_CLASS_ENT(PointerToMemberRepresentation, GeneralFunction),
  };
  static constexpr auto PtrMemberRepNames =
      BUILD_ENUM_STRINGS(PtrMemberRepNameDefs);
  return PtrMemberRepNames;
}

EnumStrings<uint16_t> getTypeModifierNames() {
  constexpr EnumStringDef<uint16_t> TypeModifierNameDefs[] = {
      CV_ENUM_CLASS_ENT(ModifierOptions, Const),
      CV_ENUM_CLASS_ENT(ModifierOptions, Volatile),
      CV_ENUM_CLASS_ENT(ModifierOptions, Unaligned),
  };
  static constexpr auto TypeModifierNames =
      BUILD_ENUM_STRINGS(TypeModifierNameDefs);
  return TypeModifierNames;
}

EnumStrings<uint8_t> getCallingConventions() {
  constexpr EnumStringDef<uint8_t> CallingConventionDefs[] = {
      CV_ENUM_CLASS_ENT(CallingConvention, NearC),
      CV_ENUM_CLASS_ENT(CallingConvention, FarC),
      CV_ENUM_CLASS_ENT(CallingConvention, NearPascal),
      CV_ENUM_CLASS_ENT(CallingConvention, FarPascal),
      CV_ENUM_CLASS_ENT(CallingConvention, NearFast),
      CV_ENUM_CLASS_ENT(CallingConvention, FarFast),
      CV_ENUM_CLASS_ENT(CallingConvention, NearStdCall),
      CV_ENUM_CLASS_ENT(CallingConvention, FarStdCall),
      CV_ENUM_CLASS_ENT(CallingConvention, NearSysCall),
      CV_ENUM_CLASS_ENT(CallingConvention, FarSysCall),
      CV_ENUM_CLASS_ENT(CallingConvention, ThisCall),
      CV_ENUM_CLASS_ENT(CallingConvention, MipsCall),
      CV_ENUM_CLASS_ENT(CallingConvention, Generic),
      CV_ENUM_CLASS_ENT(CallingConvention, AlphaCall),
      CV_ENUM_CLASS_ENT(CallingConvention, PpcCall),
      CV_ENUM_CLASS_ENT(CallingConvention, SHCall),
      CV_ENUM_CLASS_ENT(CallingConvention, ArmCall),
      CV_ENUM_CLASS_ENT(CallingConvention, AM33Call),
      CV_ENUM_CLASS_ENT(CallingConvention, TriCall),
      CV_ENUM_CLASS_ENT(CallingConvention, SH5Call),
      CV_ENUM_CLASS_ENT(CallingConvention, M32RCall),
      CV_ENUM_CLASS_ENT(CallingConvention, ClrCall),
      CV_ENUM_CLASS_ENT(CallingConvention, Inline),
      CV_ENUM_CLASS_ENT(CallingConvention, NearVector),
      CV_ENUM_CLASS_ENT(CallingConvention, Swift),
  };
  static constexpr auto CallingConventions =
      BUILD_ENUM_STRINGS(CallingConventionDefs);
  return CallingConventions;
}

EnumStrings<uint8_t> getFunctionOptionEnum() {
  constexpr EnumStringDef<uint8_t> FunctionOptionEnumDefs[] = {
      CV_ENUM_CLASS_ENT(FunctionOptions, CxxReturnUdt),
      CV_ENUM_CLASS_ENT(FunctionOptions, Constructor),
      CV_ENUM_CLASS_ENT(FunctionOptions, ConstructorWithVirtualBases),
  };
  static constexpr auto FunctionOptionEnum =
      BUILD_ENUM_STRINGS(FunctionOptionEnumDefs);
  return FunctionOptionEnum;
}

EnumStrings<uint16_t> getLabelTypeEnum() {
  constexpr EnumStringDef<uint16_t> LabelTypeEnumDefs[] = {
      CV_ENUM_CLASS_ENT(LabelType, Near),
      CV_ENUM_CLASS_ENT(LabelType, Far),
  };
  static constexpr auto LabelTypeEnum = BUILD_ENUM_STRINGS(LabelTypeEnumDefs);
  return LabelTypeEnum;
}

EnumStrings<uint16_t> getJumpTableEntrySizeNames() {
  constexpr EnumStringDef<uint16_t> JumpTableEntrySizeNameDefs[] = {
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, Int8),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, UInt8),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, Int16),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, UInt16),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, Int32),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, UInt32),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, Pointer),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, UInt8ShiftLeft),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, UInt16ShiftLeft),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, Int8ShiftLeft),
      CV_ENUM_CLASS_ENT(JumpTableEntrySize, Int16ShiftLeft),
  };
  static constexpr auto JumpTableEntrySizeNames =
      BUILD_ENUM_STRINGS(JumpTableEntrySizeNameDefs);
  return JumpTableEntrySizeNames;
}

} // end namespace codeview
} // end namespace llvm
