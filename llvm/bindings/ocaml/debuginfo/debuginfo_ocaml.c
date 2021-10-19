/*===-- debuginfo_ocaml.c - LLVM OCaml Glue ---------------------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
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

#include <string.h>

#include "caml/memory.h"
#include "caml/mlvalues.h"
#include "llvm-c/Core.h"
#include "llvm-c/DebugInfo.h"
#include "llvm-c/Support.h"

#include "llvm_ocaml.h"

// This is identical to the definition in llvm_debuginfo.ml:DIFlag.t
typedef enum {
  i_DIFlagZero,
  i_DIFlagPrivate,
  i_DIFlagProtected,
  i_DIFlagPublic,
  i_DIFlagFwdDecl,
  i_DIFlagAppleBlock,
  i_DIFlagReservedBit4,
  i_DIFlagVirtual,
  i_DIFlagArtificial,
  i_DIFlagExplicit,
  i_DIFlagPrototyped,
  i_DIFlagObjcClassComplete,
  i_DIFlagObjectPointer,
  i_DIFlagVector,
  i_DIFlagStaticMember,
  i_DIFlagLValueReference,
  i_DIFlagRValueReference,
  i_DIFlagReserved,
  i_DIFlagSingleInheritance,
  i_DIFlagMultipleInheritance,
  i_DIFlagVirtualInheritance,
  i_DIFlagIntroducedVirtual,
  i_DIFlagBitField,
  i_DIFlagNoReturn,
  i_DIFlagTypePassByValue,
  i_DIFlagTypePassByReference,
  i_DIFlagEnumClass,
  i_DIFlagFixedEnum,
  i_DIFlagThunk,
  i_DIFlagNonTrivial,
  i_DIFlagBigEndian,
  i_DIFlagLittleEndian,
  i_DIFlagIndirectVirtualBase,
  i_DIFlagAccessibility,
  i_DIFlagPtrToMemberRep
} LLVMDIFlag_i;

static LLVMDIFlags map_DIFlag(LLVMDIFlag_i DIF) {
  switch (DIF) {
  case i_DIFlagZero:
    return LLVMDIFlagZero;
  case i_DIFlagPrivate:
    return LLVMDIFlagPrivate;
  case i_DIFlagProtected:
    return LLVMDIFlagProtected;
  case i_DIFlagPublic:
    return LLVMDIFlagPublic;
  case i_DIFlagFwdDecl:
    return LLVMDIFlagFwdDecl;
  case i_DIFlagAppleBlock:
    return LLVMDIFlagAppleBlock;
  case i_DIFlagReservedBit4:
    return LLVMDIFlagReservedBit4;
  case i_DIFlagVirtual:
    return LLVMDIFlagVirtual;
  case i_DIFlagArtificial:
    return LLVMDIFlagArtificial;
  case i_DIFlagExplicit:
    return LLVMDIFlagExplicit;
  case i_DIFlagPrototyped:
    return LLVMDIFlagPrototyped;
  case i_DIFlagObjcClassComplete:
    return LLVMDIFlagObjcClassComplete;
  case i_DIFlagObjectPointer:
    return LLVMDIFlagObjectPointer;
  case i_DIFlagVector:
    return LLVMDIFlagVector;
  case i_DIFlagStaticMember:
    return LLVMDIFlagStaticMember;
  case i_DIFlagLValueReference:
    return LLVMDIFlagLValueReference;
  case i_DIFlagRValueReference:
    return LLVMDIFlagRValueReference;
  case i_DIFlagReserved:
    return LLVMDIFlagReserved;
  case i_DIFlagSingleInheritance:
    return LLVMDIFlagSingleInheritance;
  case i_DIFlagMultipleInheritance:
    return LLVMDIFlagMultipleInheritance;
  case i_DIFlagVirtualInheritance:
    return LLVMDIFlagVirtualInheritance;
  case i_DIFlagIntroducedVirtual:
    return LLVMDIFlagIntroducedVirtual;
  case i_DIFlagBitField:
    return LLVMDIFlagBitField;
  case i_DIFlagNoReturn:
    return LLVMDIFlagNoReturn;
  case i_DIFlagTypePassByValue:
    return LLVMDIFlagTypePassByValue;
  case i_DIFlagTypePassByReference:
    return LLVMDIFlagTypePassByReference;
  case i_DIFlagEnumClass:
    return LLVMDIFlagEnumClass;
  case i_DIFlagFixedEnum:
    return LLVMDIFlagFixedEnum;
  case i_DIFlagThunk:
    return LLVMDIFlagThunk;
  case i_DIFlagNonTrivial:
    return LLVMDIFlagNonTrivial;
  case i_DIFlagBigEndian:
    return LLVMDIFlagBigEndian;
  case i_DIFlagLittleEndian:
    return LLVMDIFlagLittleEndian;
  case i_DIFlagIndirectVirtualBase:
    return LLVMDIFlagIndirectVirtualBase;
  case i_DIFlagAccessibility:
    return LLVMDIFlagAccessibility;
  case i_DIFlagPtrToMemberRep:
    return LLVMDIFlagPtrToMemberRep;
  }
}

value llvm_debug_metadata_version(value Unit) {
  return Val_int(LLVMDebugMetadataVersion());
}

value llvm_get_module_debug_metadata_version(LLVMModuleRef Module) {
  return Val_int(LLVMGetModuleDebugMetadataVersion(Module));
}

#define DIFlags_val(v) (*(LLVMDIFlags *)(Data_custom_val(v)))

static struct custom_operations diflags_ops = {
    (char *)"DebugInfo.lldiflags", custom_finalize_default,
    custom_compare_default,        custom_hash_default,
    custom_serialize_default,      custom_deserialize_default,
    custom_compare_ext_default};

static value alloc_diflags(LLVMDIFlags Flags) {
  value V = alloc_custom(&diflags_ops, sizeof(LLVMDIFlags), 0, 1);
  DIFlags_val(V) = Flags;
  return V;
}

LLVMDIFlags llvm_diflags_get(value i_Flag) {
  LLVMDIFlags Flags = map_DIFlag(Int_val(i_Flag));
  return alloc_diflags(Flags);
}

LLVMDIFlags llvm_diflags_set(value Flags, value i_Flag) {
  LLVMDIFlags FlagsNew = DIFlags_val(Flags) | map_DIFlag(Int_val(i_Flag));
  return alloc_diflags(FlagsNew);
}

value llvm_diflags_test(value Flags, value i_Flag) {
  LLVMDIFlags Flag = map_DIFlag(Int_val(i_Flag));
  return Val_bool((DIFlags_val(Flags) & Flag) == Flag);
}

#define DIBuilder_val(v) (*(LLVMDIBuilderRef *)(Data_custom_val(v)))

static void llvm_finalize_dibuilder(value B) {
  LLVMDIBuilderFinalize(DIBuilder_val(B));
  LLVMDisposeDIBuilder(DIBuilder_val(B));
}

static struct custom_operations dibuilder_ops = {
    (char *)"DebugInfo.lldibuilder", llvm_finalize_dibuilder,
    custom_compare_default,          custom_hash_default,
    custom_serialize_default,        custom_deserialize_default,
    custom_compare_ext_default};

static value alloc_dibuilder(LLVMDIBuilderRef B) {
  value V = alloc_custom(&dibuilder_ops, sizeof(LLVMDIBuilderRef), 0, 1);
  DIBuilder_val(V) = B;
  return V;
}

/* llmodule -> lldibuilder */
value llvm_dibuilder(LLVMModuleRef M) {
  return alloc_dibuilder(LLVMCreateDIBuilder(M));
}

value llvm_dibuild_finalize(value Builder) {
  LLVMDIBuilderFinalize(DIBuilder_val(Builder));
  return Val_unit;
}

LLVMMetadataRef llvm_dibuild_create_compile_unit_native(
    value Builder, value Lang, LLVMMetadataRef FileRef, value Producer,
    value IsOptimized, value Flags, value RuntimeVer, value SplitName,
    value Kind, value DWOId, value SplitDebugInline,
    value DebugInfoForProfiling, value SysRoot, value SDK) {
  return LLVMDIBuilderCreateCompileUnit(
      DIBuilder_val(Builder), Int_val(Lang), FileRef, String_val(Producer),
      caml_string_length(Producer), Bool_val(IsOptimized), String_val(Flags),
      caml_string_length(Flags), Int_val(RuntimeVer), String_val(SplitName),
      caml_string_length(SplitName), Int_val(Kind), Int_val(DWOId),
      Bool_val(SplitDebugInline), Bool_val(DebugInfoForProfiling),
      String_val(SysRoot), caml_string_length(SysRoot), String_val(SDK),
      caml_string_length(SDK));
}

LLVMMetadataRef llvm_dibuild_create_compile_unit_bytecode(value *argv,
                                                          int argn) {
  return llvm_dibuild_create_compile_unit_native(
      argv[0],                  // Builder
      argv[1],                  // Lang
      (LLVMMetadataRef)argv[2], // FileRef
      argv[3],                  // Producer
      argv[4],                  // IsOptimized
      argv[5],                  // Flags
      argv[6],                  // RuntimeVer
      argv[7],                  // SplitName
      argv[8],                  // Kind
      argv[9],                  // DWOId
      argv[10],                 // SplitDebugInline
      argv[11],                 // DebugInfoForProfiling
      argv[12],                 // SysRoot
      argv[13]                  // SDK
  );
}

LLVMMetadataRef llvm_dibuild_create_file(value Builder, value Filename,
                                         value Directory) {
  return LLVMDIBuilderCreateFile(DIBuilder_val(Builder), String_val(Filename),
                                 caml_string_length(Filename),
                                 String_val(Directory),
                                 caml_string_length(Directory));
}

LLVMMetadataRef
llvm_dibuild_create_module_native(value Builder, LLVMMetadataRef ParentScope,
                                  value Name, value ConfigMacros,
                                  value IncludePath, value SysRoot) {
  return LLVMDIBuilderCreateModule(
      DIBuilder_val(Builder), ParentScope, String_val(Name),
      caml_string_length(Name), String_val(ConfigMacros),
      caml_string_length(ConfigMacros), String_val(IncludePath),
      caml_string_length(IncludePath), String_val(SysRoot),
      caml_string_length(SysRoot));
}

LLVMMetadataRef llvm_dibuild_create_module_bytecode(value *argv, int argn) {
  return llvm_dibuild_create_module_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // ParentScope
      argv[2],                  // Name
      argv[3],                  // ConfigMacros
      argv[4],                  // IncludePath
      argv[5]                   // SysRoot
  );
}

LLVMMetadataRef llvm_dibuild_create_namespace(value Builder,
                                              LLVMMetadataRef ParentScope,
                                              value Name, value ExportSymbols) {
  return LLVMDIBuilderCreateNameSpace(
      DIBuilder_val(Builder), ParentScope, String_val(Name),
      caml_string_length(Name), Bool_val(ExportSymbols));
}

LLVMMetadataRef llvm_dibuild_create_function_native(
    value Builder, LLVMMetadataRef Scope, value Name, value LinkageName,
    LLVMMetadataRef File, value LineNo, LLVMMetadataRef Ty, value IsLocalToUnit,
    value IsDefinition, value ScopeLine, value Flags, value IsOptimized) {
  return LLVMDIBuilderCreateFunction(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      String_val(LinkageName), caml_string_length(LinkageName), File,
      Int_val(LineNo), Ty, Bool_val(IsLocalToUnit), Bool_val(IsDefinition),
      Int_val(ScopeLine), DIFlags_val(Flags), Bool_val(IsOptimized));
}

LLVMMetadataRef llvm_dibuild_create_function_bytecode(value *argv, int argn) {
  return llvm_dibuild_create_function_native(argv[0], // Builder,
                                             (LLVMMetadataRef)argv[1], // Scope
                                             argv[2],                  // Name
                                             argv[3], // LinkageName
                                             (LLVMMetadataRef)argv[4], // File
                                             argv[5],                  // LineNo
                                             (LLVMMetadataRef)argv[6], // Ty
                                             argv[7],  // IsLocalUnit
                                             argv[8],  // IsDefinition
                                             argv[9],  // ScopeLine
                                             argv[10], // Flags
                                             argv[11]  // IsOptimized
  );
}

LLVMMetadataRef llvm_dibuild_create_lexical_block(value Builder,
                                                  LLVMMetadataRef Scope,
                                                  LLVMMetadataRef File,
                                                  value Line, value Column) {
  return LLVMDIBuilderCreateLexicalBlock(DIBuilder_val(Builder), Scope, File,
                                         Int_val(Line), Int_val(Column));
}

LLVMMetadataRef llvm_metadata_null() { return (LLVMMetadataRef)NULL; }

LLVMMetadataRef llvm_dibuild_create_debug_location(LLVMContextRef Ctx,
                                                   value Line, value Column,
                                                   LLVMMetadataRef Scope,
                                                   LLVMMetadataRef InlinedAt) {
  return LLVMDIBuilderCreateDebugLocation(Ctx, Int_val(Line), Int_val(Column),
                                          Scope, InlinedAt);
}

value llvm_di_location_get_line(LLVMMetadataRef Location) {
  return Val_int(LLVMDILocationGetLine(Location));
}

value llvm_di_location_get_column(LLVMMetadataRef Location) {
  return Val_int(LLVMDILocationGetColumn(Location));
}

LLVMMetadataRef llvm_di_location_get_scope(LLVMMetadataRef Location) {
  return LLVMDILocationGetScope(Location);
}

value llvm_di_location_get_inlined_at(LLVMMetadataRef Location) {
  return (ptr_to_option(LLVMDILocationGetInlinedAt(Location)));
}

value llvm_di_scope_get_file(LLVMMetadataRef Scope) {
  return (ptr_to_option(LLVMDIScopeGetFile(Scope)));
}

value llvm_di_file_get_directory(LLVMMetadataRef File) {
  unsigned Len;
  const char *Directory = LLVMDIFileGetDirectory(File, &Len);
  return cstr_to_string(Directory, Len);
}

value llvm_di_file_get_filename(LLVMMetadataRef File) {
  unsigned Len;
  const char *Filename = LLVMDIFileGetFilename(File, &Len);
  return cstr_to_string(Filename, Len);
}

value llvm_di_file_get_source(LLVMMetadataRef File) {
  unsigned Len;
  const char *Source = LLVMDIFileGetSource(File, &Len);
  return cstr_to_string(Source, Len);
}

LLVMMetadataRef llvm_dibuild_get_or_create_type_array(value Builder,
                                                      value Data) {

  return LLVMDIBuilderGetOrCreateTypeArray(DIBuilder_val(Builder),
                                           (LLVMMetadataRef *)Op_val(Data),
                                           Wosize_val(Data));
}

LLVMMetadataRef llvm_dibuild_get_or_create_array(value Builder, value Data) {

  return LLVMDIBuilderGetOrCreateArray(DIBuilder_val(Builder),
                                       (LLVMMetadataRef *)Op_val(Data),
                                       Wosize_val(Data));
}

LLVMMetadataRef llvm_dibuild_create_subroutine_type(value Builder,
                                                    LLVMMetadataRef File,
                                                    value ParameterTypes,
                                                    value Flags) {

  return LLVMDIBuilderCreateSubroutineType(
      DIBuilder_val(Builder), File, (LLVMMetadataRef *)Op_val(ParameterTypes),
      Wosize_val(ParameterTypes), DIFlags_val(Flags));
}

LLVMMetadataRef llvm_dibuild_create_enumerator(value Builder, value Name,
                                               value Value, value IsUnsigned) {
  return LLVMDIBuilderCreateEnumerator(
      DIBuilder_val(Builder), String_val(Name), caml_string_length(Name),
      (int64_t)Int_val(Value), Bool_val(IsUnsigned));
}

LLVMMetadataRef llvm_dibuild_create_enumeration_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNumber, value SizeInBits, value AlignInBits, value Elements,
    LLVMMetadataRef ClassTy) {
  return LLVMDIBuilderCreateEnumerationType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNumber), (uint64_t)Int_val(SizeInBits),
      Int_val(AlignInBits), (LLVMMetadataRef *)Op_val(Elements),
      Wosize_val(Elements), ClassTy);
}

LLVMMetadataRef llvm_dibuild_create_enumeration_type_bytecode(value *argv,
                                                              int argn) {
  return llvm_dibuild_create_enumeration_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Scope
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // File
      argv[4],                  // LineNumber
      argv[5],                  // SizeInBits
      argv[6],                  // AlignInBits
      argv[7],                  // Elements
      (LLVMMetadataRef)argv[8]  // ClassTy
  );
}

LLVMMetadataRef llvm_dibuild_create_union_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNumber, value SizeInBits, value AlignInBits, value Flags,
    value Elements, value RunTimeLanguage, value UniqueId) {

  return LLVMDIBuilderCreateUnionType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNumber), (uint64_t)Int_val(SizeInBits),
      Int_val(AlignInBits), DIFlags_val(Flags),
      (LLVMMetadataRef *)Op_val(Elements), Wosize_val(Elements),
      Int_val(RunTimeLanguage), String_val(UniqueId),
      caml_string_length(UniqueId));
}

LLVMMetadataRef llvm_dibuild_create_union_type_bytecode(value *argv, int argn) {
  return llvm_dibuild_create_union_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Scope
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // File
      argv[4],                  // LineNumber
      argv[5],                  // SizeInBits
      argv[6],                  // AlignInBits
      argv[7],                  // Flags
      argv[8],                  // Elements
      argv[9],                  // RunTimeLanguage
      argv[10]                  // UniqueId
  );
}

LLVMMetadataRef llvm_dibuild_create_array_type(value Builder, value Size,
                                               value AlignInBits,
                                               LLVMMetadataRef Ty,
                                               value Subscripts) {
  return LLVMDIBuilderCreateArrayType(
      DIBuilder_val(Builder), (uint64_t)Int_val(Size), Int_val(AlignInBits), Ty,
      (LLVMMetadataRef *)Op_val(Subscripts), Wosize_val(Subscripts));
}

LLVMMetadataRef llvm_dibuild_create_vector_type(value Builder, value Size,
                                                value AlignInBits,
                                                LLVMMetadataRef Ty,
                                                value Subscripts) {
  return LLVMDIBuilderCreateVectorType(
      DIBuilder_val(Builder), (uint64_t)Int_val(Size), Int_val(AlignInBits), Ty,
      (LLVMMetadataRef *)Op_val(Subscripts), Wosize_val(Subscripts));
}

LLVMMetadataRef llvm_dibuild_create_unspecified_type(value Builder,
                                                     value Name) {
  return LLVMDIBuilderCreateUnspecifiedType(
      DIBuilder_val(Builder), String_val(Name), caml_string_length(Name));
}

LLVMMetadataRef llvm_dibuild_create_basic_type(value Builder, value Name,
                                               value SizeInBits, value Encoding,
                                               value Flags) {

  return LLVMDIBuilderCreateBasicType(
      DIBuilder_val(Builder), String_val(Name), caml_string_length(Name),
      (uint64_t)Int_val(SizeInBits), Int_val(Encoding), DIFlags_val(Flags));
}

LLVMMetadataRef llvm_dibuild_create_pointer_type_native(
    value Builder, LLVMMetadataRef PointeeTy, value SizeInBits,
    value AlignInBits, value AddressSpace, value Name) {
  return LLVMDIBuilderCreatePointerType(
      DIBuilder_val(Builder), PointeeTy, (uint64_t)Int_val(SizeInBits),
      Int_val(AlignInBits), Int_val(AddressSpace), String_val(Name),
      caml_string_length(Name));
}

LLVMMetadataRef llvm_dibuild_create_pointer_type_bytecode(value *argv,
                                                          int argn) {
  return llvm_dibuild_create_pointer_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // PointeeTy
      argv[2],                  // SizeInBits
      argv[3],                  // AlignInBits
      argv[4],                  // AddressSpace
      argv[5]                   // Name
  );
}

LLVMMetadataRef llvm_dibuild_create_struct_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNumber, value SizeInBits, value AlignInBits, value Flags,
    LLVMMetadataRef DerivedFrom, value Elements, value RunTimeLanguage,
    LLVMMetadataRef VTableHolder, value UniqueId) {

  return LLVMDIBuilderCreateStructType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNumber), (uint64_t)Int_val(SizeInBits),
      Int_val(AlignInBits), DIFlags_val(Flags), DerivedFrom,
      (LLVMMetadataRef *)Op_val(Elements), Wosize_val(Elements),
      Int_val(RunTimeLanguage), VTableHolder, String_val(UniqueId),
      caml_string_length(UniqueId));
}

LLVMMetadataRef llvm_dibuild_create_struct_type_bytecode(value *argv,
                                                         int argn) {
  return llvm_dibuild_create_struct_type_native(
      argv[0],                   // Builder
      (LLVMMetadataRef)argv[1],  // Scope
      argv[2],                   // Name
      (LLVMMetadataRef)argv[3],  // File
      argv[4],                   // LineNumber
      argv[5],                   // SizeInBits
      argv[6],                   // AlignInBits
      argv[7],                   // Flags
      (LLVMMetadataRef)argv[8],  // DeriviedFrom
      argv[9],                   // Elements
      argv[10],                  // RunTimeLanguage
      (LLVMMetadataRef)argv[11], // VTableHolder
      argv[12]                   // UniqueId
  );
}

LLVMMetadataRef llvm_dibuild_create_member_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNumber, value SizeInBits, value AlignInBits, value OffsetInBits,
    value Flags, LLVMMetadataRef Ty) {

  return LLVMDIBuilderCreateMemberType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNumber), (uint64_t)Int_val(SizeInBits),
      Int_val(AlignInBits), (uint64_t)Int_val(OffsetInBits), DIFlags_val(Flags),
      Ty);
}

LLVMMetadataRef llvm_dibuild_create_member_type_bytecode(value *argv,
                                                         int argn) {
  return llvm_dibuild_create_member_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Scope
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // File
      argv[4],                  // LineNumber
      argv[5],                  // SizeInBits
      argv[6],                  // AlignInBits
      argv[7],                  // OffsetInBits
      argv[8],                  // Flags
      (LLVMMetadataRef)argv[9]  // Ty
  );
}

LLVMMetadataRef llvm_dibuild_create_static_member_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNumber, LLVMMetadataRef Type, value Flags,
    LLVMValueRef ConstantVal, value AlignInBits) {

  return LLVMDIBuilderCreateStaticMemberType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNumber), Type, DIFlags_val(Flags), ConstantVal,
      Int_val(AlignInBits));
}

LLVMMetadataRef llvm_dibuild_create_static_member_type_bytecode(value *argv,
                                                                int argn) {
  return llvm_dibuild_create_static_member_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Scope
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // File
      argv[4],                  // LineNumber
      (LLVMMetadataRef)argv[5], // Type
      argv[6],                  // Flags,
      (LLVMValueRef)argv[7],    // ConstantVal
      argv[8]                   // AlignInBits
  );
}

LLVMMetadataRef llvm_dibuild_create_member_pointer_type_native(
    value Builder, LLVMMetadataRef PointeeType, LLVMMetadataRef ClassType,
    uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags) {

  return LLVMDIBuilderCreateMemberPointerType(
      DIBuilder_val(Builder), PointeeType, ClassType,
      (uint64_t)Int_val(SizeInBits), Int_val(AlignInBits), Flags);
}

LLVMMetadataRef llvm_dibuild_create_member_pointer_type_bytecode(value *argv,
                                                                 int argn) {
  return llvm_dibuild_create_member_pointer_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // PointeeType
      (LLVMMetadataRef)argv[2], // ClassType
      argv[3],                  // SizeInBits
      argv[4],                  // AlignInBits
      argv[5]                   // Flags
  );
}

LLVMMetadataRef llvm_dibuild_create_object_pointer_type(value Builder,
                                                        LLVMMetadataRef Type) {
  return LLVMDIBuilderCreateObjectPointerType(DIBuilder_val(Builder), Type);
}

LLVMMetadataRef llvm_dibuild_create_qualified_type(value Builder, value Tag,
                                                   LLVMMetadataRef Type) {

  return LLVMDIBuilderCreateQualifiedType(DIBuilder_val(Builder), Int_val(Tag),
                                          Type);
}

LLVMMetadataRef llvm_dibuild_create_reference_type(value Builder, value Tag,
                                                   LLVMMetadataRef Type) {

  return LLVMDIBuilderCreateReferenceType(DIBuilder_val(Builder), Int_val(Tag),
                                          Type);
}

LLVMMetadataRef llvm_dibuild_create_null_ptr_type(value Builder) {

  return LLVMDIBuilderCreateNullPtrType(DIBuilder_val(Builder));
}

LLVMMetadataRef llvm_dibuild_create_typedef_native(
    value Builder, LLVMMetadataRef Type, value Name, LLVMMetadataRef File,
    value LineNo, LLVMMetadataRef Scope, value AlignInBits) {

  return LLVMDIBuilderCreateTypedef(
      DIBuilder_val(Builder), Type, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNo), Scope, Int_val(AlignInBits));
}

LLVMMetadataRef llvm_dibuild_create_typedef_bytecode(value *argv, int argn) {

  return llvm_dibuild_create_typedef_native(argv[0],                  // Builder
                                            (LLVMMetadataRef)argv[1], // Type
                                            argv[2],                  // Name
                                            (LLVMMetadataRef)argv[3], // File
                                            argv[4],                  // LineNo
                                            (LLVMMetadataRef)argv[5], // Scope
                                            argv[6] // AlignInBits
  );
}

LLVMMetadataRef
llvm_dibuild_create_inheritance_native(value Builder, LLVMMetadataRef Ty,
                                       LLVMMetadataRef BaseTy, value BaseOffset,
                                       value VBPtrOffset, value Flags) {

  return LLVMDIBuilderCreateInheritance(DIBuilder_val(Builder), Ty, BaseTy,
                                        (uint64_t)Int_val(BaseOffset),
                                        Int_val(VBPtrOffset), Flags);
}

LLVMMetadataRef llvm_dibuild_create_inheritance_bytecode(value *argv, int arg) {

  return llvm_dibuild_create_inheritance_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Ty
      (LLVMMetadataRef)argv[2], // BaseTy
      argv[3],                  // BaseOffset
      argv[4],                  // VBPtrOffset
      argv[5]                   // Flags
  );
}

LLVMMetadataRef llvm_dibuild_create_forward_decl_native(
    value Builder, value Tag, value Name, LLVMMetadataRef Scope,
    LLVMMetadataRef File, value Line, value RuntimeLang, value SizeInBits,
    value AlignInBits, value UniqueIdentifier) {
  return LLVMDIBuilderCreateForwardDecl(
      DIBuilder_val(Builder), Int_val(Tag), String_val(Name),
      caml_string_length(Name), Scope, File, Int_val(Line),
      Int_val(RuntimeLang), (uint64_t)Int_val(SizeInBits), Int_val(AlignInBits),
      String_val(UniqueIdentifier), caml_string_length(UniqueIdentifier));
}

LLVMMetadataRef llvm_dibuild_create_forward_decl_bytecode(value *argv,
                                                          int arg) {

  return llvm_dibuild_create_forward_decl_native(
      argv[0],                  // Builder
      argv[1],                  // Tag
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // Scope
      (LLVMMetadataRef)argv[4], // File
      argv[5],                  // Line
      argv[6],                  // RuntimeLang
      argv[7],                  // SizeInBits
      argv[8],                  // AlignInBits
      argv[9]                   // UniqueIdentifier
  );
}

LLVMMetadataRef llvm_dibuild_create_replaceable_composite_type_native(
    value Builder, value Tag, value Name, LLVMMetadataRef Scope,
    LLVMMetadataRef File, value Line, value RuntimeLang, value SizeInBits,
    value AlignInBits, value Flags, value UniqueIdentifier) {

  return LLVMDIBuilderCreateReplaceableCompositeType(
      DIBuilder_val(Builder), Int_val(Tag), String_val(Name),
      caml_string_length(Name), Scope, File, Int_val(Line),
      Int_val(RuntimeLang), (uint64_t)Int_val(SizeInBits), Int_val(AlignInBits),
      DIFlags_val(Flags), String_val(UniqueIdentifier),
      caml_string_length(UniqueIdentifier));
}

LLVMMetadataRef
llvm_dibuild_create_replaceable_composite_type_bytecode(value *argv, int arg) {

  return llvm_dibuild_create_replaceable_composite_type_native(
      argv[0],                  // Builder
      argv[1],                  // Tag
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // Scope
      (LLVMMetadataRef)argv[4], // File
      argv[5],                  // Line
      argv[6],                  // RuntimeLang
      argv[7],                  // SizeInBits
      argv[8],                  // AlignInBits
      argv[9],                  // Flags
      argv[10]                  // UniqueIdentifier
  );
}

LLVMMetadataRef llvm_dibuild_create_bit_field_member_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNum, value SizeInBits, value OffsetInBits,
    value StorageOffsetInBits, value Flags, LLVMMetadataRef Ty) {

  return LLVMDIBuilderCreateBitFieldMemberType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNum), (uint64_t)Int_val(SizeInBits),
      (uint64_t)Int_val(OffsetInBits), (uint64_t)Int_val(StorageOffsetInBits),
      DIFlags_val(Flags), Ty);
}

LLVMMetadataRef llvm_dibuild_create_bit_field_member_type_bytecode(value *argv,
                                                                   int arg) {

  return llvm_dibuild_create_bit_field_member_type_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Scope
      argv[2],                  // Name
      (LLVMMetadataRef)argv[3], // File
      argv[4],                  // LineNum
      argv[5],                  // SizeInBits
      argv[6],                  // OffsetInBits
      argv[7],                  // StorageOffsetInBits
      argv[8],                  // Flags
      (LLVMMetadataRef)argv[9]  // Ty
  );
}

LLVMMetadataRef llvm_dibuild_create_class_type_native(
    value Builder, LLVMMetadataRef Scope, value Name, LLVMMetadataRef File,
    value LineNumber, value SizeInBits, value AlignInBits, value OffsetInBits,
    value Flags, LLVMMetadataRef DerivedFrom, value Elements,
    LLVMMetadataRef VTableHolder, LLVMMetadataRef TemplateParamsNode,
    value UniqueIdentifier) {

  return LLVMDIBuilderCreateClassType(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      File, Int_val(LineNumber), (uint64_t)Int_val(SizeInBits),
      Int_val(AlignInBits), (uint64_t)Int_val(OffsetInBits), DIFlags_val(Flags),
      DerivedFrom, (LLVMMetadataRef *)Op_val(Elements), Wosize_val(Elements),
      VTableHolder, TemplateParamsNode, String_val(UniqueIdentifier),
      caml_string_length(UniqueIdentifier));
}

LLVMMetadataRef llvm_dibuild_create_class_type_bytecode(value *argv, int arg) {

  return llvm_dibuild_create_class_type_native(
      argv[0],                   // Builder
      (LLVMMetadataRef)argv[1],  // Scope
      argv[2],                   // Name
      (LLVMMetadataRef)argv[3],  // File
      argv[4],                   // LineNumber
      argv[5],                   // SizeInBits
      argv[6],                   // AlignInBits
      argv[7],                   // OffsetInBits
      argv[8],                   // Flags
      (LLVMMetadataRef)argv[9],  // DerivedFrom
      argv[10],                  // Elements
      (LLVMMetadataRef)argv[11], // VTableHolder
      (LLVMMetadataRef)argv[12], // TemplateParamsNode
      argv[13]                   // UniqueIdentifier
  );
}

LLVMMetadataRef llvm_dibuild_create_artificial_type(value Builder,
                                                    LLVMMetadataRef Type) {
  return LLVMDIBuilderCreateArtificialType(DIBuilder_val(Builder), Type);
}

value llvm_di_type_get_name(LLVMMetadataRef DType) {
  size_t Len;
  const char *Name = LLVMDITypeGetName(DType, &Len);
  return cstr_to_string(Name, Len);
}

value llvm_di_type_get_size_in_bits(LLVMMetadataRef DType) {
  uint64_t Size = LLVMDITypeGetSizeInBits(DType);
  return Val_int((int)Size);
}

value llvm_di_type_get_offset_in_bits(LLVMMetadataRef DType) {
  uint64_t Size = LLVMDITypeGetOffsetInBits(DType);
  return Val_int((int)Size);
}

value llvm_di_type_get_align_in_bits(LLVMMetadataRef DType) {
  uint32_t Size = LLVMDITypeGetAlignInBits(DType);
  return Val_int(Size);
}

value llvm_di_type_get_line(LLVMMetadataRef DType) {
  unsigned Line = LLVMDITypeGetLine(DType);
  return Val_int(Line);
}

value llvm_di_type_get_flags(LLVMMetadataRef DType) {
  LLVMDIFlags Flags = LLVMDITypeGetLine(DType);
  return alloc_diflags(Flags);
}

value llvm_get_subprogram(LLVMValueRef Func) {
  return (ptr_to_option(LLVMGetSubprogram(Func)));
}

value llvm_set_subprogram(LLVMValueRef Func, LLVMMetadataRef SP) {
  LLVMSetSubprogram(Func, SP);
  return Val_unit;
}

value llvm_di_subprogram_get_line(LLVMMetadataRef Subprogram) {
  return Val_int(LLVMDISubprogramGetLine(Subprogram));
}

value llvm_instr_get_debug_loc(LLVMValueRef Inst) {
  return (ptr_to_option(LLVMInstructionGetDebugLoc(Inst)));
}

value llvm_instr_set_debug_loc(LLVMValueRef Inst, LLVMMetadataRef Loc) {
  LLVMInstructionSetDebugLoc(Inst, Loc);
  return Val_unit;
}

LLVMMetadataRef llvm_dibuild_create_constant_value_expression(value Builder,
                                                              value Value) {
  return LLVMDIBuilderCreateConstantValueExpression(DIBuilder_val(Builder),
                                                    (int64_t)Int_val(Value));
}

LLVMMetadataRef llvm_dibuild_create_global_variable_expression_native(
    value Builder, LLVMMetadataRef Scope, value Name, value Linkage,
    LLVMMetadataRef File, value Line, LLVMMetadataRef Ty, value LocalToUnit,
    LLVMMetadataRef Expr, LLVMMetadataRef Decl, value AlignInBits) {
  return LLVMDIBuilderCreateGlobalVariableExpression(
      DIBuilder_val(Builder), Scope, String_val(Name), caml_string_length(Name),
      String_val(Linkage), caml_string_length(Linkage), File, Int_val(Line), Ty,
      Bool_val(LocalToUnit), Expr, Decl, Int_val(AlignInBits));
}

LLVMMetadataRef
llvm_dibuild_create_global_variable_expression_bytecode(value *argv, int arg) {

  return llvm_dibuild_create_global_variable_expression_native(
      argv[0],                  // Builder
      (LLVMMetadataRef)argv[1], // Scope
      argv[2],                  // Name
      argv[3],                  // Linkage
      (LLVMMetadataRef)argv[4], // File
      argv[5],                  // Line
      (LLVMMetadataRef)argv[6], // Ty
      argv[7],                  // LocalToUnit
      (LLVMMetadataRef)argv[8], // Expr
      (LLVMMetadataRef)argv[9], // Decl
      argv[10]                  // AlignInBits
  );
}

value llvm_di_global_variable_expression_get_variable(LLVMMetadataRef GVE) {
  return (ptr_to_option(LLVMDIGlobalVariableExpressionGetVariable(GVE)));
}

value llvm_di_variable_get_line(LLVMMetadataRef Variable) {
  return Val_int(LLVMDIVariableGetLine(Variable));
}

value llvm_di_variable_get_file(LLVMMetadataRef Variable) {
  return (ptr_to_option(LLVMDIVariableGetFile(Variable)));
}

value llvm_get_metadata_kind(LLVMMetadataRef Metadata) {
  return Val_int(LLVMGetMetadataKind(Metadata));
}
