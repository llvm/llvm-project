//===-- AMDGPURuntimeMetadata.h - AMDGPU Runtime Metadata -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// Enums and structure types used by runtime metadata.
///
/// Runtime requests certain information (metadata) about kernels to be able
/// to execute the kernels and answer the queries about the kernels.
/// The metadata is represented as a byte stream in an ELF section of a
/// binary (code object). The byte stream consists of key-value pairs.
/// Each key is an 8 bit unsigned integer. Each value can be an integer,
/// a string, or a stream of key-value pairs. There are 3 levels of key-value
/// pair streams. At the beginning of the ELF section is the top level
/// key-value pair stream. A kernel-level key-value pair stream starts after
/// encountering KeyKernelBegin and ends immediately before encountering
/// KeyKernelEnd. A kernel-argument-level key-value pair stream starts
/// after encountering KeyArgBegin and ends immediately before encountering
/// KeyArgEnd. A kernel-level key-value pair stream can only appear in a top
/// level key-value pair stream. A kernel-argument-level key-value pair stream
/// can only appear in a kernel-level key-value pair stream.
///
/// The format should be kept backward compatible. New enum values and bit
/// fields should be appended at the end. It is suggested to bump up the
/// revision number whenever the format changes and document the change
/// in the revision in this header.
///
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPURUNTIMEMETADATA_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPURUNTIMEMETADATA_H

#include <stdint.h>

namespace AMDGPU {

namespace RuntimeMD {

  // Version and revision of runtime metadata
  const unsigned char MDVersion   = 1;
  const unsigned char MDRevision  = 0;

  // ELF section name containing runtime metadata
  const char SectionName[] = ".AMDGPU.runtime_metadata";

  // Enumeration values of keys in runtime metadata.
  enum Key {
    KeyNull                     = 0, // Place holder. Ignored when encountered
    KeyMDVersion                = 1, // Runtime metadata version
    KeyLanguage                 = 2, // Language
    KeyLanguageVersion          = 3, // Language version
    KeyKernelBegin              = 4, // Beginning of kernel-level stream
    KeyKernelEnd                = 5, // End of kernel-level stream
    KeyKernelName               = 6, // Kernel name
    KeyArgBegin                 = 7, // Beginning of kernel-arg-level stream
    KeyArgEnd                   = 8, // End of kernel-arg-level stream
    KeyArgSize                  = 9, // Kernel arg size
    KeyArgAlign                 = 10, // Kernel arg alignment
    KeyArgTypeName              = 11, // Kernel type name
    KeyArgName                  = 12, // Kernel name
    KeyArgTypeKind              = 13, // Kernel argument type kind
    KeyArgValueType             = 14, // Kernel argument value type
    KeyArgAddrQual              = 15, // Kernel argument address qualifier
    KeyArgAccQual               = 16, // Kernel argument access qualifier
    KeyArgIsConst               = 17, // Kernel argument is const qualified
    KeyArgIsRestrict            = 18, // Kernel argument is restrict qualified
    KeyArgIsVolatile            = 19, // Kernel argument is volatile qualified
    KeyArgIsPipe                = 20, // Kernel argument is pipe qualified
    KeyReqdWorkGroupSize        = 21, // Required work group size
    KeyWorkGroupSizeHint        = 22, // Work group size hint
    KeyVecTypeHint              = 23, // Vector type hint
    KeyKernelIndex              = 24, // Kernel index for device enqueue
    KeySGPRs                    = 25, // Number of SGPRs
    KeyVGPRs                    = 26, // Number of VGPRs
    KeyMinWavesPerSIMD          = 27, // Minimum number of waves per SIMD
    KeyMaxWavesPerSIMD          = 28, // Maximum number of waves per SIMD
    KeyFlatWorkGroupSizeLimits  = 29, // Flat work group size limits
    KeyMaxWorkGroupSize         = 30, // Maximum work group size
    KeyNoPartialWorkGroups      = 31, // No partial work groups
    KeyArgPointeeAlign          = 32, // Alignment of pointee type
  };

  enum Language : uint8_t {
    OpenCL_C      = 0,
    HCC           = 1,
    OpenMP        = 2,
    OpenCL_CPP    = 3,
};

  enum LanguageVersion : uint16_t {
    V100          = 100,
    V110          = 110,
    V120          = 120,
    V200          = 200,
    V210          = 210,
  };

  namespace KernelArg {
    enum TypeKind : uint8_t {
      Value     = 0,
      Pointer   = 1,
      Image     = 2,
      Sampler   = 3,
      Queue     = 4,
    };

    enum ValueType : uint16_t {
      Struct  = 0,
      I8      = 1,
      U8      = 2,
      I16     = 3,
      U16     = 4,
      F16     = 5,
      I32     = 6,
      U32     = 7,
      F32     = 8,
      I64     = 9,
      U64     = 10,
      F64     = 11,
    };

    enum AccessQualifer : uint8_t {
      None       = 0,
      ReadOnly   = 1,
      WriteOnly  = 2,
      ReadWrite  = 3,
    };

    enum AddressSpaceQualifer : uint8_t {
      Private    = 0,
      Global     = 1,
      Constant   = 2,
      Local      = 3,
    };
  } // namespace KernelArg
} // namespace RuntimeMD
} // namespace AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPURUNTIMEMETADATA_H
