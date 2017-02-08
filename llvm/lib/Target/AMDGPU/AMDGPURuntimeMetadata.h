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
/// The metadata is represented as a note element in the .note ELF section of a
/// binary (code object). The desc field of the note element is a YAML string
/// consisting of key-value pairs. Each key is a string. Each value can be
/// an integer, a string, or an YAML sequence. There are 3 levels of YAML maps.
/// At the beginning of the YAML string is the module level YAML map. A
/// kernel-level YAML map is in the amd.Kernels sequence. A
/// kernel-argument-level map is in the amd.Args sequence.
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

#include <cstdint>
#include <vector>
#include <string>

namespace AMDGPU {
namespace RuntimeMD {

  // Version and revision of runtime metadata
  const unsigned char MDVersion   = 2;
  const unsigned char MDRevision  = 1;

  // Name of keys for runtime metadata.
  namespace KeyName {

    // Runtime metadata version
    const char MDVersion[] = "amd.MDVersion";

    // Instruction set architecture information
    const char IsaInfo[] = "amd.IsaInfo";
    // Wavefront size
    const char IsaInfoWavefrontSize[] = "amd.IsaInfoWavefrontSize";
    // Local memory size in bytes
    const char IsaInfoLocalMemorySize[] = "amd.IsaInfoLocalMemorySize";
    // Number of execution units per compute unit
    const char IsaInfoEUsPerCU[] = "amd.IsaInfoEUsPerCU";
    // Maximum number of waves per execution unit
    const char IsaInfoMaxWavesPerEU[] = "amd.IsaInfoMaxWavesPerEU";
    // Maximum flat work group size
    const char IsaInfoMaxFlatWorkGroupSize[] = "amd.IsaInfoMaxFlatWorkGroupSize";
    // SGPR allocation granularity
    const char IsaInfoSGPRAllocGranule[] = "amd.IsaInfoSGPRAllocGranule";
    // Total number of SGPRs
    const char IsaInfoTotalNumSGPRs[] = "amd.IsaInfoTotalNumSGPRs";
    // Addressable number of SGPRs
    const char IsaInfoAddressableNumSGPRs[] = "amd.IsaInfoAddressableNumSGPRs";
    // VGPR allocation granularity
    const char IsaInfoVGPRAllocGranule[] = "amd.IsaInfoVGPRAllocGranule";
    // Total number of VGPRs
    const char IsaInfoTotalNumVGPRs[] = "amd.IsaInfoTotalNumVGPRs";
    // Addressable number of VGPRs
    const char IsaInfoAddressableNumVGPRs[] = "amd.IsaInfoAddressableNumVGPRs";

    // Language
    const char Language[] = "amd.Language";
    // Language version
    const char LanguageVersion[] = "amd.LanguageVersion";

    // Kernels
    const char Kernels[] = "amd.Kernels";
    // Kernel name
    const char KernelName[] = "amd.KernelName";
    // Kernel arguments
    const char Args[] = "amd.Args";
    // Kernel argument size in bytes
    const char ArgSize[] = "amd.ArgSize";
    // Kernel argument alignment
    const char ArgAlign[] = "amd.ArgAlign";
    // Kernel argument type name
    const char ArgTypeName[] = "amd.ArgTypeName";
    // Kernel argument name
    const char ArgName[] = "amd.ArgName";
    // Kernel argument kind
    const char ArgKind[] = "amd.ArgKind";
    // Kernel argument value type
    const char ArgValueType[] = "amd.ArgValueType";
    // Kernel argument address qualifier
    const char ArgAddrQual[] = "amd.ArgAddrQual";
    // Kernel argument access qualifier
    const char ArgAccQual[] = "amd.ArgAccQual";
    // Kernel argument is const qualified
    const char ArgIsConst[] = "amd.ArgIsConst";
    // Kernel argument is restrict qualified
    const char ArgIsRestrict[] = "amd.ArgIsRestrict";
    // Kernel argument is volatile qualified
    const char ArgIsVolatile[] = "amd.ArgIsVolatile";
    // Kernel argument is pipe qualified
    const char ArgIsPipe[] = "amd.ArgIsPipe";
    // Required work group size
    const char ReqdWorkGroupSize[] = "amd.ReqdWorkGroupSize";
    // Work group size hint
    const char WorkGroupSizeHint[] = "amd.WorkGroupSizeHint";
    // Vector type hint
    const char VecTypeHint[] = "amd.VecTypeHint";
    // Kernel index for device enqueue
    const char KernelIndex[] = "amd.KernelIndex";
    // No partial work groups
    const char NoPartialWorkGroups[] = "amd.NoPartialWorkGroups";
    // Prinf function call information
    const char PrintfInfo[] = "amd.PrintfInfo";
    // The actual kernel argument access qualifier
    const char ArgActualAcc[] = "amd.ArgActualAcc";
    // Alignment of pointee type
    const char ArgPointeeAlign[] = "amd.ArgPointeeAlign";

  } // end namespace KeyName

  namespace KernelArg {

    enum Kind : uint8_t {
      ByValue                 = 0,
      GlobalBuffer            = 1,
      DynamicSharedPointer    = 2,
      Sampler                 = 3,
      Image                   = 4,
      Pipe                    = 5,
      Queue                   = 6,
      HiddenGlobalOffsetX     = 7,
      HiddenGlobalOffsetY     = 8,
      HiddenGlobalOffsetZ     = 9,
      HiddenNone              = 10,
      HiddenPrintfBuffer      = 11,
      HiddenDefaultQueue      = 12,
      HiddenCompletionAction  = 13,
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

    // Avoid using 'None' since it conflicts with a macro in X11 header file.
    enum AccessQualifer : uint8_t {
      AccNone    = 0,
      ReadOnly   = 1,
      WriteOnly  = 2,
      ReadWrite  = 3,
    };

    enum AddressSpaceQualifer : uint8_t {
      Private    = 0,
      Global     = 1,
      Constant   = 2,
      Local      = 3,
      Generic    = 4,
      Region     = 5,
    };

  } // end namespace KernelArg

  // Invalid values are used to indicate an optional key should not be emitted.
  const uint8_t INVALID_ADDR_QUAL     = 0xff;
  const uint8_t INVALID_ACC_QUAL      = 0xff;
  const uint32_t INVALID_KERNEL_INDEX = ~0U;

  namespace KernelArg {

    // In-memory representation of kernel argument information.
    struct Metadata {
      uint32_t Size = 0;
      uint32_t Align = 0;
      uint32_t PointeeAlign = 0;
      uint8_t Kind = 0;
      uint16_t ValueType = 0;
      std::string TypeName;
      std::string Name;
      uint8_t AddrQual = INVALID_ADDR_QUAL;
      uint8_t AccQual = INVALID_ACC_QUAL;
      uint8_t IsVolatile = 0;
      uint8_t IsConst = 0;
      uint8_t IsRestrict = 0;
      uint8_t IsPipe = 0;

      Metadata() = default;
    };

  } // end namespace KernelArg

  namespace Kernel {

    // In-memory representation of kernel information.
    struct Metadata {
      std::string Name;
      std::string Language;
      std::vector<uint8_t> LanguageVersion;
      std::vector<uint32_t> ReqdWorkGroupSize;
      std::vector<uint32_t> WorkGroupSizeHint;
      std::string VecTypeHint;
      uint32_t KernelIndex = INVALID_KERNEL_INDEX;
      uint8_t NoPartialWorkGroups = 0;
      std::vector<KernelArg::Metadata> Args;

      Metadata() = default;
    };

  } // end namespace Kernel

  namespace IsaInfo {

    /// \brief In-memory representation of instruction set architecture
    /// information.
    struct Metadata {
      /// \brief Wavefront size.
      unsigned WavefrontSize = 0;
      /// \brief Local memory size in bytes.
      unsigned LocalMemorySize = 0;
      /// \brief Number of execution units per compute unit.
      unsigned EUsPerCU = 0;
      /// \brief Maximum number of waves per execution unit.
      unsigned MaxWavesPerEU = 0;
      /// \brief Maximum flat work group size.
      unsigned MaxFlatWorkGroupSize = 0;
      /// \brief SGPR allocation granularity.
      unsigned SGPRAllocGranule = 0;
      /// \brief Total number of SGPRs.
      unsigned TotalNumSGPRs = 0;
      /// \brief Addressable number of SGPRs.
      unsigned AddressableNumSGPRs = 0;
      /// \brief VGPR allocation granularity.
      unsigned VGPRAllocGranule = 0;
      /// \brief Total number of VGPRs.
      unsigned TotalNumVGPRs = 0;
      /// \brief Addressable number of VGPRs.
      unsigned AddressableNumVGPRs = 0;

      Metadata() = default;
    };

  } // end namespace IsaInfo

  namespace Program {

    // In-memory representation of program information.
    struct Metadata {
      std::vector<uint8_t> MDVersionSeq;
      IsaInfo::Metadata IsaInfo;
      std::vector<std::string> PrintfInfo;
      std::vector<Kernel::Metadata> Kernels;

      explicit Metadata() = default;

      // Construct from an YAML string.
      explicit Metadata(const std::string &YAML);

      // Convert to YAML string.
      std::string toYAML();

      // Convert from YAML string.
      static Metadata fromYAML(const std::string &S);
    };

  } //end namespace Program

} // end namespace RuntimeMD
} // end namespace AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPURUNTIMEMETADATA_H
