#include "llvm/Support/ARMAttributeParser.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/ELFAttributes.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

struct AttributeSection {
  unsigned Tag;
  unsigned Value;

  AttributeSection(unsigned tag, unsigned value) : Tag(tag), Value(value) { }

  void write(raw_ostream &OS) {
    OS.flush();
    // length = length + "aeabi\0" + TagFile + ByteSize + Tag + Value;
    // length = 17 bytes

    OS << 'A' << (uint8_t)17 << (uint8_t)0 << (uint8_t)0 << (uint8_t)0;
    OS << "aeabi" << '\0';
    OS << (uint8_t)1 << (uint8_t)7 << (uint8_t)0 << (uint8_t)0 << (uint8_t)0;
    OS << (uint8_t)Tag << (uint8_t)Value;

  }
};

bool testBuildAttr(unsigned Tag, unsigned Value,
                   unsigned ExpectedTag, unsigned ExpectedValue) {
  std::string buffer;
  raw_string_ostream OS(buffer);
  AttributeSection Section(Tag, Value);
  Section.write(OS);
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(buffer.c_str()),
                          buffer.size());

  ARMAttributeParser Parser;
  cantFail(Parser.parse(Bytes, llvm::endianness::little));

  std::optional<unsigned> Attr = Parser.getAttributeValue(ExpectedTag);
  return Attr && *Attr == ExpectedValue;
}

void testParseError(ArrayRef<uint8_t> bytes, const char *msg) {
  ARMAttributeParser parser;
  Error e = parser.parse(bytes, llvm::endianness::little);
  EXPECT_STREQ(toString(std::move(e)).c_str(), msg);
}

bool testTagString(unsigned Tag, const char *name) {
  return ELFAttrs::attrTypeAsString(Tag, ARMBuildAttrs::getARMAttributeTags())
             .str() == name;
}

TEST(ARMAttributeParser, UnknownCPU_arch) {
  static const uint8_t bytes[] = {'A', 15, 0, 0, 0, 'a', 'e', 'a', 'b',
                                  'i', 0,  1, 7, 0, 0,   0,   6,   23};
  testParseError(bytes, "unknown CPU_arch value: 23");
}

TEST(CPUArchBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(6, "Tag_CPU_arch"));

  EXPECT_TRUE(testBuildAttr(6, 0, ARMBuildAttrs::CPU_arch,
                            ARMBuildAttrs::Pre_v4));
  EXPECT_TRUE(testBuildAttr(6, 1, ARMBuildAttrs::CPU_arch,
                            ARMBuildAttrs::v4));
  EXPECT_TRUE(testBuildAttr(6, 2, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v4T));
  EXPECT_TRUE(testBuildAttr(6, 3, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v5T));
  EXPECT_TRUE(testBuildAttr(6, 4, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v5TE));
  EXPECT_TRUE(testBuildAttr(6, 5, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v5TEJ));
  EXPECT_TRUE(testBuildAttr(6, 6, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v6));
  EXPECT_TRUE(testBuildAttr(6, 7, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v6KZ));
  EXPECT_TRUE(testBuildAttr(6, 8, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v6T2));
  EXPECT_TRUE(testBuildAttr(6, 9, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v6K));
  EXPECT_TRUE(testBuildAttr(6, 10, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v7));
  EXPECT_TRUE(testBuildAttr(6, 11, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v6_M));
  EXPECT_TRUE(testBuildAttr(6, 12, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v6S_M));
  EXPECT_TRUE(testBuildAttr(6, 13, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v7E_M));
  EXPECT_TRUE(testBuildAttr(6, 14, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v8_A));
  EXPECT_TRUE(testBuildAttr(6, 15, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v8_R));
  EXPECT_TRUE(testBuildAttr(6, 16, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v8_M_Base));
  EXPECT_TRUE(testBuildAttr(6, 17, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v8_M_Main));
  EXPECT_TRUE(testBuildAttr(6, 21, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v8_1_M_Main));
  EXPECT_TRUE(testBuildAttr(6, 22, ARMBuildAttrs::CPU_arch,
                               ARMBuildAttrs::v9_A));

}

TEST(CPUArchProfileBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(7, "Tag_CPU_arch_profile"));
  EXPECT_TRUE(testBuildAttr(7, 'A', ARMBuildAttrs::CPU_arch_profile,
                               ARMBuildAttrs::ApplicationProfile));
  EXPECT_TRUE(testBuildAttr(7, 'R', ARMBuildAttrs::CPU_arch_profile,
                               ARMBuildAttrs::RealTimeProfile));
  EXPECT_TRUE(testBuildAttr(7, 'M', ARMBuildAttrs::CPU_arch_profile,
                               ARMBuildAttrs::MicroControllerProfile));
  EXPECT_TRUE(testBuildAttr(7, 'S', ARMBuildAttrs::CPU_arch_profile,
                               ARMBuildAttrs::SystemProfile));
}

TEST(ARMISABuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(8, "Tag_ARM_ISA_use"));
  EXPECT_TRUE(testBuildAttr(8, 0, ARMBuildAttrs::ARM_ISA_use,
                               ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(8, 1, ARMBuildAttrs::ARM_ISA_use,
                               ARMBuildAttrs::Allowed));
}

TEST(ThumbISABuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(9, "Tag_THUMB_ISA_use"));
  EXPECT_TRUE(testBuildAttr(9, 0, ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(9, 1, ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::Allowed));
  EXPECT_TRUE(testBuildAttr(9, 2, ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::AllowThumb32));
  EXPECT_TRUE(testBuildAttr(9, 3, ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::AllowThumbDerived));
}

TEST(FPArchBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(10, "Tag_FP_arch"));
  EXPECT_TRUE(testBuildAttr(10, 0, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(10, 1, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::Allowed));
  EXPECT_TRUE(testBuildAttr(10, 2, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPv2));
  EXPECT_TRUE(testBuildAttr(10, 3, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPv3A));
  EXPECT_TRUE(testBuildAttr(10, 4, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPv3B));
  EXPECT_TRUE(testBuildAttr(10, 5, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPv4A));
  EXPECT_TRUE(testBuildAttr(10, 6, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPv4B));
  EXPECT_TRUE(testBuildAttr(10, 7, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPARMv8A));
  EXPECT_TRUE(testBuildAttr(10, 8, ARMBuildAttrs::FP_arch,
                               ARMBuildAttrs::AllowFPARMv8B));
}

TEST(WMMXBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(11, "Tag_WMMX_arch"));
  EXPECT_TRUE(testBuildAttr(11, 0, ARMBuildAttrs::WMMX_arch,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(11, 1, ARMBuildAttrs::WMMX_arch,
                            ARMBuildAttrs::AllowWMMXv1));
  EXPECT_TRUE(testBuildAttr(11, 2, ARMBuildAttrs::WMMX_arch,
                            ARMBuildAttrs::AllowWMMXv2));
}

TEST(SIMDBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(12, "Tag_Advanced_SIMD_arch"));
  EXPECT_TRUE(testBuildAttr(12, 0, ARMBuildAttrs::Advanced_SIMD_arch,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(12, 1, ARMBuildAttrs::Advanced_SIMD_arch,
                            ARMBuildAttrs::AllowNeon));
  EXPECT_TRUE(testBuildAttr(12, 2, ARMBuildAttrs::Advanced_SIMD_arch,
                            ARMBuildAttrs::AllowNeon2));
  EXPECT_TRUE(testBuildAttr(12, 3, ARMBuildAttrs::Advanced_SIMD_arch,
                            ARMBuildAttrs::AllowNeonARMv8));
  EXPECT_TRUE(testBuildAttr(12, 4, ARMBuildAttrs::Advanced_SIMD_arch,
                            ARMBuildAttrs::AllowNeonARMv8_1a));
}

TEST(FPHPBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(36, "Tag_FP_HP_extension"));
  EXPECT_TRUE(testBuildAttr(36, 0, ARMBuildAttrs::FP_HP_extension,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(36, 1, ARMBuildAttrs::FP_HP_extension,
                            ARMBuildAttrs::AllowHPFP));
}

TEST(MVEBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(48, "Tag_MVE_arch"));
  EXPECT_TRUE(testBuildAttr(48, 0, ARMBuildAttrs::MVE_arch,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(48, 1, ARMBuildAttrs::MVE_arch,
                            ARMBuildAttrs::AllowMVEInteger));
  EXPECT_TRUE(testBuildAttr(48, 2, ARMBuildAttrs::MVE_arch,
                            ARMBuildAttrs::AllowMVEIntegerAndFloat));
}

TEST(CPUAlignBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(34, "Tag_CPU_unaligned_access"));
  EXPECT_TRUE(testBuildAttr(34, 0, ARMBuildAttrs::CPU_unaligned_access,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(34, 1, ARMBuildAttrs::CPU_unaligned_access,
                            ARMBuildAttrs::Allowed));
}

TEST(T2EEBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(66, "Tag_T2EE_use"));
  EXPECT_TRUE(testBuildAttr(66, 0, ARMBuildAttrs::T2EE_use,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(66, 1, ARMBuildAttrs::T2EE_use,
                            ARMBuildAttrs::Allowed));
}

TEST(VirtualizationBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(68, "Tag_Virtualization_use"));
  EXPECT_TRUE(testBuildAttr(68, 0, ARMBuildAttrs::Virtualization_use,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(68, 1, ARMBuildAttrs::Virtualization_use,
                            ARMBuildAttrs::AllowTZ));
  EXPECT_TRUE(testBuildAttr(68, 2, ARMBuildAttrs::Virtualization_use,
                            ARMBuildAttrs::AllowVirtualization));
  EXPECT_TRUE(testBuildAttr(68, 3, ARMBuildAttrs::Virtualization_use,
                            ARMBuildAttrs::AllowTZVirtualization));
}

TEST(MPBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(42, "Tag_MPextension_use"));
  EXPECT_TRUE(testBuildAttr(42, 0, ARMBuildAttrs::MPextension_use,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(42, 1, ARMBuildAttrs::MPextension_use,
                            ARMBuildAttrs::AllowMP));
}

TEST(DivBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(44, "Tag_DIV_use"));
  EXPECT_TRUE(testBuildAttr(44, 0, ARMBuildAttrs::DIV_use,
                            ARMBuildAttrs::AllowDIVIfExists));
  EXPECT_TRUE(testBuildAttr(44, 1, ARMBuildAttrs::DIV_use,
                            ARMBuildAttrs::DisallowDIV));
  EXPECT_TRUE(testBuildAttr(44, 2, ARMBuildAttrs::DIV_use,
                            ARMBuildAttrs::AllowDIVExt));
}

TEST(PCS_ConfigBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(13, "Tag_PCS_config"));
  EXPECT_TRUE(testBuildAttr(13, 0, ARMBuildAttrs::PCS_config, 0));
  EXPECT_TRUE(testBuildAttr(13, 1, ARMBuildAttrs::PCS_config, 1));
  EXPECT_TRUE(testBuildAttr(13, 2, ARMBuildAttrs::PCS_config, 2));
  EXPECT_TRUE(testBuildAttr(13, 3, ARMBuildAttrs::PCS_config, 3));
  EXPECT_TRUE(testBuildAttr(13, 4, ARMBuildAttrs::PCS_config, 4));
  EXPECT_TRUE(testBuildAttr(13, 5, ARMBuildAttrs::PCS_config, 5));
  EXPECT_TRUE(testBuildAttr(13, 6, ARMBuildAttrs::PCS_config, 6));
  EXPECT_TRUE(testBuildAttr(13, 7, ARMBuildAttrs::PCS_config, 7));
}

TEST(PCS_R9BuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(14, "Tag_ABI_PCS_R9_use"));
  EXPECT_TRUE(testBuildAttr(14, 0, ARMBuildAttrs::ABI_PCS_R9_use,
                            ARMBuildAttrs::R9IsGPR));
  EXPECT_TRUE(testBuildAttr(14, 1, ARMBuildAttrs::ABI_PCS_R9_use,
                            ARMBuildAttrs::R9IsSB));
  EXPECT_TRUE(testBuildAttr(14, 2, ARMBuildAttrs::ABI_PCS_R9_use,
                            ARMBuildAttrs::R9IsTLSPointer));
  EXPECT_TRUE(testBuildAttr(14, 3, ARMBuildAttrs::ABI_PCS_R9_use,
                            ARMBuildAttrs::R9Reserved));
}

TEST(PCS_RWBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(15, "Tag_ABI_PCS_RW_data"));
  EXPECT_TRUE(testBuildAttr(15, 0, ARMBuildAttrs::ABI_PCS_RW_data,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(15, 1, ARMBuildAttrs::ABI_PCS_RW_data,
                            ARMBuildAttrs::AddressRWPCRel));
  EXPECT_TRUE(testBuildAttr(15, 2, ARMBuildAttrs::ABI_PCS_RW_data,
                            ARMBuildAttrs::AddressRWSBRel));
  EXPECT_TRUE(testBuildAttr(15, 3, ARMBuildAttrs::ABI_PCS_RW_data,
                            ARMBuildAttrs::AddressRWNone));
}

TEST(PCS_ROBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(16, "Tag_ABI_PCS_RO_data"));
  EXPECT_TRUE(testBuildAttr(16, 0, ARMBuildAttrs::ABI_PCS_RO_data,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(16, 1, ARMBuildAttrs::ABI_PCS_RO_data,
                            ARMBuildAttrs::AddressROPCRel));
  EXPECT_TRUE(testBuildAttr(16, 2, ARMBuildAttrs::ABI_PCS_RO_data,
                            ARMBuildAttrs::AddressRONone));
}

TEST(PCS_GOTBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(17, "Tag_ABI_PCS_GOT_use"));
  EXPECT_TRUE(testBuildAttr(17, 0, ARMBuildAttrs::ABI_PCS_GOT_use,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(17, 1, ARMBuildAttrs::ABI_PCS_GOT_use,
                            ARMBuildAttrs::AddressDirect));
  EXPECT_TRUE(testBuildAttr(17, 2, ARMBuildAttrs::ABI_PCS_GOT_use,
                            ARMBuildAttrs::AddressGOT));
}

TEST(PCS_WCharBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(18, "Tag_ABI_PCS_wchar_t"));
  EXPECT_TRUE(testBuildAttr(18, 0, ARMBuildAttrs::ABI_PCS_wchar_t,
                            ARMBuildAttrs::WCharProhibited));
  EXPECT_TRUE(testBuildAttr(18, 2, ARMBuildAttrs::ABI_PCS_wchar_t,
                            ARMBuildAttrs::WCharWidth2Bytes));
  EXPECT_TRUE(testBuildAttr(18, 4, ARMBuildAttrs::ABI_PCS_wchar_t,
                            ARMBuildAttrs::WCharWidth4Bytes));
}

TEST(EnumSizeBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(26, "Tag_ABI_enum_size"));
  EXPECT_TRUE(testBuildAttr(26, 0, ARMBuildAttrs::ABI_enum_size,
                            ARMBuildAttrs::EnumProhibited));
  EXPECT_TRUE(testBuildAttr(26, 1, ARMBuildAttrs::ABI_enum_size,
                            ARMBuildAttrs::EnumSmallest));
  EXPECT_TRUE(testBuildAttr(26, 2, ARMBuildAttrs::ABI_enum_size,
                            ARMBuildAttrs::Enum32Bit));
  EXPECT_TRUE(testBuildAttr(26, 3, ARMBuildAttrs::ABI_enum_size,
                            ARMBuildAttrs::Enum32BitABI));
}

TEST(AlignNeededBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(24, "Tag_ABI_align_needed"));
  EXPECT_TRUE(testBuildAttr(24, 0, ARMBuildAttrs::ABI_align_needed,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(24, 1, ARMBuildAttrs::ABI_align_needed,
                            ARMBuildAttrs::Align8Byte));
  EXPECT_TRUE(testBuildAttr(24, 2, ARMBuildAttrs::ABI_align_needed,
                            ARMBuildAttrs::Align4Byte));
  EXPECT_TRUE(testBuildAttr(24, 3, ARMBuildAttrs::ABI_align_needed,
                            ARMBuildAttrs::AlignReserved));
}

TEST(AlignPreservedBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(25, "Tag_ABI_align_preserved"));
  EXPECT_TRUE(testBuildAttr(25, 0, ARMBuildAttrs::ABI_align_preserved,
                            ARMBuildAttrs::AlignNotPreserved));
  EXPECT_TRUE(testBuildAttr(25, 1, ARMBuildAttrs::ABI_align_preserved,
                            ARMBuildAttrs::AlignPreserve8Byte));
  EXPECT_TRUE(testBuildAttr(25, 2, ARMBuildAttrs::ABI_align_preserved,
                            ARMBuildAttrs::AlignPreserveAll));
  EXPECT_TRUE(testBuildAttr(25, 3, ARMBuildAttrs::ABI_align_preserved,
                            ARMBuildAttrs::AlignReserved));
}

TEST(FPRoundingBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(19, "Tag_ABI_FP_rounding"));
  EXPECT_TRUE(testBuildAttr(19, 0, ARMBuildAttrs::ABI_FP_rounding, 0));
  EXPECT_TRUE(testBuildAttr(19, 1, ARMBuildAttrs::ABI_FP_rounding, 1));
}

TEST(FPDenormalBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(20, "Tag_ABI_FP_denormal"));
  EXPECT_TRUE(testBuildAttr(20, 0, ARMBuildAttrs::ABI_FP_denormal,
                            ARMBuildAttrs::PositiveZero));
  EXPECT_TRUE(testBuildAttr(20, 1, ARMBuildAttrs::ABI_FP_denormal,
                            ARMBuildAttrs::IEEEDenormals));
  EXPECT_TRUE(testBuildAttr(20, 2, ARMBuildAttrs::ABI_FP_denormal,
                            ARMBuildAttrs::PreserveFPSign));
}

TEST(FPExceptionsBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(21, "Tag_ABI_FP_exceptions"));
  EXPECT_TRUE(testBuildAttr(21, 0, ARMBuildAttrs::ABI_FP_exceptions, 0));
  EXPECT_TRUE(testBuildAttr(21, 1, ARMBuildAttrs::ABI_FP_exceptions, 1));
}

TEST(FPUserExceptionsBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(22, "Tag_ABI_FP_user_exceptions"));
  EXPECT_TRUE(testBuildAttr(22, 0, ARMBuildAttrs::ABI_FP_user_exceptions, 0));
  EXPECT_TRUE(testBuildAttr(22, 1, ARMBuildAttrs::ABI_FP_user_exceptions, 1));
}

TEST(FPNumberModelBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(23, "Tag_ABI_FP_number_model"));
  EXPECT_TRUE(testBuildAttr(23, 0, ARMBuildAttrs::ABI_FP_number_model,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(23, 1, ARMBuildAttrs::ABI_FP_number_model,
                            ARMBuildAttrs::AllowIEEENormal));
  EXPECT_TRUE(testBuildAttr(23, 2, ARMBuildAttrs::ABI_FP_number_model,
                            ARMBuildAttrs::AllowRTABI));
  EXPECT_TRUE(testBuildAttr(23, 3, ARMBuildAttrs::ABI_FP_number_model,
                            ARMBuildAttrs::AllowIEEE754));
}

TEST(FP16BuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(38, "Tag_ABI_FP_16bit_format"));
  EXPECT_TRUE(testBuildAttr(38, 0, ARMBuildAttrs::ABI_FP_16bit_format,
                            ARMBuildAttrs::Not_Allowed));
  EXPECT_TRUE(testBuildAttr(38, 1, ARMBuildAttrs::ABI_FP_16bit_format,
                            ARMBuildAttrs::FP16FormatIEEE));
  EXPECT_TRUE(testBuildAttr(38, 2, ARMBuildAttrs::ABI_FP_16bit_format,
                            ARMBuildAttrs::FP16VFP3));
}

TEST(HardFPBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(27, "Tag_ABI_HardFP_use"));
  EXPECT_TRUE(testBuildAttr(27, 0, ARMBuildAttrs::ABI_HardFP_use,
                            ARMBuildAttrs::HardFPImplied));
  EXPECT_TRUE(testBuildAttr(27, 1, ARMBuildAttrs::ABI_HardFP_use,
                            ARMBuildAttrs::HardFPSinglePrecision));
  EXPECT_TRUE(testBuildAttr(27, 2, ARMBuildAttrs::ABI_HardFP_use, 2));
}

TEST(VFPArgsBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(28, "Tag_ABI_VFP_args"));
  EXPECT_TRUE(testBuildAttr(28, 0, ARMBuildAttrs::ABI_VFP_args,
                            ARMBuildAttrs::BaseAAPCS));
  EXPECT_TRUE(testBuildAttr(28, 1, ARMBuildAttrs::ABI_VFP_args,
                            ARMBuildAttrs::HardFPAAPCS));
  EXPECT_TRUE(testBuildAttr(28, 2, ARMBuildAttrs::ABI_VFP_args, 2));
  EXPECT_TRUE(testBuildAttr(28, 3, ARMBuildAttrs::ABI_VFP_args, 3));
}

TEST(WMMXArgsBuildAttr, testBuildAttr) {
  EXPECT_TRUE(testTagString(29, "Tag_ABI_WMMX_args"));
  EXPECT_TRUE(testBuildAttr(29, 0, ARMBuildAttrs::ABI_WMMX_args, 0));
  EXPECT_TRUE(testBuildAttr(29, 1, ARMBuildAttrs::ABI_WMMX_args, 1));
  EXPECT_TRUE(testBuildAttr(29, 2, ARMBuildAttrs::ABI_WMMX_args, 2));
}
