//===-- ArchSpec.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/StringList.h"
#include "lldb/lldb-defines.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/ARMTargetParser.h"

using namespace lldb;
using namespace lldb_private;

static bool cores_match(const ArchSpec::Core core1, const ArchSpec::Core core2,
                        bool try_inverse, bool enforce_exact_match);

namespace lldb_private {

struct CoreDefinition {
  ByteOrder default_byte_order;
  uint32_t addr_byte_size;
  uint32_t min_opcode_byte_size;
  uint32_t max_opcode_byte_size;
  llvm::Triple::ArchType machine;
  ArchSpec::Core core;
  const char *const name;
};

} // namespace lldb_private

#define AMD_GPU_CORE_DEF_R600(sub)                                             \
  {eByteOrderLittle,                                                           \
   4,                                                                          \
   4,                                                                          \
   16,                                                                         \
   llvm::Triple::r600,                                                         \
   ArchSpec::eCore_amd_gpu_r600_##sub,                                         \
   "r600"}
#define AMD_GPU_CORE_DEF_GCN(sub)                                              \
  {eByteOrderLittle,                                                           \
   8,                                                                          \
   4,                                                                          \
   16,                                                                         \
   llvm::Triple::amdgcn,                                                       \
   ArchSpec::eCore_amd_gpu_gcn_##sub,                                          \
   "amdgcn"}
// This core information can be looked using the ArchSpec::Core as the index
static constexpr const CoreDefinition g_core_definitions[] = {
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_generic,
     "arm"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv4,
     "armv4"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv4t,
     "armv4t"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv5,
     "armv5"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv5e,
     "armv5e"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv5t,
     "armv5t"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv6,
     "armv6"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv6m,
     "armv6m"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7,
     "armv7"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7a,
     "armv7a"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7l,
     "armv7l"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7f,
     "armv7f"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7s,
     "armv7s"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7k,
     "armv7k"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7m,
     "armv7m"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv7em,
     "armv7em"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm,
     ArchSpec::eCore_arm_armv8m_base, "armv8m.base"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm,
     ArchSpec::eCore_arm_armv8m_main, "armv8m.main"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm,
     ArchSpec::eCore_arm_armv8_1m_main, "armv8.1m.main"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_xscale,
     "xscale"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumb,
     "thumb"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv4t,
     "thumbv4t"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv5,
     "thumbv5"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv5e,
     "thumbv5e"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv6,
     "thumbv6"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv6m,
     "thumbv6m"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv7,
     "thumbv7"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv7s,
     "thumbv7s"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv7k,
     "thumbv7k"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv7f,
     "thumbv7f"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv7m,
     "thumbv7m"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb, ArchSpec::eCore_thumbv7em,
     "thumbv7em"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64,
     ArchSpec::eCore_arm_arm64, "arm64"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64,
     ArchSpec::eCore_arm_armv8, "armv8"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64,
     ArchSpec::eCore_arm_armv8a, "armv8a"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arm, ArchSpec::eCore_arm_armv8l,
     "armv8l"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64,
     ArchSpec::eCore_arm_arm64e, "arm64e"},
    {eByteOrderLittle, 4, 4, 4, llvm::Triple::aarch64_32,
     ArchSpec::eCore_arm_arm64_32, "arm64_32"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64,
     ArchSpec::eCore_arm_aarch64, "aarch64"},

    // mips32, mips32r2, mips32r3, mips32r5, mips32r6
    {eByteOrderBig, 4, 2, 4, llvm::Triple::mips, ArchSpec::eCore_mips32,
     "mips"},
    {eByteOrderBig, 4, 2, 4, llvm::Triple::mips, ArchSpec::eCore_mips32r2,
     "mipsr2"},
    {eByteOrderBig, 4, 2, 4, llvm::Triple::mips, ArchSpec::eCore_mips32r3,
     "mipsr3"},
    {eByteOrderBig, 4, 2, 4, llvm::Triple::mips, ArchSpec::eCore_mips32r5,
     "mipsr5"},
    {eByteOrderBig, 4, 2, 4, llvm::Triple::mips, ArchSpec::eCore_mips32r6,
     "mipsr6"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel, ArchSpec::eCore_mips32el,
     "mipsel"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel,
     ArchSpec::eCore_mips32r2el, "mipsr2el"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel,
     ArchSpec::eCore_mips32r3el, "mipsr3el"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel,
     ArchSpec::eCore_mips32r5el, "mipsr5el"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel,
     ArchSpec::eCore_mips32r6el, "mipsr6el"},

    // mips64, mips64r2, mips64r3, mips64r5, mips64r6
    {eByteOrderBig, 8, 2, 4, llvm::Triple::mips64, ArchSpec::eCore_mips64,
     "mips64"},
    {eByteOrderBig, 8, 2, 4, llvm::Triple::mips64, ArchSpec::eCore_mips64r2,
     "mips64r2"},
    {eByteOrderBig, 8, 2, 4, llvm::Triple::mips64, ArchSpec::eCore_mips64r3,
     "mips64r3"},
    {eByteOrderBig, 8, 2, 4, llvm::Triple::mips64, ArchSpec::eCore_mips64r5,
     "mips64r5"},
    {eByteOrderBig, 8, 2, 4, llvm::Triple::mips64, ArchSpec::eCore_mips64r6,
     "mips64r6"},
    {eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el,
     ArchSpec::eCore_mips64el, "mips64el"},
    {eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el,
     ArchSpec::eCore_mips64r2el, "mips64r2el"},
    {eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el,
     ArchSpec::eCore_mips64r3el, "mips64r3el"},
    {eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el,
     ArchSpec::eCore_mips64r5el, "mips64r5el"},
    {eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el,
     ArchSpec::eCore_mips64r6el, "mips64r6el"},

    // MSP430
    {eByteOrderLittle, 2, 2, 4, llvm::Triple::msp430, ArchSpec::eCore_msp430,
     "msp430"},

    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_generic,
     "powerpc"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc601,
     "ppc601"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc602,
     "ppc602"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc603,
     "ppc603"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc603e,
     "ppc603e"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc603ev,
     "ppc603ev"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc604,
     "ppc604"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc604e,
     "ppc604e"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc620,
     "ppc620"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc750,
     "ppc750"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc7400,
     "ppc7400"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc7450,
     "ppc7450"},
    {eByteOrderBig, 4, 4, 4, llvm::Triple::ppc, ArchSpec::eCore_ppc_ppc970,
     "ppc970"},

    {eByteOrderLittle, 8, 4, 4, llvm::Triple::ppc64le,
     ArchSpec::eCore_ppc64le_generic, "powerpc64le"},
    {eByteOrderBig, 8, 4, 4, llvm::Triple::ppc64, ArchSpec::eCore_ppc64_generic,
     "powerpc64"},
    {eByteOrderBig, 8, 4, 4, llvm::Triple::ppc64,
     ArchSpec::eCore_ppc64_ppc970_64, "ppc970-64"},

    {eByteOrderBig, 8, 2, 6, llvm::Triple::systemz,
     ArchSpec::eCore_s390x_generic, "s390x"},

    {eByteOrderLittle, 4, 4, 4, llvm::Triple::sparc,
     ArchSpec::eCore_sparc_generic, "sparc"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::sparcv9,
     ArchSpec::eCore_sparc9_generic, "sparcv9"},

    {eByteOrderLittle, 4, 1, 15, llvm::Triple::x86, ArchSpec::eCore_x86_32_i386,
     "i386"},
    {eByteOrderLittle, 4, 1, 15, llvm::Triple::x86, ArchSpec::eCore_x86_32_i486,
     "i486"},
    {eByteOrderLittle, 4, 1, 15, llvm::Triple::x86,
     ArchSpec::eCore_x86_32_i486sx, "i486sx"},
    {eByteOrderLittle, 4, 1, 15, llvm::Triple::x86, ArchSpec::eCore_x86_32_i686,
     "i686"},

    {eByteOrderLittle, 8, 1, 15, llvm::Triple::x86_64,
     ArchSpec::eCore_x86_64_x86_64, "x86_64"},
    {eByteOrderLittle, 8, 1, 15, llvm::Triple::x86_64,
     ArchSpec::eCore_x86_64_x86_64h, "x86_64h"},
    {eByteOrderLittle, 8, 1, 15, llvm::Triple::x86_64,
     ArchSpec::eCore_x86_64_amd64, "amd64"},

    {eByteOrderLittle, 4, 4, 4, llvm::Triple::hexagon,
     ArchSpec::eCore_hexagon_generic, "hexagon"},
    {eByteOrderLittle, 4, 4, 4, llvm::Triple::hexagon,
     ArchSpec::eCore_hexagon_hexagonv4, "hexagonv4"},
    {eByteOrderLittle, 4, 4, 4, llvm::Triple::hexagon,
     ArchSpec::eCore_hexagon_hexagonv5, "hexagonv5"},

    {eByteOrderLittle, 4, 2, 8, llvm::Triple::riscv32, ArchSpec::eCore_riscv32,
     "riscv32"},
    {eByteOrderLittle, 8, 2, 8, llvm::Triple::riscv64, ArchSpec::eCore_riscv64,
     "riscv64"},

    {eByteOrderLittle, 4, 4, 4, llvm::Triple::loongarch32,
     ArchSpec::eCore_loongarch32, "loongarch32"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::loongarch64,
     ArchSpec::eCore_loongarch64, "loongarch64"},

    {eByteOrderLittle, 4, 4, 4, llvm::Triple::UnknownArch,
     ArchSpec::eCore_uknownMach32, "unknown-mach-32"},
    {eByteOrderLittle, 8, 4, 4, llvm::Triple::UnknownArch,
     ArchSpec::eCore_uknownMach64, "unknown-mach-64"},
    {eByteOrderLittle, 4, 2, 4, llvm::Triple::arc, ArchSpec::eCore_arc, "arc"},

    {eByteOrderLittle, 2, 2, 4, llvm::Triple::avr, ArchSpec::eCore_avr, "avr"},

    {eByteOrderLittle, 4, 1, 4, llvm::Triple::wasm32, ArchSpec::eCore_wasm32,
     "wasm32"},
    AMD_GPU_CORE_DEF_R600(R600),
    AMD_GPU_CORE_DEF_R600(R630),
    AMD_GPU_CORE_DEF_R600(RS880),
    AMD_GPU_CORE_DEF_R600(RV670),
    AMD_GPU_CORE_DEF_R600(RV710),
    AMD_GPU_CORE_DEF_R600(RV730),
    AMD_GPU_CORE_DEF_R600(RV770),
    AMD_GPU_CORE_DEF_R600(CEDAR),
    AMD_GPU_CORE_DEF_R600(CYPRESS),
    AMD_GPU_CORE_DEF_R600(JUNIPER),
    AMD_GPU_CORE_DEF_R600(REDWOOD),
    AMD_GPU_CORE_DEF_R600(SUMO),
    AMD_GPU_CORE_DEF_R600(BARTS),
    AMD_GPU_CORE_DEF_R600(CAICOS),
    AMD_GPU_CORE_DEF_R600(CAYMAN),
    AMD_GPU_CORE_DEF_R600(TURKS),
    AMD_GPU_CORE_DEF_GCN(GFX600),
    AMD_GPU_CORE_DEF_GCN(GFX601),
    AMD_GPU_CORE_DEF_GCN(GFX602),
    AMD_GPU_CORE_DEF_GCN(GFX700),
    AMD_GPU_CORE_DEF_GCN(GFX701),
    AMD_GPU_CORE_DEF_GCN(GFX702),
    AMD_GPU_CORE_DEF_GCN(GFX703),
    AMD_GPU_CORE_DEF_GCN(GFX704),
    AMD_GPU_CORE_DEF_GCN(GFX705),
    AMD_GPU_CORE_DEF_GCN(GFX801),
    AMD_GPU_CORE_DEF_GCN(GFX802),
    AMD_GPU_CORE_DEF_GCN(GFX803),
    AMD_GPU_CORE_DEF_GCN(GFX805),
    AMD_GPU_CORE_DEF_GCN(GFX810),
    AMD_GPU_CORE_DEF_GCN(GFX900),
    AMD_GPU_CORE_DEF_GCN(GFX902),
    AMD_GPU_CORE_DEF_GCN(GFX904),
    AMD_GPU_CORE_DEF_GCN(GFX906),
    AMD_GPU_CORE_DEF_GCN(GFX908),
    AMD_GPU_CORE_DEF_GCN(GFX909),
    AMD_GPU_CORE_DEF_GCN(GFX90A),
    AMD_GPU_CORE_DEF_GCN(GFX90C),
    AMD_GPU_CORE_DEF_GCN(GFX942),
    AMD_GPU_CORE_DEF_GCN(GFX950),
    AMD_GPU_CORE_DEF_GCN(GFX1010),
    AMD_GPU_CORE_DEF_GCN(GFX1011),
    AMD_GPU_CORE_DEF_GCN(GFX1012),
    AMD_GPU_CORE_DEF_GCN(GFX1013),
    AMD_GPU_CORE_DEF_GCN(GFX1030),
    AMD_GPU_CORE_DEF_GCN(GFX1031),
    AMD_GPU_CORE_DEF_GCN(GFX1032),
    AMD_GPU_CORE_DEF_GCN(GFX1033),
    AMD_GPU_CORE_DEF_GCN(GFX1034),
    AMD_GPU_CORE_DEF_GCN(GFX1035),
    AMD_GPU_CORE_DEF_GCN(GFX1036),
    AMD_GPU_CORE_DEF_GCN(GFX1100),
    AMD_GPU_CORE_DEF_GCN(GFX1101),
    AMD_GPU_CORE_DEF_GCN(GFX1102),
    AMD_GPU_CORE_DEF_GCN(GFX1103),
    AMD_GPU_CORE_DEF_GCN(GFX1150),
    AMD_GPU_CORE_DEF_GCN(GFX1151),
    AMD_GPU_CORE_DEF_GCN(GFX1152),
    AMD_GPU_CORE_DEF_GCN(GFX1153),
    AMD_GPU_CORE_DEF_GCN(GFX1154),
    AMD_GPU_CORE_DEF_GCN(GFX1170),
    AMD_GPU_CORE_DEF_GCN(GFX1171),
    AMD_GPU_CORE_DEF_GCN(GFX1172),
    AMD_GPU_CORE_DEF_GCN(GFX1200),
    AMD_GPU_CORE_DEF_GCN(GFX1201),
    AMD_GPU_CORE_DEF_GCN(GFX1250),
    AMD_GPU_CORE_DEF_GCN(GFX1251),
    AMD_GPU_CORE_DEF_GCN(GFX1310),
    AMD_GPU_CORE_DEF_GCN(GFX9_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX9_4_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX10_1_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX10_3_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX11_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX12_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX12_5_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX11_7_GENERIC),
    AMD_GPU_CORE_DEF_GCN(GFX13_GENERIC),
    {eByteOrderLittle, 8, 4, 16, llvm::Triple::amdgcn,
     ArchSpec::eCore_amd_gpu_unknown, "amdgcn"},
};

// Ensure that we have an entry in the g_core_definitions for each core. If you
// comment out an entry above, you will need to comment out the corresponding
// ArchSpec::Core enumeration.
static_assert(sizeof(g_core_definitions) / sizeof(CoreDefinition) ==
                  ArchSpec::kNumCores,
              "make sure we have one core definition for each core");

template <int I> struct ArchSpecValidator : ArchSpecValidator<I + 1> {
  static_assert(g_core_definitions[I].core == I,
                "g_core_definitions order doesn't match Core enumeration");
};

template <> struct ArchSpecValidator<ArchSpec::kNumCores> {};

ArchSpecValidator<ArchSpec::eCore_arm_generic> validator;

struct ArchDefinitionEntry {
  ArchSpec::Core core;
  uint32_t cpu;
  uint32_t sub = LLDB_INVALID_CPUTYPE;
  uint32_t cpu_mask = UINT32_MAX;
  uint32_t sub_mask = UINT32_MAX;
};

struct ArchDefinition {
  ArchitectureType type;
  size_t num_entries;
  const ArchDefinitionEntry *entries;
  const char *name;
};

void ArchSpec::ListSupportedArchNames(StringList &list) {
  for (const auto &def : g_core_definitions)
    list.AppendString(def.name);
}

void ArchSpec::AutoComplete(CompletionRequest &request) {
  for (const auto &def : g_core_definitions)
    request.TryCompleteCurrentArg(def.name);
}

#define CPU_ANY (UINT32_MAX)

//===----------------------------------------------------------------------===//
// A table that gets searched linearly for matches. This table is used to
// convert cpu type and subtypes to architecture names, and to convert
// architecture names to cpu types and subtypes. The ordering is important and
// allows the precedence to be set when the table is built.
#define SUBTYPE_MASK 0x00FFFFFFu

// clang-format off
static const ArchDefinitionEntry g_macho_arch_entries[] = {
    {ArchSpec::eCore_arm_generic,     llvm::MachO::CPU_TYPE_ARM,        CPU_ANY,                                UINT32_MAX, UINT32_MAX},
    {ArchSpec::eCore_arm_generic,     llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_ALL,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv4,       llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V4T,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv4t,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V4T,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv6,       llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V6,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv6m,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V6M,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv5,       llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V5TEJ,     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv5e,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V5TEJ,     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv5t,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V5TEJ,     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_xscale,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_XSCALE,    UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv7,       llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv7f,      llvm::MachO::CPU_TYPE_ARM,        10,                                     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv7s,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7S,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv7k,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7K,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv7m,      llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7M,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv7em,     llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7EM,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv8m_base,     llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V8M_BASE,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv8m_main,     llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V8M_MAIN,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_armv8_1m_main,     llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V8_1M_MAIN,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64e,      llvm::MachO::CPU_TYPE_ARM64,      llvm::MachO::CPU_SUBTYPE_ARM64E,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64,       llvm::MachO::CPU_TYPE_ARM64,      llvm::MachO::CPU_SUBTYPE_ARM64_ALL,     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64,       llvm::MachO::CPU_TYPE_ARM64,      llvm::MachO::CPU_SUBTYPE_ARM64_V8,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64,       llvm::MachO::CPU_TYPE_ARM64,      13,                                     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64_32,    llvm::MachO::CPU_TYPE_ARM64_32,   0,                                      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64_32,    llvm::MachO::CPU_TYPE_ARM64_32,   1,                                      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_arm_arm64,       llvm::MachO::CPU_TYPE_ARM64,      CPU_ANY,                                UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumb,           llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_ALL,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv4t,        llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V4T,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv5,         llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V5,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv5e,        llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V5,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv6,         llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V6,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv6m,        llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V6M,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv7,         llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7,        UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv7f,        llvm::MachO::CPU_TYPE_ARM,        10,                                     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv7s,        llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7S,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv7k,        llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7K,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv7m,        llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7M,       UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_thumbv7em,       llvm::MachO::CPU_TYPE_ARM,        llvm::MachO::CPU_SUBTYPE_ARM_V7EM,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_generic,     llvm::MachO::CPU_TYPE_POWERPC,    CPU_ANY,                                UINT32_MAX, UINT32_MAX},
    {ArchSpec::eCore_ppc_generic,     llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_ALL,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc601,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_601,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc602,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_602,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc603,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_603,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc603e,     llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_603e,  UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc603ev,    llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_603ev, UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc604,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_604,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc604e,     llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_604e,  UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc620,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_620,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc750,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_750,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc7400,     llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_7400,  UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc7450,     llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_7450,  UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc_ppc970,      llvm::MachO::CPU_TYPE_POWERPC,    llvm::MachO::CPU_SUBTYPE_POWERPC_970,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc64_generic,   llvm::MachO::CPU_TYPE_POWERPC64,  llvm::MachO::CPU_SUBTYPE_POWERPC_ALL,   UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc64le_generic, llvm::MachO::CPU_TYPE_POWERPC64,  CPU_ANY,                                UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_ppc64_ppc970_64, llvm::MachO::CPU_TYPE_POWERPC64,  100,                                    UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_32_i386,     llvm::MachO::CPU_TYPE_I386,       llvm::MachO::CPU_SUBTYPE_I386_ALL,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_32_i486,     llvm::MachO::CPU_TYPE_I386,       llvm::MachO::CPU_SUBTYPE_486,           UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_32_i486sx,   llvm::MachO::CPU_TYPE_I386,       llvm::MachO::CPU_SUBTYPE_486SX,         UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_32_i386,     llvm::MachO::CPU_TYPE_I386,       CPU_ANY,                                UINT32_MAX, UINT32_MAX},
    {ArchSpec::eCore_x86_64_x86_64,   llvm::MachO::CPU_TYPE_X86_64,     llvm::MachO::CPU_SUBTYPE_X86_64_ALL,    UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_64_x86_64,   llvm::MachO::CPU_TYPE_X86_64,     llvm::MachO::CPU_SUBTYPE_X86_ARCH1,     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_64_x86_64h,  llvm::MachO::CPU_TYPE_X86_64,     llvm::MachO::CPU_SUBTYPE_X86_64_H,      UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_x86_64_x86_64,   llvm::MachO::CPU_TYPE_X86_64,     CPU_ANY, UINT32_MAX, UINT32_MAX},
    {ArchSpec::eCore_riscv32,         llvm::MachO::CPU_TYPE_RISCV,      llvm::MachO::CPU_SUBTYPE_RISCV_ALL,     UINT32_MAX, SUBTYPE_MASK},
    {ArchSpec::eCore_riscv32,         llvm::MachO::CPU_TYPE_RISCV,      CPU_ANY,                                UINT32_MAX, SUBTYPE_MASK},
    // Catch any unknown mach architectures so we can always use the object and symbol mach-o files
    {ArchSpec::eCore_uknownMach32,    0,                                0,                                      0xFF000000u, 0x00000000u},
    {ArchSpec::eCore_uknownMach64,    llvm::MachO::CPU_ARCH_ABI64,      0,                                      0xFF000000u, 0x00000000u}
};
// clang-format on

static const ArchDefinition g_macho_arch_def = {eArchTypeMachO,
                                                std::size(g_macho_arch_entries),
                                                g_macho_arch_entries, "mach-o"};

#define AMD_GPU_ARCH_DEF_R600(sub)                                             \
  {ArchSpec::eCore_amd_gpu_r600_##sub, llvm::ELF::EM_AMDGPU,                   \
   llvm::ELF::EF_AMDGPU_MACH_R600_##sub}
#define AMD_GPU_ARCH_DEF_GCN(sub)                                              \
  {ArchSpec::eCore_amd_gpu_gcn_##sub, llvm::ELF::EM_AMDGPU,                    \
   llvm::ELF::EF_AMDGPU_MACH_AMDGCN_##sub}
//===----------------------------------------------------------------------===//
// A table that gets searched linearly for matches. This table is used to
// convert cpu type and subtypes to architecture names, and to convert
// architecture names to cpu types and subtypes. The ordering is important and
// allows the precedence to be set when the table is built.
// clang-format off
static const ArchDefinitionEntry g_elf_arch_entries[] = {
    {ArchSpec::eCore_sparc_generic,   llvm::ELF::EM_SPARC       }, // Sparc
    {ArchSpec::eCore_x86_32_i386,     llvm::ELF::EM_386         }, // Intel 80386
    {ArchSpec::eCore_x86_32_i486,     llvm::ELF::EM_IAMCU       }, // Intel MCU // FIXME: is this correct?
    {ArchSpec::eCore_ppc_generic,     llvm::ELF::EM_PPC         }, // PowerPC
    {ArchSpec::eCore_ppc64le_generic, llvm::ELF::EM_PPC64,      ArchSpec::eCore_ppc64le_generic},   // PowerPC64le
    {ArchSpec::eCore_ppc64_generic,   llvm::ELF::EM_PPC64,      ArchSpec::eCore_ppc64_generic},     // PowerPC64
    {ArchSpec::eCore_arm_generic,     llvm::ELF::EM_ARM         }, // ARM
    {ArchSpec::eCore_arm_aarch64,     llvm::ELF::EM_AARCH64     }, // ARM64
    {ArchSpec::eCore_s390x_generic,   llvm::ELF::EM_S390        }, // SystemZ
    {ArchSpec::eCore_sparc9_generic,  llvm::ELF::EM_SPARCV9     }, // SPARC V9
    {ArchSpec::eCore_x86_64_x86_64,   llvm::ELF::EM_X86_64      }, // AMD64
    {ArchSpec::eCore_mips32,          llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips32}, // mips32
    {ArchSpec::eCore_mips32r2,        llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips32r2}, // mips32r2
    {ArchSpec::eCore_mips32r6,        llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips32r6}, // mips32r6
    {ArchSpec::eCore_mips32el,        llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips32el}, // mips32el
    {ArchSpec::eCore_mips32r2el,      llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips32r2el}, // mips32r2el
    {ArchSpec::eCore_mips32r6el,      llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips32r6el}, // mips32r6el
    {ArchSpec::eCore_mips64,          llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips64},
    {ArchSpec::eCore_mips64r2,        llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips64r2}, // mips64r2
    {ArchSpec::eCore_mips64r6,        llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips64r6}, // mips64r6
    {ArchSpec::eCore_mips64el,        llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips64el}, // mips64el
    {ArchSpec::eCore_mips64r2el,      llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips64r2el}, // mips64r2el
    {ArchSpec::eCore_mips64r6el,      llvm::ELF::EM_MIPS,       ArchSpec::eMIPSSubType_mips64r6el}, // mips64r6el
    {ArchSpec::eCore_msp430,          llvm::ELF::EM_MSP430      }, // MSP430
    {ArchSpec::eCore_hexagon_generic, llvm::ELF::EM_HEXAGON     }, // HEXAGON
    {ArchSpec::eCore_arc,             llvm::ELF::EM_ARC_COMPACT2}, // ARC
    {ArchSpec::eCore_avr,             llvm::ELF::EM_AVR         }, // AVR
    {ArchSpec::eCore_riscv32,         llvm::ELF::EM_RISCV,      ArchSpec::eRISCVSubType_riscv32}, // riscv32
    {ArchSpec::eCore_riscv64,         llvm::ELF::EM_RISCV,      ArchSpec::eRISCVSubType_riscv64}, // riscv64
    {ArchSpec::eCore_loongarch32,     llvm::ELF::EM_LOONGARCH,  ArchSpec::eLoongArchSubType_loongarch32}, // loongarch32
    {ArchSpec::eCore_loongarch64,     llvm::ELF::EM_LOONGARCH,  ArchSpec::eLoongArchSubType_loongarch64}, // loongarch64
    AMD_GPU_ARCH_DEF_R600(R600),
    AMD_GPU_ARCH_DEF_R600(R630),
    AMD_GPU_ARCH_DEF_R600(RS880),
    AMD_GPU_ARCH_DEF_R600(RV670),
    AMD_GPU_ARCH_DEF_R600(RV710),
    AMD_GPU_ARCH_DEF_R600(RV730),
    AMD_GPU_ARCH_DEF_R600(RV770),
    AMD_GPU_ARCH_DEF_R600(CEDAR),
    AMD_GPU_ARCH_DEF_R600(CYPRESS),
    AMD_GPU_ARCH_DEF_R600(JUNIPER),
    AMD_GPU_ARCH_DEF_R600(REDWOOD),
    AMD_GPU_ARCH_DEF_R600(SUMO),
    AMD_GPU_ARCH_DEF_R600(BARTS),
    AMD_GPU_ARCH_DEF_R600(CAICOS),
    AMD_GPU_ARCH_DEF_R600(CAYMAN),
    AMD_GPU_ARCH_DEF_R600(TURKS),
    AMD_GPU_ARCH_DEF_GCN(GFX600),
    AMD_GPU_ARCH_DEF_GCN(GFX601),
    AMD_GPU_ARCH_DEF_GCN(GFX602),
    AMD_GPU_ARCH_DEF_GCN(GFX700),
    AMD_GPU_ARCH_DEF_GCN(GFX701),
    AMD_GPU_ARCH_DEF_GCN(GFX702),
    AMD_GPU_ARCH_DEF_GCN(GFX703),
    AMD_GPU_ARCH_DEF_GCN(GFX704),
    AMD_GPU_ARCH_DEF_GCN(GFX705),
    AMD_GPU_ARCH_DEF_GCN(GFX801),
    AMD_GPU_ARCH_DEF_GCN(GFX802),
    AMD_GPU_ARCH_DEF_GCN(GFX803),
    AMD_GPU_ARCH_DEF_GCN(GFX805),
    AMD_GPU_ARCH_DEF_GCN(GFX810),
    AMD_GPU_ARCH_DEF_GCN(GFX900),
    AMD_GPU_ARCH_DEF_GCN(GFX902),
    AMD_GPU_ARCH_DEF_GCN(GFX904),
    AMD_GPU_ARCH_DEF_GCN(GFX906),
    AMD_GPU_ARCH_DEF_GCN(GFX908),
    AMD_GPU_ARCH_DEF_GCN(GFX909),
    AMD_GPU_ARCH_DEF_GCN(GFX90A),
    AMD_GPU_ARCH_DEF_GCN(GFX90C),
    AMD_GPU_ARCH_DEF_GCN(GFX942),
    AMD_GPU_ARCH_DEF_GCN(GFX950),
    AMD_GPU_ARCH_DEF_GCN(GFX1010),
    AMD_GPU_ARCH_DEF_GCN(GFX1011),
    AMD_GPU_ARCH_DEF_GCN(GFX1012),
    AMD_GPU_ARCH_DEF_GCN(GFX1013),
    AMD_GPU_ARCH_DEF_GCN(GFX1030),
    AMD_GPU_ARCH_DEF_GCN(GFX1031),
    AMD_GPU_ARCH_DEF_GCN(GFX1032),
    AMD_GPU_ARCH_DEF_GCN(GFX1033),
    AMD_GPU_ARCH_DEF_GCN(GFX1034),
    AMD_GPU_ARCH_DEF_GCN(GFX1035),
    AMD_GPU_ARCH_DEF_GCN(GFX1036),
    AMD_GPU_ARCH_DEF_GCN(GFX1100),
    AMD_GPU_ARCH_DEF_GCN(GFX1101),
    AMD_GPU_ARCH_DEF_GCN(GFX1102),
    AMD_GPU_ARCH_DEF_GCN(GFX1103),
    AMD_GPU_ARCH_DEF_GCN(GFX1150),
    AMD_GPU_ARCH_DEF_GCN(GFX1151),
    AMD_GPU_ARCH_DEF_GCN(GFX1152),
    AMD_GPU_ARCH_DEF_GCN(GFX1153),
    AMD_GPU_ARCH_DEF_GCN(GFX1154),
    AMD_GPU_ARCH_DEF_GCN(GFX1170),
    AMD_GPU_ARCH_DEF_GCN(GFX1171),
    AMD_GPU_ARCH_DEF_GCN(GFX1172),
    AMD_GPU_ARCH_DEF_GCN(GFX1200),
    AMD_GPU_ARCH_DEF_GCN(GFX1201),
    AMD_GPU_ARCH_DEF_GCN(GFX1250),
    AMD_GPU_ARCH_DEF_GCN(GFX1251),
    AMD_GPU_ARCH_DEF_GCN(GFX1310),
    AMD_GPU_ARCH_DEF_GCN(GFX9_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX9_4_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX10_1_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX10_3_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX11_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX12_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX12_5_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX11_7_GENERIC),
    AMD_GPU_ARCH_DEF_GCN(GFX13_GENERIC),
    // Any AMDGPU object with no recognized model resolves here.
    {ArchSpec::eCore_amd_gpu_unknown, llvm::ELF::EM_AMDGPU},
};
// clang-format on

static const ArchDefinition g_elf_arch_def = {
    eArchTypeELF,
    std::size(g_elf_arch_entries),
    g_elf_arch_entries,
    "elf",
};
// clang-format off
static const ArchDefinitionEntry g_coff_arch_entries[] = {
    {ArchSpec::eCore_x86_32_i386,   llvm::COFF::IMAGE_FILE_MACHINE_I386}, // Intel 80x86
    {ArchSpec::eCore_ppc_generic,   llvm::COFF::IMAGE_FILE_MACHINE_POWERPC}, // PowerPC
    {ArchSpec::eCore_ppc_generic,   llvm::COFF::IMAGE_FILE_MACHINE_POWERPCFP}, // PowerPC (with FPU)
    {ArchSpec::eCore_arm_generic,   llvm::COFF::IMAGE_FILE_MACHINE_ARM}, // ARM
    {ArchSpec::eCore_arm_armv7,     llvm::COFF::IMAGE_FILE_MACHINE_ARMNT}, // ARMv7
    {ArchSpec::eCore_thumb,         llvm::COFF::IMAGE_FILE_MACHINE_THUMB}, // ARMv7
    {ArchSpec::eCore_x86_64_x86_64, llvm::COFF::IMAGE_FILE_MACHINE_AMD64}, // AMD64
    {ArchSpec::eCore_arm_arm64,     llvm::COFF::IMAGE_FILE_MACHINE_ARM64} // ARM64
};
// clang-format on

static const ArchDefinition g_coff_arch_def = {
    eArchTypeCOFF,
    std::size(g_coff_arch_entries),
    g_coff_arch_entries,
    "pe-coff",
};

// clang-format off
static const ArchDefinitionEntry g_xcoff_arch_entries[] = {
    {ArchSpec::eCore_ppc_generic,   llvm::XCOFF::TCPU_PPC},
    {ArchSpec::eCore_ppc64_generic, llvm::XCOFF::TCPU_PPC64}
};
// clang-format on

static const ArchDefinition g_xcoff_arch_def = {
    eArchTypeXCOFF,
    std::size(g_xcoff_arch_entries),
    g_xcoff_arch_entries,
    "xcoff",
};

//===----------------------------------------------------------------------===//
// Table of all ArchDefinitions
static const ArchDefinition *g_arch_definitions[] = {
    &g_macho_arch_def, &g_elf_arch_def, &g_coff_arch_def, &g_xcoff_arch_def};

//===----------------------------------------------------------------------===//
// Static helper functions.

// Get the architecture definition for a given object type.
static const ArchDefinition *FindArchDefinition(ArchitectureType arch_type) {
  for (const ArchDefinition *def : g_arch_definitions) {
    if (def->type == arch_type)
      return def;
  }
  return nullptr;
}

// Get an architecture definition by name.
static const CoreDefinition *FindCoreDefinition(llvm::StringRef name) {
  for (const auto &def : g_core_definitions) {
    if (name.equals_insensitive(def.name))
      return &def;
  }
  return nullptr;
}

static inline const CoreDefinition *FindCoreDefinition(ArchSpec::Core core) {
  if (core < std::size(g_core_definitions))
    return &g_core_definitions[core];
  return nullptr;
}

// Get a definition entry by cpu type and subtype.
static const ArchDefinitionEntry *
FindArchDefinitionEntry(const ArchDefinition *def, uint32_t cpu, uint32_t sub) {
  if (def == nullptr)
    return nullptr;

  const ArchDefinitionEntry *entries = def->entries;
  for (size_t i = 0; i < def->num_entries; ++i) {
    if (entries[i].cpu == (cpu & entries[i].cpu_mask))
      if (entries[i].sub == (sub & entries[i].sub_mask))
        return &entries[i];
  }
  return nullptr;
}

static const ArchDefinitionEntry *
FindArchDefinitionEntry(const ArchDefinition *def, ArchSpec::Core core) {
  if (def == nullptr)
    return nullptr;

  const ArchDefinitionEntry *entries = def->entries;
  for (size_t i = 0; i < def->num_entries; ++i) {
    if (entries[i].core == core)
      return &entries[i];
  }
  return nullptr;
}

static llvm::StringRef GetAMDGPUVariantName(uint32_t sub);

//===----------------------------------------------------------------------===//
// Constructors and destructors.

ArchSpec::ArchSpec() = default;

ArchSpec::ArchSpec(const char *triple_cstr) {
  if (triple_cstr)
    SetTriple(triple_cstr);
}

ArchSpec::ArchSpec(llvm::StringRef triple_str) { SetTriple(triple_str); }

ArchSpec::ArchSpec(const llvm::Triple &triple) { SetTriple(triple); }

ArchSpec::ArchSpec(ArchitectureType arch_type, uint32_t cpu, uint32_t subtype) {
  SetArchitecture(arch_type, cpu, subtype);
}

ArchSpec::~ArchSpec() = default;

void ArchSpec::Clear() {
  m_triple = llvm::Triple();
  m_core = kCore_invalid;
  m_byte_order = eByteOrderInvalid;
  m_flags = 0;
}

//===----------------------------------------------------------------------===//
// Predicates.

const char *ArchSpec::GetArchitectureName() const {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def)
    return core_def->name;
  return "unknown";
}

bool ArchSpec::IsMIPS() const { return GetTriple().isMIPS(); }

bool ArchSpec::IsNVPTX() const { return GetTriple().isNVPTX(); }

std::string ArchSpec::GetTargetABI() const {

  std::string abi;

  if (IsMIPS()) {
    switch (GetFlags() & ArchSpec::eMIPSABI_mask) {
    case ArchSpec::eMIPSABI_N64:
      abi = "n64";
      return abi;
    case ArchSpec::eMIPSABI_N32:
      abi = "n32";
      return abi;
    case ArchSpec::eMIPSABI_O32:
      abi = "o32";
      return abi;
    default:
      return abi;
    }
  }
  return abi;
}

void ArchSpec::SetFlags(const std::string &elf_abi) {

  uint32_t flag = GetFlags();
  if (IsMIPS()) {
    if (elf_abi == "n64")
      flag |= ArchSpec::eMIPSABI_N64;
    else if (elf_abi == "n32")
      flag |= ArchSpec::eMIPSABI_N32;
    else if (elf_abi == "o32")
      flag |= ArchSpec::eMIPSABI_O32;
  }
  SetFlags(flag);
}

std::string ArchSpec::GetClangTargetCPU() const {
  std::string cpu;
  if (IsMIPS()) {
    switch (m_core) {
    case ArchSpec::eCore_mips32:
    case ArchSpec::eCore_mips32el:
      cpu = "mips32";
      break;
    case ArchSpec::eCore_mips32r2:
    case ArchSpec::eCore_mips32r2el:
      cpu = "mips32r2";
      break;
    case ArchSpec::eCore_mips32r3:
    case ArchSpec::eCore_mips32r3el:
      cpu = "mips32r3";
      break;
    case ArchSpec::eCore_mips32r5:
    case ArchSpec::eCore_mips32r5el:
      cpu = "mips32r5";
      break;
    case ArchSpec::eCore_mips32r6:
    case ArchSpec::eCore_mips32r6el:
      cpu = "mips32r6";
      break;
    case ArchSpec::eCore_mips64:
    case ArchSpec::eCore_mips64el:
      cpu = "mips64";
      break;
    case ArchSpec::eCore_mips64r2:
    case ArchSpec::eCore_mips64r2el:
      cpu = "mips64r2";
      break;
    case ArchSpec::eCore_mips64r3:
    case ArchSpec::eCore_mips64r3el:
      cpu = "mips64r3";
      break;
    case ArchSpec::eCore_mips64r5:
    case ArchSpec::eCore_mips64r5el:
      cpu = "mips64r5";
      break;
    case ArchSpec::eCore_mips64r6:
    case ArchSpec::eCore_mips64r6el:
      cpu = "mips64r6";
      break;
    default:
      break;
    }
  }

  if (GetTriple().isARM())
    cpu = llvm::ARM::getARMCPUForArch(GetTriple(), "").str();

  if (GetTriple().isAMDGPU()) {
    uint32_t sub = GetElfCPUSubType();
    if (sub != LLDB_INVALID_CPUTYPE)
      cpu = GetAMDGPUVariantName(sub);
  }
  return cpu;
}

static const ArchDefinitionEntry *
FindArchDefEntryIfCoreIsValid(const ArchDefinition *def, ArchSpec::Core core) {
  if (const CoreDefinition *core_def = FindCoreDefinition(core))
    return FindArchDefinitionEntry(def, core_def->core);

  return nullptr;
}

static uint32_t GetCPUType(const ArchDefinition *def, ArchSpec::Core core) {
  if (const ArchDefinitionEntry *arch_def =
          FindArchDefEntryIfCoreIsValid(def, core))
    return arch_def->cpu;
  return LLDB_INVALID_CPUTYPE;
}

static uint32_t GetCPUSubType(const ArchDefinition *def, ArchSpec::Core core) {
  if (const ArchDefinitionEntry *arch_def =
          FindArchDefEntryIfCoreIsValid(def, core))
    return arch_def->sub;
  return LLDB_INVALID_CPUTYPE;
}

uint32_t ArchSpec::GetMachOCPUType() const {
  return GetCPUType(&g_macho_arch_def, m_core);
}

uint32_t ArchSpec::GetMachOCPUSubType() const {
  return GetCPUSubType(&g_macho_arch_def, m_core);
}

uint32_t ArchSpec::GetElfCPUSubType() const {
  return GetCPUSubType(&g_elf_arch_def, m_core);
}

llvm::Triple::ArchType ArchSpec::GetMachine() const {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def)
    return core_def->machine;

  return llvm::Triple::UnknownArch;
}

uint32_t ArchSpec::GetAddressByteSize() const {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def) {
    if (core_def->machine == llvm::Triple::mips64 ||
        core_def->machine == llvm::Triple::mips64el) {
      // For N32/O32 applications Address size is 4 bytes.
      if (m_flags & (eMIPSABI_N32 | eMIPSABI_O32))
        return 4;
    }
    return core_def->addr_byte_size;
  }
  return 0;
}

ByteOrder ArchSpec::GetDefaultEndian() const {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def)
    return core_def->default_byte_order;
  return eByteOrderInvalid;
}

bool ArchSpec::CharIsSignedByDefault() const {
  switch (m_triple.getArch()) {
  default:
    return true;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    return m_triple.isOSDarwin() || m_triple.isOSWindows();

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    return m_triple.isOSDarwin();

  case llvm::Triple::riscv64:
  case llvm::Triple::riscv32:
  case llvm::Triple::ppc64le:
  case llvm::Triple::systemz:
  case llvm::Triple::xcore:
  case llvm::Triple::arc:
    return false;
  }
}

lldb::ByteOrder ArchSpec::GetByteOrder() const {
  if (m_byte_order == eByteOrderInvalid)
    return GetDefaultEndian();
  return m_byte_order;
}

//===----------------------------------------------------------------------===//
// Mutators.

bool ArchSpec::SetTriple(const llvm::Triple &triple) {
  m_triple = triple;
  UpdateCore();
  return IsValid();
}

bool lldb_private::ParseMachCPUDashSubtypeTriple(llvm::StringRef triple_str,
                                                 ArchSpec &arch) {
  // Accept "12-10" or "12.10" as cpu type/subtype
  if (triple_str.empty())
    return false;

  size_t pos = triple_str.find_first_of("-.");
  if (pos == llvm::StringRef::npos)
    return false;

  llvm::StringRef cpu_str = triple_str.substr(0, pos);
  llvm::StringRef remainder = triple_str.substr(pos + 1);
  if (cpu_str.empty() || remainder.empty())
    return false;

  llvm::StringRef sub_str;
  llvm::StringRef vendor;
  llvm::StringRef os;
  std::tie(sub_str, remainder) = remainder.split('-');
  std::tie(vendor, os) = remainder.split('-');

  uint32_t cpu = 0;
  uint32_t sub = 0;
  if (cpu_str.getAsInteger(10, cpu) || sub_str.getAsInteger(10, sub))
    return false;

  if (!arch.SetArchitecture(eArchTypeMachO, cpu, sub))
    return false;
  if (!vendor.empty() && !os.empty()) {
    arch.GetTriple().setVendorName(vendor);
    arch.GetTriple().setOSName(os);
  }

  return true;
}

bool ArchSpec::SetTriple(llvm::StringRef triple) {
  if (triple.empty()) {
    Clear();
    return false;
  }

  if (ParseMachCPUDashSubtypeTriple(triple, *this))
    return true;

  SetTriple(llvm::Triple(llvm::Triple::normalize(triple)));
  return IsValid();
}

bool ArchSpec::ContainsOnlyArch(const llvm::Triple &normalized_triple) {
  return !normalized_triple.getArchName().empty() &&
         normalized_triple.getOSName().empty() &&
         normalized_triple.getVendorName().empty() &&
         normalized_triple.getEnvironmentName().empty();
}

void ArchSpec::MergeFrom(const ArchSpec &other) {
  // ios-macabi always wins over macosx.
  if ((GetTriple().getOS() == llvm::Triple::MacOSX ||
       GetTriple().getOS() == llvm::Triple::UnknownOS) &&
      other.GetTriple().getOS() == llvm::Triple::IOS &&
      other.GetTriple().getEnvironment() == llvm::Triple::MacABI) {
    (*this) = other;
    return;
  }

  if (!TripleVendorWasSpecified() && other.TripleVendorWasSpecified())
    GetTriple().setVendor(other.GetTriple().getVendor());
  if (!TripleOSWasSpecified() && other.TripleOSWasSpecified())
    GetTriple().setOS(other.GetTriple().getOS());
  if (GetTriple().getArch() == llvm::Triple::UnknownArch) {
    GetTriple().setArch(other.GetTriple().getArch());

    // MachO unknown64 isn't really invalid as the debugger can still obtain
    // information from the binary, e.g. line tables. As such, we don't update
    // the core here.
    if (other.GetCore() != eCore_uknownMach64)
      UpdateCore();
  }
  if (!TripleEnvironmentWasSpecified() &&
      other.TripleEnvironmentWasSpecified()) {
    GetTriple().setEnvironment(other.GetTriple().getEnvironment());
  }
  // If this and other are both arm ArchSpecs and this ArchSpec is a generic
  // "some kind of arm" spec but the other ArchSpec is a specific arm core,
  // adopt the specific arm core.
  if (GetTriple().getArch() == llvm::Triple::arm &&
      other.GetTriple().getArch() == llvm::Triple::arm &&
      IsCompatibleMatch(other) && GetCore() == ArchSpec::eCore_arm_generic &&
      other.GetCore() != ArchSpec::eCore_arm_generic) {
    m_core = other.GetCore();
    CoreUpdated(false);
  }
  if (GetFlags() == 0) {
    SetFlags(other.GetFlags());
  }
}

static llvm::StringRef GetAMDGPUVariantName(uint32_t sub) {
  switch (sub) {
  case llvm::ELF::EF_AMDGPU_MACH_R600_R600:
    return "r600";
  case llvm::ELF::EF_AMDGPU_MACH_R600_R630:
    return "r630";
  case llvm::ELF::EF_AMDGPU_MACH_R600_RS880:
    return "rs880";
  case llvm::ELF::EF_AMDGPU_MACH_R600_RV670:
    return "rv670";
  case llvm::ELF::EF_AMDGPU_MACH_R600_RV710:
    return "rv710";
  case llvm::ELF::EF_AMDGPU_MACH_R600_RV730:
    return "rv730";
  case llvm::ELF::EF_AMDGPU_MACH_R600_RV770:
    return "rv770";
  case llvm::ELF::EF_AMDGPU_MACH_R600_CEDAR:
    return "cedar";
  case llvm::ELF::EF_AMDGPU_MACH_R600_CYPRESS:
    return "cypress";
  case llvm::ELF::EF_AMDGPU_MACH_R600_JUNIPER:
    return "juniper";
  case llvm::ELF::EF_AMDGPU_MACH_R600_REDWOOD:
    return "redwood";
  case llvm::ELF::EF_AMDGPU_MACH_R600_SUMO:
    return "sumo";
  case llvm::ELF::EF_AMDGPU_MACH_R600_BARTS:
    return "barts";
  case llvm::ELF::EF_AMDGPU_MACH_R600_CAICOS:
    return "caicos";
  case llvm::ELF::EF_AMDGPU_MACH_R600_CAYMAN:
    return "cayman";
  case llvm::ELF::EF_AMDGPU_MACH_R600_TURKS:
    return "turks";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX600:
    return "gfx600";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX601:
    return "gfx601";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX602:
    return "gfx602";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX700:
    return "gfx700";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX701:
    return "gfx701";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX702:
    return "gfx702";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX703:
    return "gfx703";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX704:
    return "gfx704";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX705:
    return "gfx705";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX801:
    return "gfx801";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX802:
    return "gfx802";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX803:
    return "gfx803";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX805:
    return "gfx805";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX810:
    return "gfx810";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX900:
    return "gfx900";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX902:
    return "gfx902";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX904:
    return "gfx904";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX906:
    return "gfx906";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX908:
    return "gfx908";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX909:
    return "gfx909";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A:
    return "gfx90a";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C:
    return "gfx90c";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX942:
    return "gfx942";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX950:
    return "gfx950";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1010:
    return "gfx1010";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1011:
    return "gfx1011";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1012:
    return "gfx1012";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1013:
    return "gfx1013";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030:
    return "gfx1030";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1031:
    return "gfx1031";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1032:
    return "gfx1032";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1033:
    return "gfx1033";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1034:
    return "gfx1034";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1035:
    return "gfx1035";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1036:
    return "gfx1036";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100:
    return "gfx1100";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1101:
    return "gfx1101";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1102:
    return "gfx1102";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1103:
    return "gfx1103";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1150:
    return "gfx1150";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1151:
    return "gfx1151";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1152:
    return "gfx1152";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1153:
    return "gfx1153";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1154:
    return "gfx1154";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1170:
    return "gfx1170";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1171:
    return "gfx1171";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1172:
    return "gfx1172";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1200:
    return "gfx1200";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1201:
    return "gfx1201";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250:
    return "gfx1250";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1251:
    return "gfx1251";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1310:
    return "gfx1310";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC:
    return "gfx9-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC:
    return "gfx9-4-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC:
    return "gfx10-1-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC:
    return "gfx10-3-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC:
    return "gfx11-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC:
    return "gfx12-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC:
    return "gfx12-5-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX11_7_GENERIC:
    return "gfx11-7-generic";
  case llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX13_GENERIC:
    return "gfx13-generic";
  default:
    break;
  }
  return llvm::StringRef("unknown");
}

static ArchSpec::Core GetAMDGPUVariantToCoreR600(llvm::StringRef core_name) {
  return llvm::StringSwitch<ArchSpec::Core>(core_name)
      .Case("r600", ArchSpec::eCore_amd_gpu_r600_R600)
      .Case("r630", ArchSpec::eCore_amd_gpu_r600_R630)
      .Case("rs880", ArchSpec::eCore_amd_gpu_r600_RS880)
      .Case("rv670", ArchSpec::eCore_amd_gpu_r600_RV670)
      .Case("rv710", ArchSpec::eCore_amd_gpu_r600_RV710)
      .Case("rv730", ArchSpec::eCore_amd_gpu_r600_RV730)
      .Case("rv770", ArchSpec::eCore_amd_gpu_r600_RV770)
      .Case("cedar", ArchSpec::eCore_amd_gpu_r600_CEDAR)
      .Case("cypress", ArchSpec::eCore_amd_gpu_r600_CYPRESS)
      .Case("juniper", ArchSpec::eCore_amd_gpu_r600_JUNIPER)
      .Case("redwood", ArchSpec::eCore_amd_gpu_r600_REDWOOD)
      .Case("sumo", ArchSpec::eCore_amd_gpu_r600_SUMO)
      .Case("barts", ArchSpec::eCore_amd_gpu_r600_BARTS)
      .Case("caicos", ArchSpec::eCore_amd_gpu_r600_CAICOS)
      .Case("cayman", ArchSpec::eCore_amd_gpu_r600_CAYMAN)
      .Case("turks", ArchSpec::eCore_amd_gpu_r600_TURKS)
      .Default(ArchSpec::eCore_amd_gpu_unknown);
}

static ArchSpec::Core GetAMDGPUVariantToCoreGCN(llvm::StringRef core_name) {
  return llvm::StringSwitch<ArchSpec::Core>(core_name)
      .Case("gfx600", ArchSpec::eCore_amd_gpu_gcn_GFX600)
      .Case("gfx601", ArchSpec::eCore_amd_gpu_gcn_GFX601)
      .Case("gfx602", ArchSpec::eCore_amd_gpu_gcn_GFX602)
      .Case("gfx700", ArchSpec::eCore_amd_gpu_gcn_GFX700)
      .Case("gfx701", ArchSpec::eCore_amd_gpu_gcn_GFX701)
      .Case("gfx702", ArchSpec::eCore_amd_gpu_gcn_GFX702)
      .Case("gfx703", ArchSpec::eCore_amd_gpu_gcn_GFX703)
      .Case("gfx704", ArchSpec::eCore_amd_gpu_gcn_GFX704)
      .Case("gfx705", ArchSpec::eCore_amd_gpu_gcn_GFX705)
      .Case("gfx801", ArchSpec::eCore_amd_gpu_gcn_GFX801)
      .Case("gfx802", ArchSpec::eCore_amd_gpu_gcn_GFX802)
      .Case("gfx803", ArchSpec::eCore_amd_gpu_gcn_GFX803)
      .Case("gfx805", ArchSpec::eCore_amd_gpu_gcn_GFX805)
      .Case("gfx810", ArchSpec::eCore_amd_gpu_gcn_GFX810)
      .Case("gfx900", ArchSpec::eCore_amd_gpu_gcn_GFX900)
      .Case("gfx902", ArchSpec::eCore_amd_gpu_gcn_GFX902)
      .Case("gfx904", ArchSpec::eCore_amd_gpu_gcn_GFX904)
      .Case("gfx906", ArchSpec::eCore_amd_gpu_gcn_GFX906)
      .Case("gfx908", ArchSpec::eCore_amd_gpu_gcn_GFX908)
      .Case("gfx909", ArchSpec::eCore_amd_gpu_gcn_GFX909)
      .Case("gfx90a", ArchSpec::eCore_amd_gpu_gcn_GFX90A)
      .Case("gfx90c", ArchSpec::eCore_amd_gpu_gcn_GFX90C)
      .Case("gfx942", ArchSpec::eCore_amd_gpu_gcn_GFX942)
      .Case("gfx950", ArchSpec::eCore_amd_gpu_gcn_GFX950)
      .Case("gfx1010", ArchSpec::eCore_amd_gpu_gcn_GFX1010)
      .Case("gfx1011", ArchSpec::eCore_amd_gpu_gcn_GFX1011)
      .Case("gfx1012", ArchSpec::eCore_amd_gpu_gcn_GFX1012)
      .Case("gfx1013", ArchSpec::eCore_amd_gpu_gcn_GFX1013)
      .Case("gfx1030", ArchSpec::eCore_amd_gpu_gcn_GFX1030)
      .Case("gfx1031", ArchSpec::eCore_amd_gpu_gcn_GFX1031)
      .Case("gfx1032", ArchSpec::eCore_amd_gpu_gcn_GFX1032)
      .Case("gfx1033", ArchSpec::eCore_amd_gpu_gcn_GFX1033)
      .Case("gfx1034", ArchSpec::eCore_amd_gpu_gcn_GFX1034)
      .Case("gfx1035", ArchSpec::eCore_amd_gpu_gcn_GFX1035)
      .Case("gfx1036", ArchSpec::eCore_amd_gpu_gcn_GFX1036)
      .Case("gfx1100", ArchSpec::eCore_amd_gpu_gcn_GFX1100)
      .Case("gfx1101", ArchSpec::eCore_amd_gpu_gcn_GFX1101)
      .Case("gfx1102", ArchSpec::eCore_amd_gpu_gcn_GFX1102)
      .Case("gfx1103", ArchSpec::eCore_amd_gpu_gcn_GFX1103)
      .Case("gfx1150", ArchSpec::eCore_amd_gpu_gcn_GFX1150)
      .Case("gfx1151", ArchSpec::eCore_amd_gpu_gcn_GFX1151)
      .Case("gfx1152", ArchSpec::eCore_amd_gpu_gcn_GFX1152)
      .Case("gfx1153", ArchSpec::eCore_amd_gpu_gcn_GFX1153)
      .Case("gfx1154", ArchSpec::eCore_amd_gpu_gcn_GFX1154)
      .Case("gfx1170", ArchSpec::eCore_amd_gpu_gcn_GFX1170)
      .Case("gfx1171", ArchSpec::eCore_amd_gpu_gcn_GFX1171)
      .Case("gfx1172", ArchSpec::eCore_amd_gpu_gcn_GFX1172)
      .Case("gfx1200", ArchSpec::eCore_amd_gpu_gcn_GFX1200)
      .Case("gfx1201", ArchSpec::eCore_amd_gpu_gcn_GFX1201)
      .Case("gfx1250", ArchSpec::eCore_amd_gpu_gcn_GFX1250)
      .Case("gfx1251", ArchSpec::eCore_amd_gpu_gcn_GFX1251)
      .Case("gfx1310", ArchSpec::eCore_amd_gpu_gcn_GFX1310)
      .Case("gfx9-generic", ArchSpec::eCore_amd_gpu_gcn_GFX9_GENERIC)
      .Case("gfx9-4-generic", ArchSpec::eCore_amd_gpu_gcn_GFX9_4_GENERIC)
      .Case("gfx10-1-generic", ArchSpec::eCore_amd_gpu_gcn_GFX10_1_GENERIC)
      .Case("gfx10-3-generic", ArchSpec::eCore_amd_gpu_gcn_GFX10_3_GENERIC)
      .Case("gfx11-generic", ArchSpec::eCore_amd_gpu_gcn_GFX11_GENERIC)
      .Case("gfx12-generic", ArchSpec::eCore_amd_gpu_gcn_GFX12_GENERIC)
      .Case("gfx12-5-generic", ArchSpec::eCore_amd_gpu_gcn_GFX12_5_GENERIC)
      .Case("gfx11-7-generic", ArchSpec::eCore_amd_gpu_gcn_GFX11_7_GENERIC)
      .Case("gfx13-generic", ArchSpec::eCore_amd_gpu_gcn_GFX13_GENERIC)
      .Default(ArchSpec::eCore_amd_gpu_unknown);
}

bool ArchSpec::SetArchitecture(ArchitectureType arch_type, uint32_t cpu,
                               uint32_t sub, uint32_t os) {
  m_core = kCore_invalid;
  bool update_triple = true;
  const ArchDefinition *arch_def = FindArchDefinition(arch_type);
  if (arch_def) {
    const ArchDefinitionEntry *arch_def_entry =
        FindArchDefinitionEntry(arch_def, cpu, sub);
    if (arch_def_entry) {
      const CoreDefinition *core_def = FindCoreDefinition(arch_def_entry->core);
      if (core_def) {
        m_core = core_def->core;
        update_triple = false;
        // Always use the architecture name because it might be more
        // descriptive than the architecture enum ("armv7" ->
        // llvm::Triple::arm).
        m_triple.setArchName(llvm::StringRef(core_def->name));
        if (arch_type == eArchTypeMachO) {
          m_triple.setVendor(llvm::Triple::Apple);

          // Don't set the OS.  It could be simulator, macosx, ios, watchos,
          // tvos, bridgeos.  We could get close with the cpu type - but we
          // can't get it right all of the time.  Better to leave this unset
          // so other sections of code will set it when they have more
          // information. NB: don't call m_triple.setOS
          // (llvm::Triple::UnknownOS). That sets the OSName to "unknown" and
          // the ArchSpec::TripleVendorWasSpecified() method says that any
          // OSName setting means it was specified.
        } else if (arch_type == eArchTypeELF) {
          switch (os) {
          case llvm::ELF::ELFOSABI_AIX:
            m_triple.setOS(llvm::Triple::OSType::AIX);
            break;
          case llvm::ELF::ELFOSABI_FREEBSD:
            m_triple.setOS(llvm::Triple::OSType::FreeBSD);
            break;
          case llvm::ELF::ELFOSABI_GNU:
            m_triple.setOS(llvm::Triple::OSType::Linux);
            break;
          case llvm::ELF::ELFOSABI_NETBSD:
            m_triple.setOS(llvm::Triple::OSType::NetBSD);
            break;
          case llvm::ELF::ELFOSABI_OPENBSD:
            m_triple.setOS(llvm::Triple::OSType::OpenBSD);
            break;
          case llvm::ELF::ELFOSABI_SOLARIS:
            m_triple.setOS(llvm::Triple::OSType::Solaris);
            break;
          case llvm::ELF::ELFOSABI_STANDALONE:
            m_triple.setOS(llvm::Triple::OSType::UnknownOS);
            break;
          case llvm::ELF::ELFOSABI_AMDGPU_HSA:
            m_triple.setVendor(llvm::Triple::VendorType::AMD);
            m_triple.setOS(llvm::Triple::OSType::AMDHSA);
            break;
          }
        } else if (arch_type == eArchTypeCOFF && os == llvm::Triple::Win32) {
          m_triple.setVendor(llvm::Triple::PC);
          m_triple.setOS(llvm::Triple::Win32);
        } else if (arch_type == eArchTypeXCOFF && os == llvm::Triple::AIX) {
          m_triple.setVendor(llvm::Triple::IBM);
          m_triple.setOS(llvm::Triple::AIX);
        } else {
          m_triple.setVendor(llvm::Triple::UnknownVendor);
          m_triple.setOS(llvm::Triple::UnknownOS);
        }
        switch (m_triple.getArch()) {
        case llvm::Triple::UnknownArch:
          // Fall back onto setting the machine type if the arch by name
          // failed...
          m_triple.setArch(core_def->machine);
          break;
        case llvm::Triple::r600:
        case llvm::Triple::amdgcn: {
          // AMDGPU arches are special: they append a 5th element to the triple
          // that comes after the environment and contains the sub type name.
          std::string environment("-");
          environment += GetAMDGPUVariantName(arch_def_entry->sub);
          m_triple.setEnvironmentName(environment);
          break;
        }
        default:
          break;
        }
      }
    } else {
      Log *log(GetLog(LLDBLog::Target | LLDBLog::Process | LLDBLog::Platform));
      LLDB_LOGF(log,
                "Unable to find a core definition for cpu 0x%" PRIx32
                " sub %" PRId32,
                cpu, sub);
    }
  }
  CoreUpdated(update_triple);
  return IsValid();
}

uint32_t ArchSpec::GetMinimumOpcodeByteSize() const {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def)
    return core_def->min_opcode_byte_size;
  return 0;
}

uint32_t ArchSpec::GetMaximumOpcodeByteSize() const {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def)
    return core_def->max_opcode_byte_size;
  return 0;
}

static bool IsCompatibleEnvironment(llvm::Triple::EnvironmentType lhs,
                                    llvm::Triple::EnvironmentType rhs) {
  if (lhs == rhs)
    return true;

  // Apple simulators are a different platform than what they simulate.
  // As the environments are different at this point, if one of them is a
  // simulator, then they are different.
  if (lhs == llvm::Triple::Simulator || rhs == llvm::Triple::Simulator)
    return false;

  // If any of the environment is unknown then they are compatible
  if (lhs == llvm::Triple::UnknownEnvironment ||
      rhs == llvm::Triple::UnknownEnvironment)
    return true;

  // If one of the environment is Android and the other one is EABI then they
  // are considered to be compatible. This is required as a workaround for
  // shared libraries compiled for Android without the NOTE section indicating
  // that they are using the Android ABI.
  if ((lhs == llvm::Triple::Android && rhs == llvm::Triple::EABI) ||
      (rhs == llvm::Triple::Android && lhs == llvm::Triple::EABI) ||
      (lhs == llvm::Triple::GNUEABI && rhs == llvm::Triple::EABI) ||
      (rhs == llvm::Triple::GNUEABI && lhs == llvm::Triple::EABI) ||
      (lhs == llvm::Triple::GNUEABIHF && rhs == llvm::Triple::EABIHF) ||
      (rhs == llvm::Triple::GNUEABIHF && lhs == llvm::Triple::EABIHF))
    return true;

  return false;
}

bool ArchSpec::IsMatch(const ArchSpec &rhs, MatchType match) const {
  if (GetByteOrder() != rhs.GetByteOrder() ||
      !cores_match(GetCore(), rhs.GetCore(), true, match == ExactMatch))
    return false;

  const llvm::Triple &lhs_triple = GetTriple();
  const llvm::Triple &rhs_triple = rhs.GetTriple();

  const llvm::Triple::VendorType lhs_triple_vendor = lhs_triple.getVendor();
  const llvm::Triple::VendorType rhs_triple_vendor = rhs_triple.getVendor();

  const llvm::Triple::OSType lhs_triple_os = lhs_triple.getOS();
  const llvm::Triple::OSType rhs_triple_os = rhs_triple.getOS();

  bool both_windows = lhs_triple.isOSWindows() && rhs_triple.isOSWindows();

  // On Windows, the vendor field doesn't have any practical effect, but
  // it is often set to either "pc" or "w64".
  if ((lhs_triple_vendor != rhs_triple_vendor) &&
      (match == ExactMatch || !both_windows)) {
    const bool rhs_vendor_specified = rhs.TripleVendorWasSpecified();
    const bool lhs_vendor_specified = TripleVendorWasSpecified();
    // Both architectures had the vendor specified, so if they aren't equal
    // then we return false
    if (rhs_vendor_specified && lhs_vendor_specified)
      return false;

    // Only fail if both vendor types are not unknown
    if (lhs_triple_vendor != llvm::Triple::UnknownVendor &&
        rhs_triple_vendor != llvm::Triple::UnknownVendor)
      return false;
  }

  const llvm::Triple::EnvironmentType lhs_triple_env =
      lhs_triple.getEnvironment();
  const llvm::Triple::EnvironmentType rhs_triple_env =
      rhs_triple.getEnvironment();

  if (match == CompatibleMatch) {
    // x86_64-apple-ios-macabi, x86_64-apple-macosx are compatible, no match.
    if ((lhs_triple_os == llvm::Triple::IOS &&
         lhs_triple_env == llvm::Triple::MacABI &&
         rhs_triple_os == llvm::Triple::MacOSX) ||
        (lhs_triple_os == llvm::Triple::MacOSX &&
         rhs_triple_os == llvm::Triple::IOS &&
         rhs_triple_env == llvm::Triple::MacABI))
      return true;
    // x86_64-apple-driverkit, x86_64-apple-macosx are compatible, no match.
    if ((lhs_triple_os == llvm::Triple::DriverKit &&
         rhs_triple_os == llvm::Triple::MacOSX) ||
        (lhs_triple_os == llvm::Triple::MacOSX &&
         rhs_triple_os == llvm::Triple::DriverKit))
      return true;
  }

  // x86_64-apple-ios-macabi and x86_64-apple-ios are not compatible.
  if (lhs_triple_os == llvm::Triple::IOS &&
      rhs_triple_os == llvm::Triple::IOS &&
      (lhs_triple_env == llvm::Triple::MacABI ||
       rhs_triple_env == llvm::Triple::MacABI) &&
      lhs_triple_env != rhs_triple_env)
    return false;

  if (lhs_triple_os != rhs_triple_os) {
    const bool lhs_os_specified = TripleOSWasSpecified();
    const bool rhs_os_specified = rhs.TripleOSWasSpecified();
    // If both OS types are specified and different, fail.
    if (lhs_os_specified && rhs_os_specified)
      return false;

    // If the pair of os+env is both unspecified, match any other os+env combo.
    if (match == CompatibleMatch &&
        ((!lhs_os_specified && !lhs_triple.hasEnvironment()) ||
         (!rhs_os_specified && !rhs_triple.hasEnvironment())))
      return true;
  }

  if (match == CompatibleMatch && both_windows)
    return true; // The Windows environments (MSVC vs GNU) are compatible

  return IsCompatibleEnvironment(lhs_triple_env, rhs_triple_env);
}

void ArchSpec::UpdateCore() {
  llvm::StringRef arch_name(m_triple.getArchName());
  const CoreDefinition *core_def = FindCoreDefinition(arch_name);
  if (core_def) {
    m_core = core_def->core;
    // Set the byte order to the default byte order for an architecture. This
    // can be modified if needed for cases when cores handle both big and
    // little endian
    m_byte_order = core_def->default_byte_order;

    // amdgcn/r600 match their first table entry (GFX600/R600), so refine the
    // core from the model in the triple environment.
    if (m_core == eCore_amd_gpu_gcn_GFX600) {
      m_core = GetAMDGPUVariantToCoreGCN(
          m_triple.getEnvironmentName().split('-').second);
    } else if (m_core == eCore_amd_gpu_r600_R600) {
      m_core = GetAMDGPUVariantToCoreR600(
          m_triple.getEnvironmentName().split('-').second);
    }
  } else {
    Clear();
  }
}

//===----------------------------------------------------------------------===//
// Helper methods.

void ArchSpec::CoreUpdated(bool update_triple) {
  const CoreDefinition *core_def = FindCoreDefinition(m_core);
  if (core_def) {
    if (update_triple)
      m_triple = llvm::Triple(core_def->name, "unknown", "unknown");
    m_byte_order = core_def->default_byte_order;
  } else {
    if (update_triple)
      m_triple = llvm::Triple();
    m_byte_order = eByteOrderInvalid;
  }
}

//===----------------------------------------------------------------------===//
// Operators.

static bool cores_match(const ArchSpec::Core core1, const ArchSpec::Core core2,
                        bool try_inverse, bool enforce_exact_match) {
  if (core1 == core2)
    return true;

  switch (core1) {
  case ArchSpec::kCore_any:
    return true;

  case ArchSpec::eCore_arm_generic:
    if (enforce_exact_match)
      break;
    [[fallthrough]];
  case ArchSpec::kCore_arm_any:
    if (core2 >= ArchSpec::kCore_arm_first && core2 <= ArchSpec::kCore_arm_last)
      return true;
    if (core2 >= ArchSpec::kCore_thumb_first &&
        core2 <= ArchSpec::kCore_thumb_last)
      return true;
    if (core2 == ArchSpec::kCore_arm_any)
      return true;
    break;

  case ArchSpec::kCore_x86_32_any:
    if ((core2 >= ArchSpec::kCore_x86_32_first &&
         core2 <= ArchSpec::kCore_x86_32_last) ||
        (core2 == ArchSpec::kCore_x86_32_any))
      return true;
    break;

  case ArchSpec::kCore_x86_64_any:
    if ((core2 >= ArchSpec::kCore_x86_64_first &&
         core2 <= ArchSpec::kCore_x86_64_last) ||
        (core2 == ArchSpec::kCore_x86_64_any))
      return true;
    break;

  case ArchSpec::kCore_ppc_any:
    if ((core2 >= ArchSpec::kCore_ppc_first &&
         core2 <= ArchSpec::kCore_ppc_last) ||
        (core2 == ArchSpec::kCore_ppc_any))
      return true;
    break;

  case ArchSpec::kCore_ppc64_any:
    if ((core2 >= ArchSpec::kCore_ppc64_first &&
         core2 <= ArchSpec::kCore_ppc64_last) ||
        (core2 == ArchSpec::kCore_ppc64_any))
      return true;
    break;

  case ArchSpec::kCore_hexagon_any:
    if ((core2 >= ArchSpec::kCore_hexagon_first &&
         core2 <= ArchSpec::kCore_hexagon_last) ||
        (core2 == ArchSpec::kCore_hexagon_any))
      return true;
    break;

  // v. https://en.wikipedia.org/wiki/ARM_Cortex-M#Silicon_customization
  // Cortex-M0 - ARMv6-M - armv6m
  // Cortex-M3 - ARMv7-M - armv7m
  // Cortex-M4 - ARMv7E-M - armv7em
  case ArchSpec::eCore_arm_armv7em:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_generic)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7m)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv6m)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7)
        return true;
      try_inverse = true;
    }
    break;

  // v. https://en.wikipedia.org/wiki/ARM_Cortex-M#Silicon_customization
  // Cortex-M0 - ARMv6-M - armv6m
  // Cortex-M3 - ARMv7-M - armv7m
  // Cortex-M4 - ARMv7E-M - armv7em
  case ArchSpec::eCore_arm_armv7m:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_generic)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv6m)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7em)
        return true;
      try_inverse = true;
    }
    break;

  // v. https://en.wikipedia.org/wiki/ARM_Cortex-M#Silicon_customization
  // Cortex-M0 - ARMv6-M - armv6m
  // Cortex-M3 - ARMv7-M - armv7m
  // Cortex-M4 - ARMv7E-M - armv7em
  case ArchSpec::eCore_arm_armv6m:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_generic)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7em)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv6m)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_arm_armv7f:
  case ArchSpec::eCore_arm_armv7k:
  case ArchSpec::eCore_arm_armv7s:
  case ArchSpec::eCore_arm_armv7l:
  case ArchSpec::eCore_arm_armv8l:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_generic)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv7)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_x86_64_x86_64h:
  case ArchSpec::eCore_x86_64_amd64:
    if (!enforce_exact_match) {
      try_inverse = false;
      if (core2 == ArchSpec::eCore_x86_64_x86_64)
        return true;
    }
    break;

  case ArchSpec::eCore_arm_armv8:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_arm64)
        return true;
      if (core2 == ArchSpec::eCore_arm_aarch64)
        return true;
      if (core2 == ArchSpec::eCore_arm_arm64e)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_arm_arm64e:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_arm64)
        return true;
      if (core2 == ArchSpec::eCore_arm_aarch64)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv8)
        return true;
      try_inverse = false;
    }
    break;
  case ArchSpec::eCore_arm_aarch64:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_arm64)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv8)
        return true;
      if (core2 == ArchSpec::eCore_arm_arm64e)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_arm_arm64:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_aarch64)
        return true;
      if (core2 == ArchSpec::eCore_arm_armv8)
        return true;
      if (core2 == ArchSpec::eCore_arm_arm64e)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_arm_arm64_32:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_arm_generic)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_mips32:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32_first &&
          core2 <= ArchSpec::kCore_mips32_last)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_mips32el:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32el_first &&
          core2 <= ArchSpec::kCore_mips32el_last)
        return true;
      try_inverse = true;
    }
    break;

  case ArchSpec::eCore_mips64:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32_first &&
          core2 <= ArchSpec::kCore_mips32_last)
        return true;
      if (core2 >= ArchSpec::kCore_mips64_first &&
          core2 <= ArchSpec::kCore_mips64_last)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_mips64el:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32el_first &&
          core2 <= ArchSpec::kCore_mips32el_last)
        return true;
      if (core2 >= ArchSpec::kCore_mips64el_first &&
          core2 <= ArchSpec::kCore_mips64el_last)
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_mips64r2:
  case ArchSpec::eCore_mips64r3:
  case ArchSpec::eCore_mips64r5:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32_first && core2 <= (core1 - 10))
        return true;
      if (core2 >= ArchSpec::kCore_mips64_first && core2 <= (core1 - 1))
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_mips64r2el:
  case ArchSpec::eCore_mips64r3el:
  case ArchSpec::eCore_mips64r5el:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32el_first && core2 <= (core1 - 10))
        return true;
      if (core2 >= ArchSpec::kCore_mips64el_first && core2 <= (core1 - 1))
        return true;
      try_inverse = false;
    }
    break;

  case ArchSpec::eCore_mips32r2:
  case ArchSpec::eCore_mips32r3:
  case ArchSpec::eCore_mips32r5:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32_first && core2 <= core1)
        return true;
    }
    break;

  case ArchSpec::eCore_mips32r2el:
  case ArchSpec::eCore_mips32r3el:
  case ArchSpec::eCore_mips32r5el:
    if (!enforce_exact_match) {
      if (core2 >= ArchSpec::kCore_mips32el_first && core2 <= core1)
        return true;
    }
    break;

  case ArchSpec::eCore_mips32r6:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_mips32 || core2 == ArchSpec::eCore_mips32r6)
        return true;
    }
    break;

  case ArchSpec::eCore_mips32r6el:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_mips32el ||
          core2 == ArchSpec::eCore_mips32r6el)
        return true;
    }
    break;

  case ArchSpec::eCore_mips64r6:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_mips32 || core2 == ArchSpec::eCore_mips32r6)
        return true;
      if (core2 == ArchSpec::eCore_mips64 || core2 == ArchSpec::eCore_mips64r6)
        return true;
    }
    break;

  case ArchSpec::eCore_mips64r6el:
    if (!enforce_exact_match) {
      if (core2 == ArchSpec::eCore_mips32el ||
          core2 == ArchSpec::eCore_mips32r6el)
        return true;
      if (core2 == ArchSpec::eCore_mips64el ||
          core2 == ArchSpec::eCore_mips64r6el)
        return true;
    }
    break;

  default:
    break;
  }
  if (try_inverse)
    return cores_match(core2, core1, false, enforce_exact_match);
  return false;
}

bool lldb_private::operator<(const ArchSpec &lhs, const ArchSpec &rhs) {
  const ArchSpec::Core lhs_core = lhs.GetCore();
  const ArchSpec::Core rhs_core = rhs.GetCore();
  return lhs_core < rhs_core;
}

bool lldb_private::operator==(const ArchSpec &lhs, const ArchSpec &rhs) {
  return lhs.GetCore() == rhs.GetCore();
}

bool lldb_private::operator!=(const ArchSpec &lhs, const ArchSpec &rhs) {
  return !(lhs == rhs);
}

bool ArchSpec::IsFullySpecifiedTriple() const {
  if (!TripleOSWasSpecified())
    return false;

  if (!TripleVendorWasSpecified())
    return false;

  const unsigned unspecified = 0;
  const llvm::Triple &triple = GetTriple();
  if (triple.isOSDarwin() && triple.getOSMajorVersion() == unspecified)
    return false;

  return true;
}

bool ArchSpec::IsAlwaysThumbInstructions() const {
  if (GetTriple().getArch() == llvm::Triple::arm ||
      GetTriple().getArch() == llvm::Triple::thumb) {
    // v. https://en.wikipedia.org/wiki/ARM_Cortex-M
    //
    // Cortex-M0 through Cortex-M7 are ARM processor cores which can only
    // execute thumb instructions.  We map the cores to arch names like this:
    //
    // Cortex-M0, Cortex-M0+, Cortex-M1:  armv6m Cortex-M3: armv7m Cortex-M4,
    // Cortex-M7: armv7em

    if (GetCore() == ArchSpec::Core::eCore_arm_armv7m ||
        GetCore() == ArchSpec::Core::eCore_arm_armv7em ||
        GetCore() == ArchSpec::Core::eCore_arm_armv6m ||
        GetCore() == ArchSpec::Core::eCore_thumbv7m ||
        GetCore() == ArchSpec::Core::eCore_thumbv7em ||
        GetCore() == ArchSpec::Core::eCore_thumbv6m) {
      return true;
    }
    // Windows on ARM is always thumb.
    if (GetTriple().isOSWindows())
      return true;
  }
  return false;
}

void ArchSpec::DumpTriple(llvm::raw_ostream &s) const {
  const llvm::Triple &triple = GetTriple();
  llvm::StringRef arch_str = triple.getArchName();
  llvm::StringRef vendor_str = triple.getVendorName();
  llvm::StringRef os_str = triple.getOSName();
  llvm::StringRef environ_str = triple.getEnvironmentName();

  s << llvm::formatv("{0}-{1}-{2}", arch_str.empty() ? "*" : arch_str,
                     vendor_str.empty() ? "*" : vendor_str,
                     os_str.empty() ? "*" : os_str);

  if (!environ_str.empty())
    s << "-" << environ_str;
}
