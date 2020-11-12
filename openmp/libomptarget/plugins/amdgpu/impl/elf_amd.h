// AMDGPU specific e_flags.
enum : unsigned {
  // Processor selection mask for EF_AMDGPU_MACH_* values.
  EF_AMDGPU_MACH = 0x0ff,

  // Not specified processor.
  EF_AMDGPU_MACH_NONE = 0x000,

  // AMDGCN GFX8.
  EF_AMDGPU_MACH_AMDGCN_GFX801 = 0x028,
  EF_AMDGPU_MACH_AMDGCN_GFX802 = 0x029,
  EF_AMDGPU_MACH_AMDGCN_GFX803 = 0x02a,
  EF_AMDGPU_MACH_AMDGCN_GFX805 = 0x03c,
  EF_AMDGPU_MACH_AMDGCN_GFX810 = 0x02b,
  // AMDGCN GFX9.
  EF_AMDGPU_MACH_AMDGCN_GFX900 = 0x02c,
  EF_AMDGPU_MACH_AMDGCN_GFX902 = 0x02d,
  EF_AMDGPU_MACH_AMDGCN_GFX904 = 0x02e,
  EF_AMDGPU_MACH_AMDGCN_GFX906 = 0x02f,
  EF_AMDGPU_MACH_AMDGCN_GFX908 = 0x030,
  EF_AMDGPU_MACH_AMDGCN_GFX909 = 0x031,
  EF_AMDGPU_MACH_AMDGCN_GFX90C = 0x032,
  // AMDGCN GFX10.
  EF_AMDGPU_MACH_AMDGCN_GFX1010 = 0x033,
  EF_AMDGPU_MACH_AMDGCN_GFX1011 = 0x034,
  EF_AMDGPU_MACH_AMDGCN_GFX1012 = 0x035,
  EF_AMDGPU_MACH_AMDGCN_GFX1030 = 0x036,
  EF_AMDGPU_MACH_AMDGCN_GFX1031 = 0x037,
  EF_AMDGPU_MACH_AMDGCN_GFX1032 = 0x038,
  EF_AMDGPU_MACH_AMDGCN_GFX1033 = 0x039,

  // Reserved for AMDGCN-based processors.
  EF_AMDGPU_MACH_AMDGCN_RESERVED_LAST = 0x0ff,

  // First/last AMDGCN-based processors.
  EF_AMDGPU_MACH_AMDGCN_FIRST = EF_AMDGPU_MACH_AMDGCN_GFX801,
  EF_AMDGPU_MACH_AMDGCN_LAST = EF_AMDGPU_MACH_AMDGCN_RESERVED_LAST,

  // Indicates if the "xnack" target feature is enabled for all code contained
  // in the object.
  EF_AMDGPU_XNACK = 0x100,
  // Indicates if the "sram-ecc" target feature is enabled for all code
  // contained in the object.
  EF_AMDGPU_SRAM_ECC = 0x200,
};

static bool get_elf_mach_sram_ecc(__tgt_device_image *image) {
  uint32_t EFlags = elf_flags(image);
  return (EF_AMDGPU_SRAM_ECC & EFlags) != 0;
}


static bool get_elf_mach_xnack(__tgt_device_image *image) {
  uint32_t EFlags = elf_flags(image);
  return (EF_AMDGPU_XNACK & EFlags) != 0;
}

static int get_elf_mach_gfx(__tgt_device_image *image) {
  uint32_t EFlags = elf_flags(image);
  uint32_t Gfx = (EFlags & EF_AMDGPU_MACH);
  return Gfx;
}

static const char* get_elf_mach_gfx_name(__tgt_device_image *image) {
  uint32_t Gfx =get_elf_mach_gfx(image);
  switch  (Gfx) {
  case EF_AMDGPU_MACH_AMDGCN_GFX801 :  return "gfx801" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX802 :  return "gfx802" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX803 :  return "gfx803" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX805 :  return "gfx805" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX810 :  return "gfx810" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX900 :  return "gfx900" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX902 :  return "gfx902" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX904 :  return "gfx904" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX906 :  return "gfx906" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX908 :  return "gfx908" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX909 :  return "gfx909" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX90C :  return "gfx90c" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1010 :  return "gfx1010" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1011 :  return "gfx1011" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1012 :  return "gfx1012" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1030 :  return "gfx1030" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1031 :  return "gfx1031" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1032 :  return "gfx1032" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1033 :  return "gfx1033" ;
  default: return "--unknown gfx";
  }
}

static bool elf_machine_id_is_amdgcn(__tgt_device_image *image) {
  const uint16_t amdgcnMachineID = 224;
  int32_t r = elf_check_machine(image, amdgcnMachineID);
  if (!r) {
    DP("Supported machine ID not found\n");
  }
  return r;
}

