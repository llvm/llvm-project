//===------------------------- ManglingUtils.cpp -------------------------===//
//
//                              SPIR Tools
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
/*
 * Contributed by: Intel Corporation.
 */

#include "ManglingUtils.h"

namespace SPIR {

// String represenration for the primitive types.
static const char *PrimitiveNames[PRIMITIVE_NUM] = {
    "bool",
    "uchar",
    "char",
    "ushort",
    "short",
    "uint",
    "int",
    "ulong",
    "long",
    "half",
    "float",
    "double",
    "void",
    "...",
    "image1d_ro_t",
    "image1d_array_ro_t",
    "image1d_buffer_ro_t",
    "image2d_ro_t",
    "image2d_array_ro_t",
    "image2d_depth_ro_t",
    "image2d_array_depth_ro_t",
    "image2d_msaa_ro_t",
    "image2d_array_msaa_ro_t",
    "image2d_msaa_depth_ro_t",
    "image2d_array_msaa_depth_ro_t",
    "image3d_ro_t",
    "image1d_wo_t",
    "image1d_array_wo_t",
    "image1d_buffer_wo_t",
    "image2d_wo_t",
    "image2d_array_wo_t",
    "image2d_depth_wo_t",
    "image2d_array_depth_wo_t",
    "image2d_msaa_wo_t",
    "image2d_array_msaa_wo_t",
    "image2d_msaa_depth_wo_t",
    "image2d_array_msaa_depth_wo_t",
    "image3d_wo_t",
    "image1d_rw_t",
    "image1d_array_rw_t",
    "image1d_buffer_rw_t",
    "image2d_rw_t",
    "image2d_array_rw_t",
    "image2d_depth_rw_t",
    "image2d_array_depth_rw_t",
    "image2d_msaa_rw_t",
    "image2d_array_msaa_rw_t",
    "image2d_msaa_depth_rw_t",
    "image2d_array_msaa_depth_rw_t",
    "image3d_rw_t",
    "event_t",
    "pipe_ro_t",
    "pipe_wo_t",
    "reserve_id_t",
    "queue_t",
    "ndrange_t",
    "clk_event_t",
    "sampler_t",
    "kernel_enqueue_flags_t",
    "clk_profiling_info",
    "memory_order",
    "memory_scope"};

const char *MangledTypes[PRIMITIVE_NUM] = {
    "b",                                 // BOOL
    "h",                                 // UCHAR
    "c",                                 // CHAR
    "t",                                 // USHORT
    "s",                                 // SHORT
    "j",                                 // UINT
    "i",                                 // INT
    "m",                                 // ULONG
    "l",                                 // LONG
    "Dh",                                // HALF
    "f",                                 // FLOAT
    "d",                                 // DOUBLE
    "v",                                 // VOID
    "z",                                 // VarArg
    "14ocl_image1d_ro",                  // PRIMITIVE_IMAGE1D_RO_T
    "20ocl_image1d_array_ro",            // PRIMITIVE_IMAGE1D_ARRAY_RO_T
    "21ocl_image1d_buffer_ro",           // PRIMITIVE_IMAGE1D_BUFFER_RO_T
    "14ocl_image2d_ro",                  // PRIMITIVE_IMAGE2D_RO_T
    "20ocl_image2d_array_ro",            // PRIMITIVE_IMAGE2D_ARRAY_RO_T
    "20ocl_image2d_depth_ro",            // PRIMITIVE_IMAGE2D_DEPTH_RO_T
    "26ocl_image2d_array_depth_ro",      // PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RO_T
    "19ocl_image2d_msaa_ro",             // PRIMITIVE_IMAGE2D_MSAA_RO_T
    "25ocl_image2d_array_msaa_ro",       // PRIMITIVE_IMAGE2D_ARRAY_MSAA_RO_T
    "25ocl_image2d_msaa_depth_ro",       // PRIMITIVE_IMAGE2D_MSAA_DEPTH_RO_T
    "31ocl_image2d_array_msaa_depth_ro", // PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RO_T
    "14ocl_image3d_ro",                  // PRIMITIVE_IMAGE3D_RO_T
    "14ocl_image1d_wo",                  // PRIMITIVE_IMAGE1D_WO_T
    "20ocl_image1d_array_wo",            // PRIMITIVE_IMAGE1D_ARRAY_WO_T
    "21ocl_image1d_buffer_wo",           // PRIMITIVE_IMAGE1D_BUFFER_WO_T
    "14ocl_image2d_wo",                  // PRIMITIVE_IMAGE2D_WO_T
    "20ocl_image2d_array_wo",            // PRIMITIVE_IMAGE2D_ARRAY_WO_T
    "20ocl_image2d_depth_wo",            // PRIMITIVE_IMAGE2D_DEPTH_WO_T
    "26ocl_image2d_array_depth_wo",      // PRIMITIVE_IMAGE2D_ARRAY_DEPTH_WO_T
    "19ocl_image2d_msaa_wo",             // PRIMITIVE_IMAGE2D_MSAA_WO_T
    "25ocl_image2d_array_msaa_wo",       // PRIMITIVE_IMAGE2D_ARRAY_MSAA_WO_T
    "25ocl_image2d_msaa_depth_wo",       // PRIMITIVE_IMAGE2D_MSAA_DEPTH_WO_T
    "31ocl_image2d_array_msaa_depth_wo", // PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_WO_T
    "14ocl_image3d_wo",                  // PRIMITIVE_IMAGE3D_WO_T
    "14ocl_image1d_rw",                  // PRIMITIVE_IMAGE1D_RW_T
    "20ocl_image1d_array_rw",            // PRIMITIVE_IMAGE1D_ARRAY_RW_T
    "21ocl_image1d_buffer_rw",           // PRIMITIVE_IMAGE1D_BUFFER_RW_T
    "14ocl_image2d_rw",                  // PRIMITIVE_IMAGE2D_RW_T
    "20ocl_image2d_array_rw",            // PRIMITIVE_IMAGE2D_ARRAY_RW_T
    "20ocl_image2d_depth_rw",            // PRIMITIVE_IMAGE2D_DEPTH_RW_T
    "26ocl_image2d_array_depth_rw",      // PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RW_T
    "19ocl_image2d_msaa_rw",             // PRIMITIVE_IMAGE2D_MSAA_RW_T
    "25ocl_image2d_array_msaa_rw",       // PRIMITIVE_IMAGE2D_ARRAY_MSAA_RW_T
    "25ocl_image2d_msaa_depth_rw",       // PRIMITIVE_IMAGE2D_MSAA_DEPTH_RW_T
    "31ocl_image2d_array_msaa_depth_rw", // PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RW_T
    "14ocl_image3d_rw",                  // PRIMITIVE_IMAGE3D_RW_T
    "9ocl_event",                        // PRIMITIVE_EVENT_T
    "11ocl_pipe_ro",                     // PRIMITIVE_PIPE_RO_T
    "11ocl_pipe_wo",                     // PRIMITIVE_PIPE_WO_T
    "13ocl_reserveid",                   // PRIMITIVE_RESERVE_ID_T
    "9ocl_queue",                        // PRIMITIVE_QUEUE_T
    "9ndrange_t",                        // PRIMITIVE_NDRANGE_T
    "12ocl_clkevent",                    // PRIMITIVE_CLK_EVENT_T
    "11ocl_sampler",                     // PRIMITIVE_SAMPLER_T
    "i",                                 // PRIMITIVE_KERNEL_ENQUEUE_FLAGS_T
    "i",                                 // PRIMITIVE_CLK_PROFILING_INFO
#if defined(SPIRV_SPIR20_MANGLING_REQUIREMENTS)
    "i", // PRIMITIVE_MEMORY_ORDER
    "i", // PRIMITIVE_MEMORY_SCOPE
#else
    "12memory_order", // PRIMITIVE_MEMORY_ORDER
    "12memory_scope"  // PRIMITIVE_MEMORY_SCOPE
#endif
};

const char *ReadableAttribute[ATTR_NUM] = {
    "restrict", "volatile",   "const",   "__private",
    "__global", "__constant", "__local", "__generic",
};

const char *MangledAttribute[ATTR_NUM] = {
    "r", "V", "K", "", "U3AS1", "U3AS2", "U3AS3", "U3AS4",
};

// SPIR supported version - stated version is oldest supported version.
static const SPIRversion PrimitiveSupportedVersions[PRIMITIVE_NUM] = {
    SPIR12, // BOOL
    SPIR12, // UCHAR
    SPIR12, // CHAR
    SPIR12, // USHORT
    SPIR12, // SHORT
    SPIR12, // UINT
    SPIR12, // INT
    SPIR12, // ULONG
    SPIR12, // LONG
    SPIR12, // HALF
    SPIR12, // FLOAT
    SPIR12, // DOUBLE
    SPIR12, // VOID
    SPIR12, // VarArg
    SPIR12, // PRIMITIVE_IMAGE1D_RO_T
    SPIR12, // PRIMITIVE_IMAGE1D_ARRAY_RO_T
    SPIR12, // PRIMITIVE_IMAGE1D_BUFFER_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_DEPTH_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_MSAA_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_MSAA_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_MSAA_DEPTH_RO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RO_T
    SPIR12, // PRIMITIVE_IMAGE3D_RO_T
    SPIR12, // PRIMITIVE_IMAGE1D_WO_T
    SPIR12, // PRIMITIVE_IMAGE1D_ARRAY_WO_T
    SPIR12, // PRIMITIVE_IMAGE1D_BUFFER_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_DEPTH_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_DEPTH_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_MSAA_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_MSAA_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_MSAA_DEPTH_WO_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_WO_T
    SPIR12, // PRIMITIVE_IMAGE3D_WO_T
    SPIR12, // PRIMITIVE_IMAGE1D_RW_T
    SPIR12, // PRIMITIVE_IMAGE1D_ARRAY_RW_T
    SPIR12, // PRIMITIVE_IMAGE1D_BUFFER_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_DEPTH_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_MSAA_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_MSAA_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_MSAA_DEPTH_RW_T
    SPIR12, // PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RW_T
    SPIR12, // PRIMITIVE_IMAGE3D_RW_T
    SPIR12, // PRIMITIVE_EVENT_T
    SPIR20, // PRIMITIVE_PIPE_RO_T
    SPIR20, // PRIMITIVE_PIPE_WO_T
    SPIR20, // PRIMITIVE_RESERVE_ID_T
    SPIR20, // PRIMITIVE_QUEUE_T
    SPIR20, // PRIMITIVE_NDRANGE_T
    SPIR20, // PRIMITIVE_CLK_EVENT_T
    SPIR12  // PRIMITIVE_SAMPLER_T
};

const char *mangledPrimitiveString(TypePrimitiveEnum T) {
  return MangledTypes[T];
}

const char *readablePrimitiveString(TypePrimitiveEnum T) {
  return PrimitiveNames[T];
}

const char *getMangledAttribute(TypeAttributeEnum Attribute) {
  return MangledAttribute[Attribute];
}

const char *getReadableAttribute(TypeAttributeEnum Attribute) {
  return ReadableAttribute[Attribute];
}

SPIRversion getSupportedVersion(TypePrimitiveEnum T) {
  return PrimitiveSupportedVersions[T];
}

const char *mangledPrimitiveStringfromName(std::string Type) {
  for (size_t I = 0; I < (sizeof(PrimitiveNames) / sizeof(PrimitiveNames[0]));
       I++)
    if (Type == PrimitiveNames[I])
      return MangledTypes[I];
  return NULL;
}

const char *getSPIRVersionAsString(SPIRversion Version) {
  switch (Version) {
  case SPIR12:
    return "SPIR 1.2";
  case SPIR20:
    return "SPIR 2.0";
  default:
    assert(false && "Unknown SPIR Version");
    return "Unknown SPIR Version";
  }
}

} // namespace SPIR
