//===-- NVPTXBaseInfo.h - Top-level definitions for NVPTX -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the NVPTX target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXBASEINFO_H
#define NVPTXBASEINFO_H

namespace llvm {

enum AddressSpace {
  ADDRESS_SPACE_GENERIC = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_CONST_NOT_GEN = 2, // Not part of generic space
  ADDRESS_SPACE_SHARED = 3,
  ADDRESS_SPACE_CONST = 4,
  ADDRESS_SPACE_LOCAL = 5,

  // NVVM Internal
  ADDRESS_SPACE_PARAM = 101
};

enum PropertyAnnotation {
  PROPERTY_MAXNTID_X = 0,
  PROPERTY_MAXNTID_Y,
  PROPERTY_MAXNTID_Z,
  PROPERTY_REQNTID_X,
  PROPERTY_REQNTID_Y,
  PROPERTY_REQNTID_Z,
  PROPERTY_MINNCTAPERSM,
  PROPERTY_ISTEXTURE,
  PROPERTY_ISSURFACE,
  PROPERTY_ISSAMPLER,
  PROPERTY_ISREADONLY_IMAGE_PARAM,
  PROPERTY_ISWRITEONLY_IMAGE_PARAM,
  PROPERTY_ISKERNEL_FUNCTION,
  PROPERTY_ALIGN,

  // last property
  PROPERTY_LAST
};

const unsigned AnnotationNameLen = 8; // length of each annotation name
const char
PropertyAnnotationNames[PROPERTY_LAST + 1][AnnotationNameLen + 1] = {
  "maxntidx",               // PROPERTY_MAXNTID_X
  "maxntidy",               // PROPERTY_MAXNTID_Y
  "maxntidz",               // PROPERTY_MAXNTID_Z
  "reqntidx",               // PROPERTY_REQNTID_X
  "reqntidy",               // PROPERTY_REQNTID_Y
  "reqntidz",               // PROPERTY_REQNTID_Z
  "minctasm",               // PROPERTY_MINNCTAPERSM
  "texture",                // PROPERTY_ISTEXTURE
  "surface",                // PROPERTY_ISSURFACE
  "sampler",                // PROPERTY_ISSAMPLER
  "rdoimage",               // PROPERTY_ISREADONLY_IMAGE_PARAM
  "wroimage",               // PROPERTY_ISWRITEONLY_IMAGE_PARAM
  "kernel",                 // PROPERTY_ISKERNEL_FUNCTION
  "align",                  // PROPERTY_ALIGN

  // last property
  "proplast",               // PROPERTY_LAST
};

// name of named metadata used for global annotations
#if defined(__GNUC__)
// As this is declared to be static but some of the .cpp files that
// include NVVM.h do not use this array, gcc gives a warning when
// compiling those .cpp files, hence __attribute__((unused)).
__attribute__((unused))
#endif
static const char* NamedMDForAnnotations = "nvvm.annotations";

}


#endif
