//===-- AMDGPUNoteType.h - AMDGPU ELF PT_NOTE section info-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// Enums and constants for AMDGPU PT_NOTE sections.
///
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUPTNOTE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUPTNOTE_H

namespace llvm {
namespace AMDGPU {

namespace ElfNote {

const char SectionName[] = ".note";

const char NoteNameV2[] = "AMD";
const char NoteNameV3[] = "AMDGPU";

// TODO: Remove this file once we drop code object v2.
enum NoteType{
    NT_AMDGPU_HSA_RESERVED_0 = 0,
    NT_AMDGPU_HSA_CODE_OBJECT_VERSION = 1,
    NT_AMDGPU_HSA_HSAIL = 2,
    NT_AMDGPU_HSA_ISA = 3,
    NT_AMDGPU_HSA_PRODUCER = 4,
    NT_AMDGPU_HSA_PRODUCER_OPTIONS = 5,
    NT_AMDGPU_HSA_EXTENSION = 6,
    NT_AMDGPU_HSA_RESERVED_7 = 7,
    NT_AMDGPU_HSA_RESERVED_8 = 8,
    NT_AMDGPU_HSA_RESERVED_9 = 9,
    NT_AMDGPU_HSA_HLDEBUG_DEBUG = 101,
    NT_AMDGPU_HSA_HLDEBUG_TARGET = 102
};

} // End namespace ElfNote
} // End namespace AMDGPU
} // End namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNOTETYPE_H
