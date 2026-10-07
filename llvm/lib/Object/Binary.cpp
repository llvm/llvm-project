//===- Binary.cpp - A generic binary file ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Binary class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/Minidump.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Object/TapiUniversal.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <system_error>

using namespace llvm;
using namespace object;

Binary::~Binary() = default;

Binary::Binary(unsigned int Type, MemoryBufferRef Source)
    : TypeID(Type), Data(Source) {}

StringRef Binary::getData() const { return Data.getBuffer(); }

StringRef Binary::getFileName() const { return Data.getBufferIdentifier(); }

MemoryBufferRef Binary::getMemoryBufferRef() const { return Data; }
