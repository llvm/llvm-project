//===------ dxcontainer2yaml.cpp - obj2yaml conversion tool -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/ObjectYAML/DXContainerYAML.h"
#include "llvm/Support/Error.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::object;

static Expected<DXContainerYAML::Object *>
dumpDXContainer(MemoryBufferRef Source) {
  assert(file_magic::dxcontainer_object == identify_magic(Source.getBuffer()));

  Expected<DXContainer> ExDXC = DXContainer::create(Source);
  if (!ExDXC)
    return ExDXC.takeError();
  DXContainer Container = *ExDXC;

  std::unique_ptr<DXContainerYAML::Object> Obj =
      std::make_unique<DXContainerYAML::Object>();

  for (uint8_t Byte : Container.getHeader().FileHash.Digest)
    Obj->Header.Hash.push_back(Byte);
  Obj->Header.Version.Major = Container.getHeader().Version.Major;
  Obj->Header.Version.Minor = Container.getHeader().Version.Minor;
  Obj->Header.FileSize = Container.getHeader().FileSize;
  Obj->Header.PartCount = Container.getHeader().PartCount;

  Obj->Header.PartOffsets = std::vector<uint32_t>();
  for (const auto P : Container) {
    Obj->Header.PartOffsets->push_back(P.Offset);
    Obj->Parts.push_back(
        DXContainerYAML::Part(P.Part.getName().str(), P.Part.Size));
    DXContainerYAML::Part &NewPart = Obj->Parts.back();
    dxbc::PartType PT = dxbc::parsePartType(P.Part.getName());
    switch (PT) {
    case dxbc::PartType::DXIL: {
      std::optional<DXContainer::DXILData> DXIL = Container.getDXIL();
      assert(DXIL && "Since we are iterating and found a DXIL part, "
                     "this should never not have a value");
      NewPart.Program = DXContainerYAML::DXILProgram{
          DXIL->first.MajorVersion,
          DXIL->first.MinorVersion,
          DXIL->first.ShaderKind,
          DXIL->first.Size,
          DXIL->first.Bitcode.MajorVersion,
          DXIL->first.Bitcode.MinorVersion,
          DXIL->first.Bitcode.Offset,
          DXIL->first.Bitcode.Size,
          std::vector<llvm::yaml::Hex8>(
              DXIL->second, DXIL->second + DXIL->first.Bitcode.Size)};
      break;
    }
    case dxbc::PartType::SFI0: {
      std::optional<uint64_t> Flags = Container.getShaderFlags();
      // Omit the flags in the YAML if they are missing or zero.
      if (Flags && *Flags > 0)
        NewPart.Flags = DXContainerYAML::ShaderFlags(*Flags);
      break;
    }
    case dxbc::PartType::HASH: {
      std::optional<dxbc::ShaderHash> Hash = Container.getShaderHash();
      if (Hash && Hash->isPopulated())
        NewPart.Hash = DXContainerYAML::ShaderHash(*Hash);
      break;
    }
    case dxbc::PartType::PSV0: {
      const auto &PSVInfo = Container.getPSVInfo();
      if (!PSVInfo)
        break;
      if (const auto *P =
              std::get_if<dxbc::PSV::v0::RuntimeInfo>(&PSVInfo->getInfo())) {
        if (!Container.getDXIL())
          break;
        NewPart.Info =
            DXContainerYAML::PSVInfo(P, Container.getDXIL()->first.ShaderKind);
      } else if (const auto *P = std::get_if<dxbc::PSV::v1::RuntimeInfo>(
                     &PSVInfo->getInfo()))
        NewPart.Info = DXContainerYAML::PSVInfo(P);
      else if (const auto *P =
                   std::get_if<dxbc::PSV::v2::RuntimeInfo>(&PSVInfo->getInfo()))
        NewPart.Info = DXContainerYAML::PSVInfo(P);
      for (auto Res : PSVInfo->getResources())
        NewPart.Info->Resources.push_back(Res);
      break;
    }
    case dxbc::PartType::Unknown:
      break;
    }
  }

  return Obj.release();
}

llvm::Error dxcontainer2yaml(llvm::raw_ostream &Out,
                             llvm::MemoryBufferRef Source) {
  Expected<DXContainerYAML::Object *> YAMLOrErr = dumpDXContainer(Source);
  if (!YAMLOrErr)
    return YAMLOrErr.takeError();

  std::unique_ptr<DXContainerYAML::Object> YAML(YAMLOrErr.get());
  yaml::Output Yout(Out);
  Yout << *YAML;

  return Error::success();
}
