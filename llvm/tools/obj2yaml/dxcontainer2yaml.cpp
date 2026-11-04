//===------ dxcontainer2yaml.cpp - obj2yaml conversion tool -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/MC/DXContainerInfo.h"
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

  auto DXCYaml = DXContainerYAML::fromDXContainer(Container);
  if (!DXCYaml)
    return DXCYaml.takeError();

  return DXCYaml.get().release();
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
