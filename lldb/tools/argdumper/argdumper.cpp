//===-- argdumper.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/JSON.h"
#include "lldb/Utility/StreamString.h"

#include <iostream>

using namespace lldb_private;

int main(int argc, char *argv[]) {
  JSONArray::SP arguments(new JSONArray());
  for (int i = 1; i < argc; i++) {
    arguments->AppendObject(JSONString::SP(new JSONString(argv[i])));
  }

  JSONObject::SP object(new JSONObject());
  object->SetObject("arguments", arguments);

  StreamString ss;

  object->Write(ss);

  std::cout << ss.GetData() << std::endl;

  return 0;
}
