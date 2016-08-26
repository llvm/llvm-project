//===-- OCamlLanguage.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <string.h>
// C++ Includes
#include <functional>
#include <mutex>

// Other libraries and framework includes
#include "llvm/ADT/StringRef.h"

// Project includes
#include "OCamlLanguage.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/OCamlASTContext.h"

using namespace lldb;
using namespace lldb_private;

void
OCamlLanguage::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(), "OCaml Language", CreateInstance);
}

void
OCamlLanguage::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString
OCamlLanguage::GetPluginNameStatic()
{
    static ConstString g_name("OCaml");
    return g_name;
}

lldb_private::ConstString
OCamlLanguage::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
OCamlLanguage::GetPluginVersion()
{
    return 1;
}

Language *
OCamlLanguage::CreateInstance(lldb::LanguageType language)
{
    if (language == eLanguageTypeOCaml)
        return new OCamlLanguage();
    return nullptr;
}

bool
OCamlLanguage::IsNilReference(ValueObject &valobj)
{
    if (!valobj.GetCompilerType().IsReferenceType())
        return false;

    // If we failed to read the value then it is not a nil reference.
    return valobj.GetValueAsUnsigned(UINT64_MAX) == 0;
}

