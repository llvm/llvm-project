//===-- SwiftMetatype.cpp ---------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Mangled.h"
#include "SwiftMetatype.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

#include "swift/AST/Type.h"
#include "swift/AST/Types.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

static bool
FormatMetadata (SwiftLanguageRuntime::MetadataSP metadata_sp,
                SwiftLanguageRuntime *swift_runtime,
                SwiftASTContext *swift_ast_ctx,
                Stream& stream,
                const TypeSummaryOptions& options)
{
    if (!metadata_sp)
        return false;
    if (SwiftLanguageRuntime::NominalTypeMetadata* nominal_metadata = llvm::dyn_cast<SwiftLanguageRuntime::NominalTypeMetadata>(metadata_sp.get()))
    {
        Mangled mangled(ConstString(nominal_metadata->GetMangledName().c_str()));
        stream.Printf("%s", mangled.GetDemangledName(lldb::eLanguageTypeSwift).AsCString("<unknown type>"));
        return true;
    }
    else if (SwiftLanguageRuntime::MetatypeMetadata *metatype_metadata = llvm::dyn_cast<SwiftLanguageRuntime::MetatypeMetadata>(metadata_sp.get()))
    {
        if (metatype_metadata->GetInstanceMetadata())
        {
            stream.Printf("metatype for ");
            return FormatMetadata(metatype_metadata->GetInstanceMetadata(),
                                  swift_runtime,
                                  swift_ast_ctx,
                                  stream,
                                  options);
        }
    }
    else
    {
        Error error;
        CompilerType realizedtype(swift_runtime->GetTypeForMetadata(metadata_sp, swift_ast_ctx, error));
        if (error.Fail() || realizedtype.IsValid() == false)
            return false;
        stream.Printf("%s", realizedtype.GetDisplayTypeName().AsCString("<unknown type>"));
        return true;
    }
    return false;
}

bool
lldb_private::formatters::swift::SwiftMetatype_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    lldb::addr_t metadata_ptr = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (metadata_ptr == LLDB_INVALID_ADDRESS || metadata_ptr == 0)
    {
        CompilerType compiler_metatype_type(valobj.GetCompilerType());
        CompilerType instancetype(compiler_metatype_type.GetInstanceType());
        const char* ptr = instancetype.GetDisplayTypeName().AsCString(nullptr);
        if (ptr && *ptr)
        {
            stream.Printf("%s", ptr);
            return true;
        }
    }
    else
    {
        auto swift_runtime = valobj.GetProcessSP()->GetSwiftLanguageRuntime();
        if (!swift_runtime)
            return false;
        SwiftLanguageRuntime::MetadataSP metadata_sp = swift_runtime->GetMetadataForLocation(metadata_ptr);
        Error error;
        SwiftASTContext *swift_ast_ctx = valobj.GetTargetSP()->GetScratchSwiftASTContext(error);
        if (swift_ast_ctx)
        {
            if (!swift_ast_ctx->HasFatalErrors())
                return FormatMetadata(metadata_sp, swift_runtime, swift_ast_ctx, stream, options);
            else
            {
                stream.Printf ("Error getting AST context: %s.", swift_ast_ctx->GetFatalErrors().AsCString());
            }
        }
        else
        {
            stream.Printf("Unknown error getting AST Context");
        }
    }
    return false;
}

