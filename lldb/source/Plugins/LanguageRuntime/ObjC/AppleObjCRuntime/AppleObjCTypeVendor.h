//===-- AppleObjCSymbolVendor.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCSymbolVendor_h_
#define liblldb_AppleObjCSymbolVendor_h_

// C Includes
// C++ Includes

#include <map>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/TypeVendor.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

namespace lldb_private {

class AppleObjCExternalASTSource;
    
class AppleObjCTypeVendor : public TypeVendor
{
public:
    AppleObjCTypeVendor(ObjCLanguageRuntime &runtime);
    
    virtual uint32_t
    FindTypes (const ConstString &name,
               bool append,
               uint32_t max_matches,
               std::vector <ClangASTType> &types);
    
    virtual clang::ASTContext *
    GetClangASTContext ()
    {
        return m_ast_ctx.getASTContext();
    }
    
    friend class AppleObjCExternalASTSource;
private:
    clang::ObjCInterfaceDecl   *GetDeclForISA(ObjCLanguageRuntime::ObjCISA isa);
    bool                        FinishDecl(clang::ObjCInterfaceDecl *decl);
    
    ObjCLanguageRuntime        &m_runtime;
    ClangASTContext             m_ast_ctx;
    AppleObjCExternalASTSource *m_external_source;
    
    typedef llvm::DenseMap<ObjCLanguageRuntime::ObjCISA, clang::ObjCInterfaceDecl *> ISAToInterfaceMap;

    ISAToInterfaceMap           m_isa_to_interface;
};

} // namespace lldb_private

#endif  // liblldb_AppleObjCSymbolVendor_h_
