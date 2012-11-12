//===-- ClangASTImporter.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"

using namespace lldb_private;
using namespace clang;

clang::QualType 
ClangASTImporter::CopyType (clang::ASTContext *dst_ast,
                            clang::ASTContext *src_ast,
                            clang::QualType type)
{
    MinionSP minion_sp (GetMinion(dst_ast, src_ast));
    
    if (minion_sp)
        return minion_sp->Import(type);
    
    return QualType();
}

lldb::clang_type_t
ClangASTImporter::CopyType (clang::ASTContext *dst_ast,
                            clang::ASTContext *src_ast,
                            lldb::clang_type_t type)
{
    return CopyType (dst_ast, src_ast, QualType::getFromOpaquePtr(type)).getAsOpaquePtr();
}

clang::Decl *
ClangASTImporter::CopyDecl (clang::ASTContext *dst_ast,
                            clang::ASTContext *src_ast,
                            clang::Decl *decl)
{
    MinionSP minion_sp;
    
    minion_sp = GetMinion(dst_ast, src_ast);
    
    if (minion_sp)
    {
        clang::Decl *result = minion_sp->Import(decl);
        
        if (!result)
        {
            lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

            if (log)
            {
                lldb::user_id_t user_id;
                ClangASTMetadata *metadata = GetDeclMetadata(decl);
                if (metadata)
                    user_id = metadata->GetUserID();
                
                if (NamedDecl *named_decl = dyn_cast<NamedDecl>(decl))
                    log->Printf("  [ClangASTImporter] WARNING: Failed to import a %s '%s', metadata 0x%llx",
                                decl->getDeclKindName(),
                                named_decl->getNameAsString().c_str(),
                                user_id);
                else
                    log->Printf("  [ClangASTImporter] WARNING: Failed to import a %s, metadata 0x%llx",
                                decl->getDeclKindName(),
                                user_id);
            }
        }
        
        return result;
    }
    
    return NULL;
}

lldb::clang_type_t
ClangASTImporter::DeportType (clang::ASTContext *dst_ctx,
                              clang::ASTContext *src_ctx,
                              lldb::clang_type_t type)
{
    lldb::clang_type_t result = CopyType(dst_ctx, src_ctx, type);
    
    if (!result)
        return NULL;
    
    QualType qual_type = QualType::getFromOpaquePtr(type);
    
    if (const TagType *tag_type = qual_type->getAs<TagType>())
    {
        TagDecl *tag_decl = tag_type->getDecl();
        const TagType *result_tag_type = QualType::getFromOpaquePtr(result)->getAs<TagType>();
        TagDecl *result_tag_decl = result_tag_type->getDecl();
        
        if (tag_decl)
        {
            MinionSP minion_sp (GetMinion (dst_ctx, src_ctx));

            minion_sp->ImportDefinitionTo(result_tag_decl, tag_decl);
            
            ASTContextMetadataSP to_context_md = GetContextMetadata(dst_ctx);
            
            OriginMap::iterator oi = to_context_md->m_origins.find(result_tag_decl);
            
            if (oi != to_context_md->m_origins.end() &&
                oi->second.ctx == src_ctx)
                to_context_md->m_origins.erase(oi);
        }
    }
    
    return result;
}

clang::Decl *
ClangASTImporter::DeportDecl (clang::ASTContext *dst_ctx,
                              clang::ASTContext *src_ctx,
                              clang::Decl *decl)
{
    clang::Decl *result = CopyDecl(dst_ctx, src_ctx, decl);
    
    if (!result)
        return NULL;
    
    ClangASTContext::GetCompleteDecl (src_ctx, decl);

    MinionSP minion_sp (GetMinion (dst_ctx, src_ctx));
    
    if (minion_sp && isa<TagDecl>(decl))
        minion_sp->ImportDefinitionTo(result, decl);
    
    ASTContextMetadataSP to_context_md = GetContextMetadata(dst_ctx);

    OriginMap::iterator oi = to_context_md->m_origins.find(decl);
    
    if (oi != to_context_md->m_origins.end() &&
        oi->second.ctx == src_ctx)
        to_context_md->m_origins.erase(oi);
    
    return result;
}

void
ClangASTImporter::CompleteDecl (clang::Decl *decl)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)    
        log->Printf("    [ClangASTImporter] CompleteDecl called on (%sDecl*)%p",
                    decl->getDeclKindName(),
                    decl);
    
    if (ObjCInterfaceDecl *interface_decl = dyn_cast<ObjCInterfaceDecl>(decl)) 
    {
        if (!interface_decl->getDefinition())
        {
            interface_decl->startDefinition();
            CompleteObjCInterfaceDecl(interface_decl);
        }
    }
    else if (ObjCProtocolDecl *protocol_decl = dyn_cast<ObjCProtocolDecl>(decl)) 
    {
        if (!protocol_decl->getDefinition())
            protocol_decl->startDefinition();
    }
    else if (TagDecl *tag_decl = dyn_cast<TagDecl>(decl)) 
    {
        if (!tag_decl->getDefinition() && !tag_decl->isBeingDefined()) 
        {
            tag_decl->startDefinition();
            CompleteTagDecl(tag_decl);
            tag_decl->setCompleteDefinition(true);
        }
    }
    else
    {
        assert (0 && "CompleteDecl called on a Decl that can't be completed");
    }
}

bool
ClangASTImporter::CompleteTagDecl (clang::TagDecl *decl)
{   
    DeclOrigin decl_origin = GetDeclOrigin(decl);
    
    if (!decl_origin.Valid())
        return false;
    
    if (!ClangASTContext::GetCompleteDecl(decl_origin.ctx, decl_origin.decl))
        return false;
    
    MinionSP minion_sp (GetMinion(&decl->getASTContext(), decl_origin.ctx));
    
    if (minion_sp)
        minion_sp->ImportDefinitionTo(decl, decl_origin.decl);
        
    return true;
}

bool
ClangASTImporter::CompleteTagDeclWithOrigin(clang::TagDecl *decl, clang::TagDecl *origin_decl)
{
    clang::ASTContext *origin_ast_ctx = &origin_decl->getASTContext();
        
    if (!ClangASTContext::GetCompleteDecl(origin_ast_ctx, origin_decl))
        return false;
    
    MinionSP minion_sp (GetMinion(&decl->getASTContext(), origin_ast_ctx));
    
    if (minion_sp)
        minion_sp->ImportDefinitionTo(decl, origin_decl);
        
    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());

    OriginMap &origins = context_md->m_origins;

    origins[decl] = DeclOrigin(origin_ast_ctx, origin_decl);
    
    return true;
}

bool
ClangASTImporter::CompleteObjCInterfaceDecl (clang::ObjCInterfaceDecl *interface_decl)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    DeclOrigin decl_origin = GetDeclOrigin(interface_decl);
    
    if (!decl_origin.Valid())
        return false;
    
    if (!ClangASTContext::GetCompleteDecl(decl_origin.ctx, decl_origin.decl))
        return false;
    
    MinionSP minion_sp (GetMinion(&interface_decl->getASTContext(), decl_origin.ctx));
    
    if (minion_sp)
        minion_sp->ImportDefinitionTo(interface_decl, decl_origin.decl);
        
    return true;
}

ClangASTMetadata *
ClangASTImporter::GetDeclMetadata (const clang::Decl *decl)
{
    DeclOrigin decl_origin = GetDeclOrigin(decl);
    
    if (decl_origin.Valid())
        return ClangASTContext::GetMetadata(decl_origin.ctx, (uintptr_t)decl_origin.decl);
    else
        return ClangASTContext::GetMetadata(&decl->getASTContext(), (uintptr_t)decl);
}

ClangASTImporter::DeclOrigin
ClangASTImporter::GetDeclOrigin(const clang::Decl *decl)
{
    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());
    
    OriginMap &origins = context_md->m_origins;
    
    OriginMap::iterator iter = origins.find(decl);
    
    if (iter != origins.end())
        return iter->second;
    else
        return DeclOrigin();
}

void
ClangASTImporter::SetDeclOrigin (const clang::Decl *decl, clang::Decl *original_decl)
{
    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());
    
    OriginMap &origins = context_md->m_origins;
    
    OriginMap::iterator iter = origins.find(decl);
    
    if (iter != origins.end())
    {
        iter->second.decl = original_decl;
        iter->second.ctx = &original_decl->getASTContext();
    }
    else
    {
        origins[decl] = DeclOrigin(&original_decl->getASTContext(), original_decl);
    }
}

void
ClangASTImporter::RegisterNamespaceMap(const clang::NamespaceDecl *decl, 
                                       NamespaceMapSP &namespace_map)
{
    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());
    
    context_md->m_namespace_maps[decl] = namespace_map;
}

ClangASTImporter::NamespaceMapSP 
ClangASTImporter::GetNamespaceMap(const clang::NamespaceDecl *decl)
{
    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());

    NamespaceMetaMap &namespace_maps = context_md->m_namespace_maps;
    
    NamespaceMetaMap::iterator iter = namespace_maps.find(decl);
    
    if (iter != namespace_maps.end())
        return iter->second;
    else
        return NamespaceMapSP();
}

void 
ClangASTImporter::BuildNamespaceMap(const clang::NamespaceDecl *decl)
{
    assert (decl);
    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());

    const DeclContext *parent_context = decl->getDeclContext();
    const NamespaceDecl *parent_namespace = dyn_cast<NamespaceDecl>(parent_context);
    NamespaceMapSP parent_map;
    
    if (parent_namespace)
        parent_map = GetNamespaceMap(parent_namespace);
    
    NamespaceMapSP new_map;
    
    new_map.reset(new NamespaceMap);
 
    if (context_md->m_map_completer)
    {
        std::string namespace_string = decl->getDeclName().getAsString();
    
        context_md->m_map_completer->CompleteNamespaceMap (new_map, ConstString(namespace_string.c_str()), parent_map);
    }
    
    context_md->m_namespace_maps[decl] = new_map;
}

void 
ClangASTImporter::ForgetDestination (clang::ASTContext *dst_ast)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("    [ClangASTImporter] Forgetting destination (ASTContext*)%p", dst_ast); 

    m_metadata_map.erase(dst_ast);
}

void
ClangASTImporter::ForgetSource (clang::ASTContext *dst_ast, clang::ASTContext *src_ast)
{
    ASTContextMetadataSP md = MaybeGetContextMetadata (dst_ast);
    
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("    [ClangASTImporter] Forgetting source->dest (ASTContext*)%p->(ASTContext*)%p", src_ast, dst_ast); 
    
    if (!md)
        return;
 
    md->m_minions.erase(src_ast);
    
    for (OriginMap::iterator iter = md->m_origins.begin();
         iter != md->m_origins.end();
         )
    {
        if (iter->second.ctx == src_ast)
            md->m_origins.erase(iter++);
        else
            ++iter;
    }
}

ClangASTImporter::MapCompleter::~MapCompleter ()
{
    return;
}

void
ClangASTImporter::Minion::ImportDefinitionTo (clang::Decl *to, clang::Decl *from)
{
    ASTImporter::Imported(from, to);

    ObjCInterfaceDecl *to_objc_interface = dyn_cast<ObjCInterfaceDecl>(to);

    /*
    if (to_objc_interface)
        to_objc_interface->startDefinition();
 
    CXXRecordDecl *to_cxx_record = dyn_cast<CXXRecordDecl>(to);
    
    if (to_cxx_record)
        to_cxx_record->startDefinition();
    */

    ImportDefinition(from);
     
    // If we're dealing with an Objective-C class, ensure that the inheritance has
    // been set up correctly.  The ASTImporter may not do this correctly if the 
    // class was originally sourced from symbols.
    
    if (to_objc_interface)
    {
        do
        {
            ObjCInterfaceDecl *to_superclass = to_objc_interface->getSuperClass();

            if (to_superclass)
                break; // we're not going to override it if it's set
            
            ObjCInterfaceDecl *from_objc_interface = dyn_cast<ObjCInterfaceDecl>(from);
            
            if (!from_objc_interface)
                break;
            
            ObjCInterfaceDecl *from_superclass = from_objc_interface->getSuperClass();
            
            if (!from_superclass)
                break;
            
            Decl *imported_from_superclass_decl = Import(from_superclass);
                
            if (!imported_from_superclass_decl)
                break;
                
            ObjCInterfaceDecl *imported_from_superclass = dyn_cast<ObjCInterfaceDecl>(imported_from_superclass_decl);
            
            if (!imported_from_superclass)
                break;
            
            if (!to_objc_interface->hasDefinition())
                to_objc_interface->startDefinition();
            
            to_objc_interface->setSuperClass(imported_from_superclass);
        }
        while (0);
    }
}

clang::Decl 
*ClangASTImporter::Minion::Imported (clang::Decl *from, clang::Decl *to)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
    if (log)
    {
        lldb::user_id_t user_id;
        ClangASTMetadata *metadata = m_master.GetDeclMetadata(from);
        if (metadata)
            user_id = metadata->GetUserID();
        
        if (NamedDecl *from_named_decl = dyn_cast<clang::NamedDecl>(from))
        {
            std::string name_string;
            llvm::raw_string_ostream name_stream(name_string);
            from_named_decl->printName(name_stream);
            name_stream.flush();
            
            log->Printf("    [ClangASTImporter] Imported (%sDecl*)%p, named %s (from (Decl*)%p), metadata 0x%llx",
                        from->getDeclKindName(),
                        to,
                        name_string.c_str(),
                        from,
                        user_id);
        }
        else
        {
            log->Printf("    [ClangASTImporter] Imported (%sDecl*)%p (from (Decl*)%p), metadata 0x%llx",
                        from->getDeclKindName(),
                        to,
                        from,
                        user_id);
        }
    }

    ASTContextMetadataSP to_context_md = m_master.GetContextMetadata(&to->getASTContext());
    ASTContextMetadataSP from_context_md = m_master.MaybeGetContextMetadata(m_source_ctx);
    
    if (from_context_md)
    {
        OriginMap &origins = from_context_md->m_origins;
        
        OriginMap::iterator origin_iter = origins.find(from);
        
        if (origin_iter != origins.end())
        {
            to_context_md->m_origins[to] = origin_iter->second;
            
            MinionSP direct_completer = m_master.GetMinion(&to->getASTContext(), origin_iter->second.ctx);
            
            if (direct_completer.get() != this)
                direct_completer->ASTImporter::Imported(origin_iter->second.decl, to);
            
            if (log)
                log->Printf("    [ClangASTImporter] Propagated origin (Decl*)%p/(ASTContext*)%p from (ASTContext*)%p to (ASTContext*)%p",
                            origin_iter->second.decl,
                            origin_iter->second.ctx,
                            &from->getASTContext(),
                            &to->getASTContext());
        }
        else
        {
            to_context_md->m_origins[to] = DeclOrigin(m_source_ctx, from);
            
            if (log)
                log->Printf("    [ClangASTImporter] Decl has no origin information in (ASTContext*)%p",
                            &from->getASTContext());
        }
        
        if (clang::NamespaceDecl *to_namespace = dyn_cast<clang::NamespaceDecl>(to))
        {
            clang::NamespaceDecl *from_namespace = dyn_cast<clang::NamespaceDecl>(from);
            
            NamespaceMetaMap &namespace_maps = from_context_md->m_namespace_maps;
            
            NamespaceMetaMap::iterator namespace_map_iter = namespace_maps.find(from_namespace);
            
            if (namespace_map_iter != namespace_maps.end())
                to_context_md->m_namespace_maps[to_namespace] = namespace_map_iter->second;
        }
    }
    else
    {
        to_context_md->m_origins[to] = DeclOrigin (m_source_ctx, from);
        
        if (log)
            log->Printf("    [ClangASTImporter] Sourced origin (Decl*)%p/(ASTContext*)%p into (ASTContext*)%p",
                        from,
                        m_source_ctx,
                        &to->getASTContext());
    }
        
    if (TagDecl *from_tag_decl = dyn_cast<TagDecl>(from))
    {
        TagDecl *to_tag_decl = dyn_cast<TagDecl>(to);
        
        to_tag_decl->setHasExternalLexicalStorage();
        to_tag_decl->setMustBuildLookupTable();
                                
        if (log)
            log->Printf("    [ClangASTImporter] To is a TagDecl - attributes %s%s [%s->%s]",
                        (to_tag_decl->hasExternalLexicalStorage() ? " Lexical" : ""),
                        (to_tag_decl->hasExternalVisibleStorage() ? " Visible" : ""),
                        (from_tag_decl->isCompleteDefinition() ? "complete" : "incomplete"),
                        (to_tag_decl->isCompleteDefinition() ? "complete" : "incomplete"));
        
        to_tag_decl = NULL;
    }
    
    if (isa<NamespaceDecl>(from))
    {
        NamespaceDecl *to_namespace_decl = dyn_cast<NamespaceDecl>(to);
        
        m_master.BuildNamespaceMap(to_namespace_decl);
        
        to_namespace_decl->setHasExternalVisibleStorage();
    }
    
    if (isa<ObjCInterfaceDecl>(from))
    {
        ObjCInterfaceDecl *to_interface_decl = dyn_cast<ObjCInterfaceDecl>(to);
                
        to_interface_decl->setHasExternalLexicalStorage();
        to_interface_decl->setHasExternalVisibleStorage();
                 
        /*to_interface_decl->setExternallyCompleted();*/
                
        if (log)
            log->Printf("    [ClangASTImporter] To is an ObjCInterfaceDecl - attributes %s%s%s",
                        (to_interface_decl->hasExternalLexicalStorage() ? " Lexical" : ""),
                        (to_interface_decl->hasExternalVisibleStorage() ? " Visible" : ""),
                        (to_interface_decl->hasDefinition() ? " HasDefinition" : ""));
    }
    
    return clang::ASTImporter::Imported(from, to);
}
