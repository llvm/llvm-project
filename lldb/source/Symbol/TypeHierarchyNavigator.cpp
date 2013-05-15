//===-- TypeHierarchyNavigator.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/TypeHierarchyNavigator.h"

using namespace lldb;
using namespace lldb_private;

bool
TypeHierarchyNavigator::LoopThrough(TypeHierarchyNavigatorCallback callback,
                                    void* callback_baton)
{
    return LoopThrough(m_root_type,
                callback,
                eRootType,
                (callback_baton ? callback_baton : m_default_callback_baton));
}

bool
TypeHierarchyNavigator::LoopThrough(const clang::QualType& qual_type,
                                    TypeHierarchyNavigatorCallback callback,
                                    RelationshipToCurrentType reason_why_here,
                                    void* callback_baton)
{
    if (qual_type.isNull())
        return true;
    clang::QualType type = qual_type.getUnqualifiedType();
    type.removeLocalConst(); type.removeLocalVolatile(); type.removeLocalRestrict();
    const clang::Type* typePtr = type.getTypePtrOrNull();
    if (!typePtr)
        return true;
    if (!callback(type, reason_why_here, callback_baton))
        return false;
    // look for a "base type", whatever that means
    if (typePtr->isReferenceType())
    {
        if (LoopThrough(type.getNonReferenceType(), callback, eStrippedReference, callback_baton) == false)
            return false;
    }
    if (typePtr->isPointerType())
    {
        if (LoopThrough(typePtr->getPointeeType(), callback, eStrippedPointer, callback_baton) == false)
            return false;
    }
    if (typePtr->isObjCObjectPointerType())
    {
        /*
         for some reason, C++ can quite easily obtain the type hierarchy for a ValueObject
         even if the VO represent a pointer-to-class, as long as the typePtr is right
         Objective-C on the other hand cannot really complete an @interface when
         the VO refers to a pointer-to-@interface
         */
        Error error;
        ValueObject* target = m_value_object.Dereference(error).get();
        if (error.Fail() || !target)
            return true;
        if (LoopThrough(typePtr->getPointeeType(), callback, eStrippedPointer, callback_baton) == false)
            return false;
    }
    const clang::ObjCObjectType *objc_class_type = typePtr->getAs<clang::ObjCObjectType>();
    if (objc_class_type)
    {
        clang::ASTContext *ast = m_value_object.GetClangAST();
        if (ClangASTContext::GetCompleteType(ast, m_value_object.GetClangType()) && !objc_class_type->isObjCId())
        {
            clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            if (class_interface_decl)
            {
                clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                if (superclass_interface_decl)
                {
                    clang::QualType ivar_qual_type(ast->getObjCInterfaceType(superclass_interface_decl));
                    return LoopThrough(ivar_qual_type, callback, eObjCBaseClass, callback_baton);
                }
            }
        }
    }
    // for C++ classes, navigate up the hierarchy
    if (typePtr->isRecordType())
    {
        clang::CXXRecordDecl* record = typePtr->getAsCXXRecordDecl();
        if (record)
        {
            if (!record->hasDefinition())
                ClangASTContext::GetCompleteType(m_value_object.GetClangAST(), m_value_object.GetClangType());
            if (record->hasDefinition())
            {
                clang::CXXRecordDecl::base_class_iterator pos,end;
                if ( record->getNumBases() > 0)
                {
                    end = record->bases_end();
                    for (pos = record->bases_begin(); pos != end; pos++)
                        if (LoopThrough(pos->getType(), callback, eCXXBaseClass, callback_baton) == false)
                            return false;
                }
                if (record->getNumVBases() > 0)
                {
                    end = record->vbases_end();
                    for (pos = record->vbases_begin(); pos != end; pos++)
                        if (LoopThrough(pos->getType(), callback, eCXXVBaseClass, callback_baton) == false)
                            return false;
                }
            }
        }
    }
    // try to strip typedef chains
    const clang::TypedefType* type_tdef = type->getAs<clang::TypedefType>();
    if (type_tdef)
        return LoopThrough(type_tdef->getDecl()->getUnderlyingType(), callback, eStrippedTypedef, callback_baton);
    else
        return true;
}
