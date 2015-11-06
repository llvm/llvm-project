//===-- SwiftOptional.h -----------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftOptional_h_
#define liblldb_SwiftOptional_h_

#include "lldb/lldb-forward.h"

#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

namespace lldb_private
{
    namespace formatters
    {
        namespace swift
        {
            struct SwiftOptionalSummaryProvider : public TypeSummaryImpl
            {
                SwiftOptionalSummaryProvider (const TypeSummaryImpl::Flags& flags) :
                TypeSummaryImpl(TypeSummaryImpl::Kind::eInternal,TypeSummaryImpl::Flags())
                {
                }
                
                virtual
                ~SwiftOptionalSummaryProvider ()
                {
                }
                
                virtual bool
                FormatObject (ValueObject *valobj,
                              std::string& dest,
                              const TypeSummaryOptions& options);
                
                virtual std::string
                GetDescription ();
                
                virtual bool
                IsScripted ()
                {
                    return false;
                }
                
                virtual bool
                DoesPrintChildren (ValueObject* valobj) const;

                virtual bool
                DoesPrintValue (ValueObject* valobj) const;
            private:
                DISALLOW_COPY_AND_ASSIGN(SwiftOptionalSummaryProvider);
            };
            
            bool
            SwiftOptional_SummaryProvider (ValueObject& valobj, Stream& stream);
            
            class SwiftOptionalSyntheticFrontEnd : public SyntheticChildrenFrontEnd
            {
            public:
                SwiftOptionalSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
                
                virtual size_t
                CalculateNumChildren ();
                
                virtual lldb::ValueObjectSP
                GetChildAtIndex (size_t idx);
                
                virtual bool
                Update();
                
                virtual bool
                MightHaveChildren ();
                
                virtual size_t
                GetIndexOfChildWithName (const ConstString &name);
                
                virtual lldb::ValueObjectSP
                GetSyntheticValue ();
                
                virtual
                ~SwiftOptionalSyntheticFrontEnd () = default;
            private:
                bool m_is_none;
                bool m_children;
                ValueObject* m_some;
                
                bool
                IsEmpty () const;
            };
            
            SyntheticChildrenFrontEnd* SwiftOptionalSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
            SyntheticChildrenFrontEnd* SwiftUncheckedOptionalSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
            
        }
    }
}
    
#endif // liblldb_SwiftOptional_h_
