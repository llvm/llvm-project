//===-- Results.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver_Results_h__
#define __PerfTestDriver_Results_h__

#include "lldb/lldb-forward.h"
#include <map>
#include <string>
#include <vector>

namespace lldb_perf {

class Results
{
public:
    class Array;
    class Dictionary;
    class Double;
    class String;
    class Unsigned;

    class Result
    {
    public:
        enum class Type
        {
            Invalid,
            Array,
            Dictionary,
            Double,
            String,
            Unsigned
        };

        Result (Type type, const char *name, const char *description) :
            m_name (),
            m_description(),
            m_type (type)
        {
            if (name && name[0])
                m_name = name;
            if (description && description[0])
                m_description = description;
        }

        virtual
        ~Result()
        {
        }

        virtual void
        Write (Results &results) = 0;

        Array *
        GetAsArray ()
        {
            if (m_type == Type::Array)
                return (Array *)this;
            return NULL;
        }
        Dictionary *
        GetAsDictionary ()
        {
            if (m_type == Type::Dictionary)
                return (Dictionary *)this;
            return NULL;
        }
        Double *
        GetAsDouble ()
        {
            if (m_type == Type::Double)
                return (Double *)this;
            return NULL;
        }

        String *
        GetAsString ()
        {
            if (m_type == Type::String)
                return (String *)this;
            return NULL;
        }
        Unsigned *
        GetAsUnsigned ()
        {
            if (m_type == Type::Unsigned)
                return (Unsigned *)this;
            return NULL;
        }
        
        const char *
        GetName() const
        {
            if (m_name.empty())
                return NULL;
            return m_name.c_str();
        }

        const char *
        GetDescription() const
        {
            if (m_description.empty())
                return NULL;
            return m_description.c_str();
        }

        Type
        GetType() const
        {
            return m_type;
        }
    
    protected:
        std::string m_name;
        std::string m_description;
        Type m_type;
    };
    
    typedef std::shared_ptr<Result> ResultSP;

    class Array : public Result
    {
    public:
        Array (const char *name, const char *description) :
            Result (Type::Array, name, description)
        {
        }
        
        virtual
        ~Array()
        {
        }
        
        ResultSP
        Append (const ResultSP &result_sp);

        void
        ForEach (const std::function <bool (const ResultSP &)> &callback);

        virtual void
        Write (Results &results)
        {
        }
    protected:
        typedef std::vector<ResultSP> collection;
        collection m_array;
    };

    class Dictionary : public Result
    {
    public:
        Dictionary () :
            Result (Type::Dictionary, NULL, NULL)
        {
        }

        Dictionary (const char *name, const char *description) :
            Result (Type::Dictionary, name, description)
        {
        }

        virtual
        ~Dictionary()
        {
        }

        virtual void
        Write (Results &results)
        {
        }

        void
        ForEach (const std::function <bool (const std::string &, const ResultSP &)> &callback);
    
        ResultSP
        Add (const char *name, const char *description, const ResultSP &result_sp);
        
        ResultSP
        AddDouble (const char *name, const char *descriptiorn, double value);
        
        ResultSP
        AddUnsigned (const char *name, const char *description, uint64_t value);

        ResultSP
        AddString (const char *name, const char *description, const char *value);

    protected:

        typedef std::map<std::string, ResultSP> collection;
        collection m_dictionary;
    };
    
    class String : public Result
    {
    public:
        String (const char *name, const char *description, const char *value) :
            Result (Type::String, name, description),
            m_string ()
        {
            if (value && value[0])
                m_string = value;
        }

        virtual
        ~String()
        {
        }

        virtual void
        Write (Results &results)
        {
        }

        const char *
        GetValue () const
        {
            return m_string.empty() ? NULL : m_string.c_str();
        }
        
    protected:
        std::string m_string;
    };

    class Double : public Result
    {
    public:
        Double (const char *name, const char *description, double value) :
            Result (Type::Double, name, description),
            m_double (value)
        {
        }
        
        virtual
        ~Double()
        {
        }
        
        virtual void
        Write (Results &results)
        {
        }
        
        double
        GetValue () const
        {
            return m_double;
        }
        
    protected:
        double m_double;
    };

    class Unsigned : public Result
    {
    public:
        Unsigned (const char *name, const char *description, uint64_t value) :
            Result (Type::Unsigned, name, description),
            m_unsigned (value)
        {
        }
        
        virtual
        ~Unsigned()
        {
        }

        virtual void
        Write (Results &results)
        {
        }
        
        uint64_t
        GetValue () const
        {
            return m_unsigned;
        }

    protected:
        uint64_t m_unsigned;
    };

    Results () :
        m_results ()
    {
    }
    
    ~Results()
    {
    }
    
    Dictionary &
    GetDictionary ()
    {
        return m_results;
    }

    void
    Write (const char *path);
    
protected:
    Dictionary m_results;
};
    
} // namespace lldb_perf
#endif // #ifndef __PerfTestDriver_Results_h__
