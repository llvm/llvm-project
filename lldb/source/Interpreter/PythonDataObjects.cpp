//===-- PythonDataObjects.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In order to guarantee correct working with Python, Python.h *MUST* be
// the *FIRST* header file included here.
#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#include <stdio.h>

#include "lldb/Core/Stream.h"
#include "lldb/Host/File.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Interpreter/ScriptInterpreter.h"

using namespace lldb_private;
using namespace lldb;

//----------------------------------------------------------------------
// PythonObject
//----------------------------------------------------------------------
PythonObject::PythonObject (const lldb::ScriptInterpreterObjectSP &script_object_sp) :
    m_py_obj (NULL)
{
    if (script_object_sp)
        Reset ((PyObject *)script_object_sp->GetObject());
}

void
PythonObject::Dump (Stream &strm) const
{
    if (m_py_obj)
    {
        FILE *file = ::tmpfile();
        if (file)
        {
            ::PyObject_Print (m_py_obj, file, 0);
            const long length = ftell (file);
            if (length)
            {
                ::rewind(file);
                std::vector<char> file_contents (length,'\0');
                const size_t length_read = ::fread (file_contents.data(), 1, file_contents.size(), file);
                if (length_read > 0)
                    strm.Write (file_contents.data(), length_read);
            }
            ::fclose (file);
        }
    }
    else
        strm.PutCString ("NULL");
}

//----------------------------------------------------------------------
// PythonString
//----------------------------------------------------------------------

PythonString::PythonString (PyObject *py_obj) :
    PythonObject(py_obj)
{
}

PythonString::PythonString (const PythonObject &object) :
    PythonObject(object.GetPythonObject())
{
}

PythonString::PythonString (const lldb::ScriptInterpreterObjectSP &script_object_sp) :
    PythonObject (script_object_sp)
{
}

PythonString::PythonString (const char* string) :
    PythonObject(PyString_FromString(string))
{
}

PythonString::PythonString () :
    PythonObject()
{
}

PythonString::~PythonString ()
{
}

bool
PythonString::Reset (PyObject *py_obj)
{
    if (py_obj && PyString_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(NULL);
    return py_obj == NULL;
}

const char*
PythonString::GetString() const
{
    if (m_py_obj)
        return PyString_AsString(m_py_obj);
    return NULL;
}

size_t
PythonString::GetSize() const
{
    if (m_py_obj)
        return PyString_Size(m_py_obj);
    return 0;
}

void
PythonString::SetString (const char* string)
{
    PythonObject::Reset(PyString_FromString(string));
}

//----------------------------------------------------------------------
// PythonInteger
//----------------------------------------------------------------------

PythonInteger::PythonInteger (PyObject *py_obj) :
    PythonObject(py_obj)
{
}

PythonInteger::PythonInteger (const PythonObject &object) :
    PythonObject(object.GetPythonObject())
{
}

PythonInteger::PythonInteger (const lldb::ScriptInterpreterObjectSP &script_object_sp) :
    PythonObject (script_object_sp)
{
}

PythonInteger::PythonInteger (int64_t value) :
    PythonObject(PyInt_FromLong(value))
{
}


PythonInteger::~PythonInteger ()
{
}

bool
PythonInteger::Reset (PyObject *py_obj)
{
    if (py_obj && PyInt_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(NULL);
    return py_obj == NULL;
}

int64_t
PythonInteger::GetInteger()
{
    if (m_py_obj)
        return PyInt_AsLong(m_py_obj);
    else
        return UINT64_MAX;
}

void
PythonInteger::SetInteger (int64_t value)
{
    PythonObject::Reset(PyInt_FromLong(value));
}

//----------------------------------------------------------------------
// PythonList
//----------------------------------------------------------------------

PythonList::PythonList () :
    PythonObject(PyList_New(0))
{
}

PythonList::PythonList (uint32_t count) :
    PythonObject(PyList_New(count))
{
}

PythonList::PythonList (PyObject *py_obj) :
    PythonObject(py_obj)
{
}


PythonList::PythonList (const PythonObject &object) :
    PythonObject(object.GetPythonObject())
{
}

PythonList::PythonList (const lldb::ScriptInterpreterObjectSP &script_object_sp) :
    PythonObject (script_object_sp)
{
}

PythonList::~PythonList ()
{
}

bool
PythonList::Reset (PyObject *py_obj)
{
    if (py_obj && PyList_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(NULL);
    return py_obj == NULL;
}

uint32_t
PythonList::GetSize()
{
    if (m_py_obj)
        return PyList_GET_SIZE(m_py_obj);
    return 0;
}

PythonObject
PythonList::GetItemAtIndex (uint32_t index)
{
    if (m_py_obj)
        return PythonObject(PyList_GetItem(m_py_obj, index));
    return NULL;
}

void
PythonList::SetItemAtIndex (uint32_t index, const PythonObject & object)
{
    if (m_py_obj && object)
        PyList_SetItem(m_py_obj, index, object.GetPythonObject());
}

void
PythonList::AppendItem (const PythonObject &object)
{
    if (m_py_obj && object)
        PyList_Append(m_py_obj, object.GetPythonObject());
}

//----------------------------------------------------------------------
// PythonDictionary
//----------------------------------------------------------------------

PythonDictionary::PythonDictionary () :
    PythonObject(PyDict_New())
{
}

PythonDictionary::PythonDictionary (PyObject *py_obj) :
    PythonObject(py_obj)
{
}


PythonDictionary::PythonDictionary (const PythonObject &object) :
    PythonObject(object.GetPythonObject())
{
}

PythonDictionary::PythonDictionary (const lldb::ScriptInterpreterObjectSP &script_object_sp) :
    PythonObject (script_object_sp)
{
}

PythonDictionary::~PythonDictionary ()
{
}

bool
PythonDictionary::Reset (PyObject *py_obj)
{
    if (py_obj && PyDict_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(NULL);
    return py_obj == NULL;
}

uint32_t
PythonDictionary::GetSize()
{
    if (m_py_obj)
        return PyDict_Size(m_py_obj);
    return 0;
}

PythonObject
PythonDictionary::GetItemForKey (const char *key) const
{
    if (key && key[0])
    {
        PythonString python_key(key);
        return GetItemForKey(python_key);
    }
    return NULL;
}


PythonObject
PythonDictionary::GetItemForKey (const PythonString &key) const
{
    if (m_py_obj && key)
        return PythonObject(PyDict_GetItem(m_py_obj, key.GetPythonObject()));
    return PythonObject();
}


const char *
PythonDictionary::GetItemForKeyAsString (const PythonString &key, const char *fail_value) const
{
    if (m_py_obj && key)
    {
        PyObject *py_obj = PyDict_GetItem(m_py_obj, key.GetPythonObject());
        if (py_obj && PyString_Check(py_obj))
            return PyString_AsString(py_obj);
    }
    return fail_value;
}

int64_t
PythonDictionary::GetItemForKeyAsInteger (const PythonString &key, int64_t fail_value) const
{
    if (m_py_obj && key)
    {
        PyObject *py_obj = PyDict_GetItem(m_py_obj, key.GetPythonObject());
        if (py_obj)
        {
            if (PyInt_Check(py_obj))
                return PyInt_AsLong(py_obj);

            if (PyLong_Check(py_obj))
                return PyLong_AsLong(py_obj);
        }
    }
    return fail_value;
}

PythonList
PythonDictionary::GetKeys () const
{
    if (m_py_obj)
        return PythonList(PyDict_Keys(m_py_obj));
    return PythonList();
}

PythonString
PythonDictionary::GetKeyAtPosition (uint32_t pos) const
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (m_py_obj)
    {
        while (PyDict_Next(m_py_obj, &pos_iter, &key, &value))
        {
            if (pos-- == 0)
                return PythonString(key);
        }
    }
    return PythonString();
}

PythonObject
PythonDictionary::GetValueAtPosition (uint32_t pos) const
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (!m_py_obj)
        return NULL;
    
    while (PyDict_Next(m_py_obj, &pos_iter, &key, &value)) {
        if (pos-- == 0)
            return PythonObject(value);
    }
    return PythonObject();
}

void
PythonDictionary::SetItemForKey (const PythonString &key, const PythonObject &value)
{
    if (m_py_obj && key && value)
        PyDict_SetItem(m_py_obj, key.GetPythonObject(), value.GetPythonObject());
}

#endif
