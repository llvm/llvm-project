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

#include "lldb/Interpreter/PythonDataObjects.h"

using namespace lldb_private;
using namespace lldb;

PythonDataObject::PythonDataObject (PyObject* object) :
    m_object(object)
{
}

PythonDataObject::PythonDataObject () :
    m_object()
{
}

PythonDataObject::~PythonDataObject ()
{
}

PythonDataString
PythonDataObject::GetStringObject ()
{
    return PythonDataString(GetPythonObject());
}
    
PythonDataInteger
PythonDataObject::GetIntegerObject ()
{
    return PythonDataInteger(GetPythonObject());
}

PythonDataArray
PythonDataObject::GetArrayObject()
{
    return PythonDataArray(GetPythonObject());
}

PythonDataDictionary
PythonDataObject::GetDictionaryObject()
{
    return PythonDataDictionary(GetPythonObject());
}

PythonDataInteger::PythonDataInteger (bool create_empty) :
    m_object(create_empty ? PyInt_FromLong(0) : NULL)
{
}

PythonDataInteger::PythonDataInteger (PyObject* object) :
    m_object(object)
{
    if (object && !PyInt_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataInteger::PythonDataInteger (int64_t value) :
    m_object(PyInt_FromLong(value))
{
}


PythonDataInteger::~PythonDataInteger ()
{
}

int64_t
PythonDataInteger::GetInteger()
{
    if (m_object)
        return PyInt_AsLong(GetPythonObject());
    else
        return UINT64_MAX;
}

void
PythonDataInteger::SetInteger (int64_t value)
{
    m_object.Reset(PyInt_FromLong(value));
}

PythonDataString::PythonDataString (bool create_empty) :
    m_object(create_empty ? PyString_FromString("") : NULL)
{
}

PythonDataString::PythonDataString (PyObject* object) :
    m_object(object)
{
    if (object && !PyString_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataString::PythonDataString (const char* string) :
    m_object(PyString_FromString(string))
{
}

PythonDataString::~PythonDataString ()
{
}

const char*
PythonDataString::GetString() const
{
    if (m_object)
        return PyString_AsString(GetPythonObject());
    return NULL;
}

size_t
PythonDataString::GetSize() const
{
    if (m_object)
        return PyString_Size(GetPythonObject());
    return 0;
}

void
PythonDataString::SetString (const char* string)
{
    m_object.Reset(PyString_FromString(string));
}

PythonDataArray::PythonDataArray (bool create_empty) :
    m_object(create_empty ? PyList_New(0) : NULL)
{
}

PythonDataArray::PythonDataArray (uint32_t count) :
    m_object(PyList_New(count))
{
}

PythonDataArray::PythonDataArray (PyObject* object) :
    m_object(object)
{
    if (object && !PyList_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataArray::~PythonDataArray ()
{
}

uint32_t
PythonDataArray::GetSize()
{
    if (m_object)
        return PyList_GET_SIZE(GetPythonObject());
    return 0;
}

PythonDataObject
PythonDataArray::GetItemAtIndex (uint32_t index)
{
    if (m_object)
        return PythonDataObject(PyList_GetItem(GetPythonObject(), index));
    return NULL;
}

void
PythonDataArray::SetItemAtIndex (uint32_t index, const PythonDataObject & object)
{
    if (m_object && object)
        PyList_SetItem(GetPythonObject(), index, object.GetPythonObject());
}

void
PythonDataArray::AppendItem (const PythonDataObject &object)
{
    if (m_object && object)
        PyList_Append(GetPythonObject(), object.GetPythonObject());
}

PythonDataDictionary::PythonDataDictionary (bool create_empty) :
    m_object(create_empty ? PyDict_New() : NULL)
{
}

PythonDataDictionary::PythonDataDictionary (PyObject* object) :
    m_object(object)
{
    if (object && !PyDict_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataDictionary::~PythonDataDictionary ()
{
}

uint32_t
PythonDataDictionary::GetSize()
{
    if (m_object)
        return PyDict_Size(GetPythonObject());
    return 0;
}

PythonDataObject
PythonDataDictionary::GetItemForKey (const char *key) const
{
    if (key && key[0])
    {
        PythonDataString python_key(key);
        return GetItemForKey(python_key);
    }
    return NULL;
}


PythonDataObject
PythonDataDictionary::GetItemForKey (const PythonDataString &key) const
{
    if (m_object && key)
        return PythonDataObject(PyDict_GetItem(GetPythonObject(), key.GetPythonObject()));
    return PythonDataObject();
}


const char *
PythonDataDictionary::GetItemForKeyAsString (const PythonDataString &key, const char *fail_value) const
{
    if (m_object && key)
    {
        PyObject *object = PyDict_GetItem(GetPythonObject(), key.GetPythonObject());
        if (object && PyString_Check(object))
            return PyString_AsString(object);
    }
    return fail_value;
}

int64_t
PythonDataDictionary::GetItemForKeyAsInteger (const PythonDataString &key, int64_t fail_value) const
{
    if (m_object && key)
    {
        PyObject *object = PyDict_GetItem(GetPythonObject(), key.GetPythonObject());
        if (object && PyInt_Check(object))
            return PyInt_AsLong(object);
    }
    return fail_value;
}

PythonDataArray
PythonDataDictionary::GetKeys () const
{
    if (m_object)
        return PythonDataArray(PyDict_Keys(GetPythonObject()));
    return PythonDataArray();
}

PythonDataString
PythonDataDictionary::GetKeyAtPosition (uint32_t pos) const
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (m_object)
    {
        while (PyDict_Next(GetPythonObject(), &pos_iter, &key, &value))
        {
            if (pos-- == 0)
                return PythonDataString(key);
        }
    }
    return PythonDataString();
}

PythonDataObject
PythonDataDictionary::GetValueAtPosition (uint32_t pos) const
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (!m_object)
        return NULL;
    
    while (PyDict_Next(GetPythonObject(), &pos_iter, &key, &value)) {
        if (pos-- == 0)
            return PythonDataObject(value);
    }
    return PythonDataObject();
}

void
PythonDataDictionary::SetItemForKey (const PythonDataString &key, const PythonDataObject &value)
{
    if (m_object && key && value)
        PyDict_SetItem(GetPythonObject(), key.GetPythonObject(), value.GetPythonObject());
}

#endif
