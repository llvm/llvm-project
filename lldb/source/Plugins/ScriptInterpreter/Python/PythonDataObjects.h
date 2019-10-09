//===-- PythonDataObjects.h--------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// !! FIXME FIXME FIXME !!
//
// Python APIs nearly all can return an exception.   They do this
// by returning NULL, or -1, or some such value and setting
// the exception state with PyErr_Set*().   Exceptions must be
// handled before further python API functions are called.   Failure
// to do so will result in asserts on debug builds of python.
// It will also sometimes, but not usually result in crashes of
// release builds.
//
// Nearly all the code in this header does not handle python exceptions
// correctly.  It should all be converted to return Expected<> or
// Error types to capture the exception.
//
// Everything in this file except functions that return Error or
// Expected<> is considered deprecated and should not be
// used in new code.  If you need to use it, fix it first.
//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONDATAOBJECTS_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONDATAOBJECTS_H

#ifndef LLDB_DISABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "lldb/Host/File.h"
#include "lldb/Utility/StructuredData.h"

#include "llvm/ADT/ArrayRef.h"

namespace lldb_private {

class PythonObject;
class PythonBytes;
class PythonString;
class PythonList;
class PythonDictionary;
class PythonInteger;
class PythonException;

class StructuredPythonObject : public StructuredData::Generic {
public:
  StructuredPythonObject() : StructuredData::Generic() {}

  StructuredPythonObject(void *obj) : StructuredData::Generic(obj) {
    Py_XINCREF(GetValue());
  }

  ~StructuredPythonObject() override {
    if (Py_IsInitialized())
      Py_XDECREF(GetValue());
    SetValue(nullptr);
  }

  bool IsValid() const override { return GetValue() && GetValue() != Py_None; }

  void Serialize(llvm::json::OStream &s) const override;

private:
  DISALLOW_COPY_AND_ASSIGN(StructuredPythonObject);
};

enum class PyObjectType {
  Unknown,
  None,
  Boolean,
  Integer,
  Dictionary,
  List,
  String,
  Bytes,
  ByteArray,
  Module,
  Callable,
  Tuple,
  File
};

enum class PyRefType {
  Borrowed, // We are not given ownership of the incoming PyObject.
            // We cannot safely hold it without calling Py_INCREF.
  Owned     // We have ownership of the incoming PyObject.  We should
            // not call Py_INCREF.
};

namespace python {

// Take a reference that you already own, and turn it into
// a PythonObject.
//
// Most python API methods will return a +1 reference
// if they succeed or NULL if and only if
// they set an exception.   Use this to collect such return
// values, after checking for NULL.
//
// If T is not just PythonObject, then obj must be already be
// checked to be of the correct type.
template <typename T> T Take(PyObject *obj) {
  assert(obj);
  assert(!PyErr_Occurred());
  T thing(PyRefType::Owned, obj);
  assert(thing.IsValid());
  return std::move(thing);
}

// Retain a reference you have borrowed, and turn it into
// a PythonObject.
//
// A minority of python APIs return a borrowed reference
// instead of a +1.   They will also return NULL if and only
// if they set an exception.   Use this to collect such return
// values, after checking for NULL.
//
// If T is not just PythonObject, then obj must be already be
// checked to be of the correct type.
template <typename T> T Retain(PyObject *obj) {
  assert(obj);
  assert(!PyErr_Occurred());
  T thing(PyRefType::Borrowed, obj);
  assert(thing.IsValid());
  return std::move(thing);
}

} // namespace python

enum class PyInitialValue { Invalid, Empty };

template <typename T, typename Enable = void> struct PythonFormat;

template <> struct PythonFormat<unsigned long long> {
  static constexpr char format = 'K';
  static auto get(unsigned long long value) { return value; }
};

template <> struct PythonFormat<long long> {
  static constexpr char format = 'L';
  static auto get(long long value) { return value; }
};

template <typename T>
struct PythonFormat<
    T, typename std::enable_if<std::is_base_of<PythonObject, T>::value>::type> {
  static constexpr char format = 'O';
  static auto get(const T &value) { return value.get(); }
};

class PythonObject {
public:
  PythonObject() : m_py_obj(nullptr) {}

  PythonObject(PyRefType type, PyObject *py_obj) : m_py_obj(nullptr) {
    Reset(type, py_obj);
  }

  PythonObject(const PythonObject &rhs) : m_py_obj(nullptr) { Reset(rhs); }

  PythonObject(PythonObject &&rhs) {
    m_py_obj = rhs.m_py_obj;
    rhs.m_py_obj = nullptr;
  }

  virtual ~PythonObject() { Reset(); }

  void Reset() {
    // Avoid calling the virtual method since it's not necessary
    // to actually validate the type of the PyObject if we're
    // just setting to null.
    if (m_py_obj && Py_IsInitialized())
      Py_DECREF(m_py_obj);
    m_py_obj = nullptr;
  }

  void Reset(const PythonObject &rhs) {
    // Avoid calling the virtual method if it's not necessary
    // to actually validate the type of the PyObject.
    if (!rhs.IsValid())
      Reset();
    else
      Reset(PyRefType::Borrowed, rhs.m_py_obj);
  }

  // PythonObject is implicitly convertible to PyObject *, which will call the
  // wrong overload.  We want to explicitly disallow this, since a PyObject
  // *always* owns its reference.  Therefore the overload which takes a
  // PyRefType doesn't make sense, and the copy constructor should be used.
  void Reset(PyRefType type, const PythonObject &ref) = delete;

  // FIXME We shouldn't have virtual anything.  PythonObject should be a
  // strictly pass-by-value type.
  virtual void Reset(PyRefType type, PyObject *py_obj) {
    if (py_obj == m_py_obj)
      return;

    if (Py_IsInitialized())
      Py_XDECREF(m_py_obj);

    m_py_obj = py_obj;

    // If this is a borrowed reference, we need to convert it to
    // an owned reference by incrementing it.  If it is an owned
    // reference (for example the caller allocated it with PyDict_New()
    // then we must *not* increment it.
    if (m_py_obj && Py_IsInitialized() && type == PyRefType::Borrowed)
      Py_XINCREF(m_py_obj);
  }

  void Dump() const {
    if (m_py_obj)
      _PyObject_Dump(m_py_obj);
    else
      puts("NULL");
  }

  void Dump(Stream &strm) const;

  PyObject *get() const { return m_py_obj; }

  PyObject *release() {
    PyObject *result = m_py_obj;
    m_py_obj = nullptr;
    return result;
  }

  PythonObject &operator=(const PythonObject &other) {
    Reset(PyRefType::Borrowed, other.get());
    return *this;
  }

  void Reset(PythonObject &&other) {
    Reset();
    m_py_obj = other.m_py_obj;
    other.m_py_obj = nullptr;
  }

  PythonObject &operator=(PythonObject &&other) {
    Reset(std::move(other));
    return *this;
  }

  PyObjectType GetObjectType() const;

  PythonString Repr() const;

  PythonString Str() const;

  static PythonObject ResolveNameWithDictionary(llvm::StringRef name,
                                                const PythonDictionary &dict);

  template <typename T>
  static T ResolveNameWithDictionary(llvm::StringRef name,
                                     const PythonDictionary &dict) {
    return ResolveNameWithDictionary(name, dict).AsType<T>();
  }

  PythonObject ResolveName(llvm::StringRef name) const;

  template <typename T> T ResolveName(llvm::StringRef name) const {
    return ResolveName(name).AsType<T>();
  }

  bool HasAttribute(llvm::StringRef attribute) const;

  PythonObject GetAttributeValue(llvm::StringRef attribute) const;

  bool IsNone() const { return m_py_obj == Py_None; }

  bool IsValid() const { return m_py_obj != nullptr; }

  bool IsAllocated() const { return IsValid() && !IsNone(); }

  explicit operator bool() const { return IsValid() && !IsNone(); }

  template <typename T> T AsType() const {
    if (!T::Check(m_py_obj))
      return T();
    return T(PyRefType::Borrowed, m_py_obj);
  }

  StructuredData::ObjectSP CreateStructuredObject() const;

protected:
  static llvm::Error nullDeref() {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "A NULL PyObject* was dereferenced");
  }
  static llvm::Error exception(const char *s = nullptr) {
    return llvm::make_error<PythonException>(s);
  }

public:
  template <typename... T>
  llvm::Expected<PythonObject> CallMethod(const char *name,
                                          const T &... t) const {
    const char format[] = {'(', PythonFormat<T>::format..., ')', 0};
#if PY_MAJOR_VERSION < 3
    PyObject *obj = PyObject_CallMethod(m_py_obj, const_cast<char *>(name),
                                        const_cast<char *>(format),
                                        PythonFormat<T>::get(t)...);
#else
    PyObject *obj =
        PyObject_CallMethod(m_py_obj, name, format, PythonFormat<T>::get(t)...);
#endif
    if (!obj)
      return exception();
    return python::Take<PythonObject>(obj);
  }

  llvm::Expected<PythonObject> GetAttribute(const char *name) const {
    if (!m_py_obj)
      return nullDeref();
    PyObject *obj = PyObject_GetAttrString(m_py_obj, name);
    if (!obj)
      return exception();
    return python::Take<PythonObject>(obj);
  }

  llvm::Expected<bool> IsTrue() {
    if (!m_py_obj)
      return nullDeref();
    int r = PyObject_IsTrue(m_py_obj);
    if (r < 0)
      return exception();
    return !!r;
  }

  llvm::Expected<long long> AsLongLong() {
    if (!m_py_obj)
      return nullDeref();
    assert(!PyErr_Occurred());
    long long r = PyLong_AsLongLong(m_py_obj);
    if (PyErr_Occurred())
      return exception();
    return r;
  }

  llvm::Expected<bool> IsInstance(const PythonObject &cls) {
    if (!m_py_obj || !cls.IsValid())
      return nullDeref();
    int r = PyObject_IsInstance(m_py_obj, cls.get());
    if (r < 0)
      return exception();
    return !!r;
  }

protected:
  PyObject *m_py_obj;
};

namespace python {

// This is why C++ needs monads.
template <typename T> llvm::Expected<T> As(llvm::Expected<PythonObject> &&obj) {
  if (!obj)
    return obj.takeError();
  if (!T::Check(obj.get().get()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "type error");
  return T(PyRefType::Borrowed, std::move(obj.get().get()));
}

template <> llvm::Expected<bool> As<bool>(llvm::Expected<PythonObject> &&obj);

template <>
llvm::Expected<long long> As<long long>(llvm::Expected<PythonObject> &&obj);

} // namespace python

class PythonBytes : public PythonObject {
public:
  PythonBytes();
  explicit PythonBytes(llvm::ArrayRef<uint8_t> bytes);
  PythonBytes(const uint8_t *bytes, size_t length);
  PythonBytes(PyRefType type, PyObject *o);

  ~PythonBytes() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  llvm::ArrayRef<uint8_t> GetBytes() const;

  size_t GetSize() const;

  void SetBytes(llvm::ArrayRef<uint8_t> stringbytes);

  StructuredData::StringSP CreateStructuredString() const;
};

class PythonByteArray : public PythonObject {
public:
  PythonByteArray();
  explicit PythonByteArray(llvm::ArrayRef<uint8_t> bytes);
  PythonByteArray(const uint8_t *bytes, size_t length);
  PythonByteArray(PyRefType type, PyObject *o);
  PythonByteArray(const PythonBytes &object);

  ~PythonByteArray() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  llvm::ArrayRef<uint8_t> GetBytes() const;

  size_t GetSize() const;

  void SetBytes(llvm::ArrayRef<uint8_t> stringbytes);

  StructuredData::StringSP CreateStructuredString() const;
};

class PythonString : public PythonObject {
public:
  static llvm::Expected<PythonString> FromUTF8(llvm::StringRef string);

  PythonString();
  explicit PythonString(llvm::StringRef string); // safe, null on error
  PythonString(PyRefType type, PyObject *o);

  ~PythonString() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  llvm::StringRef GetString() const; // safe, empty string on error

  llvm::Expected<llvm::StringRef> AsUTF8() const;

  size_t GetSize() const;

  void SetString(llvm::StringRef string); // safe, null on error

  StructuredData::StringSP CreateStructuredString() const;
};

class PythonInteger : public PythonObject {
public:
  PythonInteger();
  explicit PythonInteger(int64_t value);
  PythonInteger(PyRefType type, PyObject *o);

  ~PythonInteger() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  int64_t GetInteger() const;

  void SetInteger(int64_t value);

  StructuredData::IntegerSP CreateStructuredInteger() const;
};

class PythonBoolean : public PythonObject {
public:
  PythonBoolean() = default;
  explicit PythonBoolean(bool value);
  PythonBoolean(PyRefType type, PyObject *o);

  ~PythonBoolean() override = default;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  bool GetValue() const;

  void SetValue(bool value);

  StructuredData::BooleanSP CreateStructuredBoolean() const;
};

class PythonList : public PythonObject {
public:
  PythonList() {}
  explicit PythonList(PyInitialValue value);
  explicit PythonList(int list_size);
  PythonList(PyRefType type, PyObject *o);

  ~PythonList() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  uint32_t GetSize() const;

  PythonObject GetItemAtIndex(uint32_t index) const;

  void SetItemAtIndex(uint32_t index, const PythonObject &object);

  void AppendItem(const PythonObject &object);

  StructuredData::ArraySP CreateStructuredArray() const;
};

class PythonTuple : public PythonObject {
public:
  PythonTuple() {}
  explicit PythonTuple(PyInitialValue value);
  explicit PythonTuple(int tuple_size);
  PythonTuple(PyRefType type, PyObject *o);
  PythonTuple(std::initializer_list<PythonObject> objects);
  PythonTuple(std::initializer_list<PyObject *> objects);

  ~PythonTuple() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  uint32_t GetSize() const;

  PythonObject GetItemAtIndex(uint32_t index) const;

  void SetItemAtIndex(uint32_t index, const PythonObject &object);

  StructuredData::ArraySP CreateStructuredArray() const;
};

class PythonDictionary : public PythonObject {
public:
  PythonDictionary() {}
  explicit PythonDictionary(PyInitialValue value);
  PythonDictionary(PyRefType type, PyObject *o);

  ~PythonDictionary() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  uint32_t GetSize() const;

  PythonList GetKeys() const;

  PythonObject GetItemForKey(const PythonObject &key) const;
  void SetItemForKey(const PythonObject &key, const PythonObject &value);

  StructuredData::DictionarySP CreateStructuredDictionary() const;
};

class PythonModule : public PythonObject {
public:
  PythonModule();
  PythonModule(PyRefType type, PyObject *o);

  ~PythonModule() override;

  static bool Check(PyObject *py_obj);

  static PythonModule BuiltinsModule();

  static PythonModule MainModule();

  static PythonModule AddModule(llvm::StringRef module);

  // safe, returns invalid on error;
  static PythonModule ImportModule(llvm::StringRef name) {
    std::string s = name;
    auto mod = Import(s.c_str());
    if (!mod) {
      llvm::consumeError(mod.takeError());
      return PythonModule();
    }
    return std::move(mod.get());
  }

  static llvm::Expected<PythonModule> Import(const char *name);

  llvm::Expected<PythonObject> Get(const char *name);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  PythonDictionary GetDictionary() const;
};

class PythonCallable : public PythonObject {
public:
  struct ArgInfo {
    size_t count;
    bool is_bound_method : 1;
    bool has_varargs : 1;
    bool has_kwargs : 1;
  };

  PythonCallable();
  PythonCallable(PyRefType type, PyObject *o);

  ~PythonCallable() override;

  static bool Check(PyObject *py_obj);

  // Bring in the no-argument base class version
  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;

  ArgInfo GetNumArguments() const;
  
  // If the callable is a Py_Class, then find the number of arguments
  // of the __init__ method.
  ArgInfo GetNumInitArguments() const;

  PythonObject operator()();

  PythonObject operator()(std::initializer_list<PyObject *> args);

  PythonObject operator()(std::initializer_list<PythonObject> args);

  template <typename Arg, typename... Args>
  PythonObject operator()(const Arg &arg, Args... args) {
    return operator()({arg, args...});
  }
};

class PythonFile : public PythonObject {
public:
  PythonFile();
  PythonFile(File &file, const char *mode);
  PythonFile(PyRefType type, PyObject *o);

  ~PythonFile() override;

  static bool Check(PyObject *py_obj);

  using PythonObject::Reset;

  void Reset(PyRefType type, PyObject *py_obj) override;
  void Reset(File &file, const char *mode);

  lldb::FileUP GetUnderlyingFile() const;

  llvm::Expected<lldb::FileSP> ConvertToFile(bool borrowed = false);
  llvm::Expected<lldb::FileSP>
  ConvertToFileForcingUseOfScriptingIOMethods(bool borrowed = false);
};

class PythonException : public llvm::ErrorInfo<PythonException> {
private:
  PyObject *m_exception_type, *m_exception, *m_traceback;
  PyObject *m_repr_bytes;

public:
  static char ID;
  const char *toCString() const;
  PythonException(const char *caller = nullptr);
  void Restore();
  ~PythonException();
  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;
};

// This extracts the underlying T out of an Expected<T> and returns it.
// If the Expected is an Error instead of a T, that error will be converted
// into a python exception, and this will return a default-constructed T.
//
// This is appropriate for use right at the boundary of python calling into
// C++, such as in a SWIG typemap.   In such a context you should simply
// check if the returned T is valid, and if it is, return a NULL back
// to python.   This will result in the Error being raised as an exception
// from python code's point of view.
//
// For example:
// ```
// Expected<Foo *> efoop = some_cpp_function();
// Foo *foop = unwrapOrSetPythonException(efoop);
// if (!foop)
//    return NULL;
// do_something(*foop);
//
// If the Error returned was itself created because a python exception was
// raised when C++ code called into python, then the original exception
// will be restored.   Otherwise a simple string exception will be raised.
template <typename T> T unwrapOrSetPythonException(llvm::Expected<T> expected) {
  if (expected)
    return expected.get();
  llvm::handleAllErrors(
      expected.takeError(), [](PythonException &E) { E.Restore(); },
      [](const llvm::ErrorInfoBase &E) {
        PyErr_SetString(PyExc_Exception, E.message().c_str());
      });
  return T();
}

} // namespace lldb_private

#endif

#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_PYTHONDATAOBJECTS_H
