//===-- ReproducerInstrumentationTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>

#include "lldb/Utility/ReproducerInstrumentation.h"

using namespace lldb_private;
using namespace lldb_private::repro;

struct Foo {
  int m = 1;
};
struct Bar {
  double m = 2;
};

bool operator==(const Foo &LHS, const Foo &RHS) { return LHS.m == RHS.m; }
bool operator==(const Bar &LHS, const Bar &RHS) { return LHS.m == RHS.m; }

struct Pod {
  bool a = true;
  bool b = false;
  char c = 'a';
  float d = 1.1f;
  int e = 2;
  long long f = 3;
  long g = 4;
  short h = 5;
  unsigned char i = 'b';
  unsigned int j = 6;
  unsigned long long k = 7;
  unsigned long l = 8;
  unsigned short m = 9;

  Pod() {}
};

class TestingRegistry : public Registry {
public:
  TestingRegistry();
};

static llvm::Optional<TestingRegistry> g_registry;
static llvm::Optional<Serializer> g_serializer;
static llvm::Optional<Deserializer> g_deserializer;

class TestInstrumentationData : public InstrumentationData {
public:
  TestInstrumentationData() : InstrumentationData() {}
  TestInstrumentationData(Serializer &serializer, Registry &registry)
      : InstrumentationData(serializer, registry) {}
  TestInstrumentationData(Deserializer &deserializer, Registry &registry)
      : InstrumentationData(deserializer, registry) {}
};

inline TestInstrumentationData GetTestInstrumentationData() {
  assert(!(g_serializer && g_deserializer));
  if (g_serializer)
    return TestInstrumentationData(*g_serializer, *g_registry);
  if (g_deserializer)
    return TestInstrumentationData(*g_deserializer, *g_registry);
  return TestInstrumentationData();
}

class TestInstrumentationDataRAII {
public:
  TestInstrumentationDataRAII(llvm::raw_string_ostream &os) {
    g_registry.emplace();
    g_serializer.emplace(os);
    g_deserializer.reset();
  }

  TestInstrumentationDataRAII(llvm::StringRef buffer) {
    g_registry.emplace();
    g_serializer.reset();
    g_deserializer.emplace(buffer);
  }

  ~TestInstrumentationDataRAII() { Reset(); }

  void Reset() {
    g_registry.reset();
    g_serializer.reset();
    g_deserializer.reset();
  }

  static std::unique_ptr<TestInstrumentationDataRAII>
  GetRecordingData(llvm::raw_string_ostream &os) {
    return std::make_unique<TestInstrumentationDataRAII>(os);
  }

  static std::unique_ptr<TestInstrumentationDataRAII>
  GetReplayData(llvm::StringRef buffer) {
    return std::make_unique<TestInstrumentationDataRAII>(buffer);
  }
};

#define LLDB_GET_INSTRUMENTATION_DATA() GetTestInstrumentationData()

enum class Class {
  Foo,
  Bar,
};

class Instrumented {
public:
  virtual ~Instrumented() = default;
  virtual void Validate() = 0;
  virtual bool IsA(Class c) = 0;
};

class InstrumentedFoo : public Instrumented {
public:
  InstrumentedFoo() = default;
  /// Instrumented methods.
  /// {
  InstrumentedFoo(int i);
  InstrumentedFoo(const InstrumentedFoo &foo);
  InstrumentedFoo &operator=(const InstrumentedFoo &foo);
  void A(int a);
  int GetA();
  void B(int &b) const;
  int &GetB();
  int C(float *c);
  float GetC();
  int D(const char *d) const;
  size_t GetD(char *buffer, size_t length);
  static void E(double e);
  double GetE();
  static int F();
  bool GetF();
  void Validate() override;
  //// }
  virtual bool IsA(Class c) override { return c == Class::Foo; }

private:
  int m_a = 0;
  mutable int m_b = 0;
  float m_c = 0;
  mutable std::string m_d = {};
  static double g_e;
  static bool g_f;
  mutable int m_called = 0;
};

class InstrumentedBar : public Instrumented {
public:
  /// Instrumented methods.
  /// {
  InstrumentedBar();
  InstrumentedFoo GetInstrumentedFoo();
  InstrumentedFoo &GetInstrumentedFooRef();
  InstrumentedFoo *GetInstrumentedFooPtr();
  void SetInstrumentedFoo(InstrumentedFoo *foo);
  void SetInstrumentedFoo(InstrumentedFoo &foo);
  void Validate() override;
  /// }
  virtual bool IsA(Class c) override { return c == Class::Bar; }

private:
  bool m_get_instrumend_foo_called = false;
  InstrumentedFoo *m_foo_set_by_ptr = nullptr;
  InstrumentedFoo *m_foo_set_by_ref = nullptr;
};

double InstrumentedFoo::g_e = 0;
bool InstrumentedFoo::g_f = false;

struct Validator {
  enum Validation { valid, invalid };
  Validator(Class clazz, Validation validation)
      : clazz(clazz), validation(validation) {}
  Class clazz;
  Validation validation;
};

void ValidateObjects(std::vector<void *> objects,
                     std::vector<Validator> validators) {
  ASSERT_EQ(validators.size(), objects.size());
  for (size_t i = 0; i < validators.size(); ++i) {
    Validator &validator = validators[i];
    Instrumented *instrumented = static_cast<Instrumented *>(objects[i]);
    EXPECT_TRUE(instrumented->IsA(validator.clazz));
    switch (validator.validation) {
    case Validator::valid:
      instrumented->Validate();
      break;
    case Validator::invalid:
      break;
    }
  }
}

InstrumentedFoo::InstrumentedFoo(int i) {
  LLDB_RECORD_CONSTRUCTOR(InstrumentedFoo, (int), i);
}

InstrumentedFoo::InstrumentedFoo(const InstrumentedFoo &foo) {
  LLDB_RECORD_CONSTRUCTOR(InstrumentedFoo, (const InstrumentedFoo &), foo);
}

InstrumentedFoo &InstrumentedFoo::operator=(const InstrumentedFoo &foo) {
  LLDB_RECORD_METHOD(InstrumentedFoo &,
                     InstrumentedFoo, operator=,(const InstrumentedFoo &), foo);
  return *this;
}

void InstrumentedFoo::A(int a) {
  LLDB_RECORD_METHOD(void, InstrumentedFoo, A, (int), a);
  B(a);
  m_a = a;
}

int InstrumentedFoo::GetA() {
  LLDB_RECORD_METHOD_NO_ARGS(int, InstrumentedFoo, GetA);

  return m_a;
}

void InstrumentedFoo::B(int &b) const {
  LLDB_RECORD_METHOD_CONST(void, InstrumentedFoo, B, (int &), b);
  m_called++;
  m_b = b;
}

int &InstrumentedFoo::GetB() {
  LLDB_RECORD_METHOD_NO_ARGS(int &, InstrumentedFoo, GetB);

  return m_b;
}

int InstrumentedFoo::C(float *c) {
  LLDB_RECORD_METHOD(int, InstrumentedFoo, C, (float *), c);
  m_c = *c;
  return 1;
}

float InstrumentedFoo::GetC() {
  LLDB_RECORD_METHOD_NO_ARGS(float, InstrumentedFoo, GetC);

  return m_c;
}

int InstrumentedFoo::D(const char *d) const {
  LLDB_RECORD_METHOD_CONST(int, InstrumentedFoo, D, (const char *), d);
  m_d = std::string(d);
  return 2;
}

size_t InstrumentedFoo::GetD(char *buffer, size_t length) {
  LLDB_RECORD_CHAR_PTR_METHOD(size_t, InstrumentedFoo, GetD, (char *, size_t),
                              buffer, "", length);
  ::snprintf(buffer, length, "%s", m_d.c_str());
  return m_d.size();
}

void InstrumentedFoo::E(double e) {
  LLDB_RECORD_STATIC_METHOD(void, InstrumentedFoo, E, (double), e);
  g_e = e;
}

double InstrumentedFoo::GetE() {
  LLDB_RECORD_METHOD_NO_ARGS(double, InstrumentedFoo, GetE);

  return g_e;
}

int InstrumentedFoo::F() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(int, InstrumentedFoo, F);
  g_f = true;
  return 3;
}

bool InstrumentedFoo::GetF() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, InstrumentedFoo, GetF);

  return g_f;
}

void InstrumentedFoo::Validate() {
  LLDB_RECORD_METHOD_NO_ARGS(void, InstrumentedFoo, Validate);
  EXPECT_EQ(m_a, 100);
  EXPECT_EQ(m_b, 200);
  EXPECT_NEAR(m_c, 300.3, 0.01);
  EXPECT_EQ(m_d, "bar");
  EXPECT_NEAR(g_e, 400.4, 0.01);
  EXPECT_EQ(g_f, true);
  EXPECT_EQ(2, m_called);
}

InstrumentedBar::InstrumentedBar() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(InstrumentedBar);
}

InstrumentedFoo InstrumentedBar::GetInstrumentedFoo() {
  LLDB_RECORD_METHOD_NO_ARGS(InstrumentedFoo, InstrumentedBar,
                             GetInstrumentedFoo);
  m_get_instrumend_foo_called = true;
  return LLDB_RECORD_RESULT(InstrumentedFoo(0));
}

InstrumentedFoo &InstrumentedBar::GetInstrumentedFooRef() {
  LLDB_RECORD_METHOD_NO_ARGS(InstrumentedFoo &, InstrumentedBar,
                             GetInstrumentedFooRef);
  InstrumentedFoo *foo = new InstrumentedFoo(0);
  m_get_instrumend_foo_called = true;
  return LLDB_RECORD_RESULT(*foo);
}

InstrumentedFoo *InstrumentedBar::GetInstrumentedFooPtr() {
  LLDB_RECORD_METHOD_NO_ARGS(InstrumentedFoo *, InstrumentedBar,
                             GetInstrumentedFooPtr);
  InstrumentedFoo *foo = new InstrumentedFoo(0);
  m_get_instrumend_foo_called = true;
  return LLDB_RECORD_RESULT(foo);
}

void InstrumentedBar::SetInstrumentedFoo(InstrumentedFoo *foo) {
  LLDB_RECORD_METHOD(void, InstrumentedBar, SetInstrumentedFoo,
                     (InstrumentedFoo *), foo);
  m_foo_set_by_ptr = foo;
}

void InstrumentedBar::SetInstrumentedFoo(InstrumentedFoo &foo) {
  LLDB_RECORD_METHOD(void, InstrumentedBar, SetInstrumentedFoo,
                     (InstrumentedFoo &), foo);
  m_foo_set_by_ref = &foo;
}

void InstrumentedBar::Validate() {
  LLDB_RECORD_METHOD_NO_ARGS(void, InstrumentedBar, Validate);

  EXPECT_TRUE(m_get_instrumend_foo_called);
  EXPECT_NE(m_foo_set_by_ptr, nullptr);
  EXPECT_EQ(m_foo_set_by_ptr, m_foo_set_by_ref);
}

TestingRegistry::TestingRegistry() {
  Registry &R = *this;

  LLDB_REGISTER_CONSTRUCTOR(InstrumentedFoo, (int i));
  LLDB_REGISTER_CONSTRUCTOR(InstrumentedFoo, (const InstrumentedFoo &));
  LLDB_REGISTER_METHOD(InstrumentedFoo &,
                       InstrumentedFoo, operator=,(const InstrumentedFoo &));
  LLDB_REGISTER_METHOD(void, InstrumentedFoo, A, (int));
  LLDB_REGISTER_METHOD_CONST(void, InstrumentedFoo, B, (int &));
  LLDB_REGISTER_METHOD(int, InstrumentedFoo, C, (float *));
  LLDB_REGISTER_METHOD_CONST(int, InstrumentedFoo, D, (const char *));
  LLDB_REGISTER_STATIC_METHOD(void, InstrumentedFoo, E, (double));
  LLDB_REGISTER_STATIC_METHOD(int, InstrumentedFoo, F, ());
  LLDB_REGISTER_METHOD(void, InstrumentedFoo, Validate, ());

  LLDB_REGISTER_CONSTRUCTOR(InstrumentedBar, ());
  LLDB_REGISTER_METHOD(InstrumentedFoo, InstrumentedBar, GetInstrumentedFoo,
                       ());
  LLDB_REGISTER_METHOD(InstrumentedFoo &, InstrumentedBar,
                       GetInstrumentedFooRef, ());
  LLDB_REGISTER_METHOD(InstrumentedFoo *, InstrumentedBar,
                       GetInstrumentedFooPtr, ());
  LLDB_REGISTER_METHOD(void, InstrumentedBar, SetInstrumentedFoo,
                       (InstrumentedFoo *));
  LLDB_REGISTER_METHOD(void, InstrumentedBar, SetInstrumentedFoo,
                       (InstrumentedFoo &));
  LLDB_REGISTER_METHOD(void, InstrumentedBar, Validate, ());
  LLDB_REGISTER_METHOD(int, InstrumentedFoo, GetA, ());
  LLDB_REGISTER_METHOD(int &, InstrumentedFoo, GetB, ());
  LLDB_REGISTER_METHOD(float, InstrumentedFoo, GetC, ());
  LLDB_REGISTER_METHOD(size_t, InstrumentedFoo, GetD, (char *, size_t));
  LLDB_REGISTER_METHOD(double, InstrumentedFoo, GetE, ());
  LLDB_REGISTER_METHOD(bool, InstrumentedFoo, GetF, ());
}

static const Pod p;

TEST(IndexToObjectTest, ObjectForIndex) {
  IndexToObject index_to_object;
  Foo foo;
  Bar bar;

  EXPECT_EQ(nullptr, index_to_object.GetObjectForIndex<Foo>(1));
  EXPECT_EQ(nullptr, index_to_object.GetObjectForIndex<Bar>(2));

  index_to_object.AddObjectForIndex<Foo>(1, foo);
  index_to_object.AddObjectForIndex<Bar>(2, &bar);

  EXPECT_EQ(&foo, index_to_object.GetObjectForIndex<Foo>(1));
  EXPECT_EQ(&bar, index_to_object.GetObjectForIndex<Bar>(2));
}

TEST(DeserializerTest, HasData) {
  {
    Deserializer deserializer("");
    EXPECT_FALSE(deserializer.HasData(1));
  }

  {
    Deserializer deserializer("a");
    EXPECT_TRUE(deserializer.HasData(1));
    EXPECT_FALSE(deserializer.HasData(2));
  }
}

TEST(SerializationRountripTest, SerializeDeserializePod) {
  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(p.a, p.b, p.c, p.d, p.e, p.f, p.g, p.h, p.i, p.j, p.k,
                          p.l, p.m);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  EXPECT_EQ(p.a, deserializer.Deserialize<bool>());
  EXPECT_EQ(p.b, deserializer.Deserialize<bool>());
  EXPECT_EQ(p.c, deserializer.Deserialize<char>());
  EXPECT_EQ(p.d, deserializer.Deserialize<float>());
  EXPECT_EQ(p.e, deserializer.Deserialize<int>());
  EXPECT_EQ(p.f, deserializer.Deserialize<long long>());
  EXPECT_EQ(p.g, deserializer.Deserialize<long>());
  EXPECT_EQ(p.h, deserializer.Deserialize<short>());
  EXPECT_EQ(p.i, deserializer.Deserialize<unsigned char>());
  EXPECT_EQ(p.j, deserializer.Deserialize<unsigned int>());
  EXPECT_EQ(p.k, deserializer.Deserialize<unsigned long long>());
  EXPECT_EQ(p.l, deserializer.Deserialize<unsigned long>());
  EXPECT_EQ(p.m, deserializer.Deserialize<unsigned short>());
}

TEST(SerializationRountripTest, SerializeDeserializePodPointers) {
  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(&p.a, &p.b, &p.c, &p.d, &p.e, &p.f, &p.g, &p.h, &p.i,
                          &p.j, &p.k, &p.l, &p.m);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  EXPECT_EQ(p.a, *deserializer.Deserialize<bool *>());
  EXPECT_EQ(p.b, *deserializer.Deserialize<bool *>());
  EXPECT_EQ(p.c, *deserializer.Deserialize<char *>());
  EXPECT_EQ(p.d, *deserializer.Deserialize<float *>());
  EXPECT_EQ(p.e, *deserializer.Deserialize<int *>());
  EXPECT_EQ(p.f, *deserializer.Deserialize<long long *>());
  EXPECT_EQ(p.g, *deserializer.Deserialize<long *>());
  EXPECT_EQ(p.h, *deserializer.Deserialize<short *>());
  EXPECT_EQ(p.i, *deserializer.Deserialize<unsigned char *>());
  EXPECT_EQ(p.j, *deserializer.Deserialize<unsigned int *>());
  EXPECT_EQ(p.k, *deserializer.Deserialize<unsigned long long *>());
  EXPECT_EQ(p.l, *deserializer.Deserialize<unsigned long *>());
  EXPECT_EQ(p.m, *deserializer.Deserialize<unsigned short *>());
}

TEST(SerializationRountripTest, SerializeDeserializePodReferences) {
  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(p.a, p.b, p.c, p.d, p.e, p.f, p.g, p.h, p.i, p.j, p.k,
                          p.l, p.m);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  EXPECT_EQ(p.a, deserializer.Deserialize<bool &>());
  EXPECT_EQ(p.b, deserializer.Deserialize<bool &>());
  EXPECT_EQ(p.c, deserializer.Deserialize<char &>());
  EXPECT_EQ(p.d, deserializer.Deserialize<float &>());
  EXPECT_EQ(p.e, deserializer.Deserialize<int &>());
  EXPECT_EQ(p.f, deserializer.Deserialize<long long &>());
  EXPECT_EQ(p.g, deserializer.Deserialize<long &>());
  EXPECT_EQ(p.h, deserializer.Deserialize<short &>());
  EXPECT_EQ(p.i, deserializer.Deserialize<unsigned char &>());
  EXPECT_EQ(p.j, deserializer.Deserialize<unsigned int &>());
  EXPECT_EQ(p.k, deserializer.Deserialize<unsigned long long &>());
  EXPECT_EQ(p.l, deserializer.Deserialize<unsigned long &>());
  EXPECT_EQ(p.m, deserializer.Deserialize<unsigned short &>());
}

TEST(SerializationRountripTest, SerializeDeserializeCString) {
  const char *cstr = "string";

  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(cstr);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  EXPECT_STREQ(cstr, deserializer.Deserialize<const char *>());
}

TEST(SerializationRountripTest, SerializeDeserializeCStringNull) {
  const char *cstr = nullptr;

  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(cstr);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  EXPECT_EQ(nullptr, deserializer.Deserialize<const char *>());
}

TEST(SerializationRountripTest, SerializeDeserializeCStringArray) {
  const char *foo = "foo";
  const char *bar = "bar";
  const char *baz = "baz";
  const char *arr[4] = {foo, bar, baz, nullptr};

  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(static_cast<const char **>(arr));

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  const char **deserialized = deserializer.Deserialize<const char **>();
  EXPECT_STREQ("foo", deserialized[0]);
  EXPECT_STREQ("bar", deserialized[1]);
  EXPECT_STREQ("baz", deserialized[2]);
}

TEST(SerializationRountripTest, SerializeDeserializeCStringArrayNullptrElem) {
  const char *arr[1] = {nullptr};

  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(static_cast<const char **>(arr));

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  const char **deserialized = deserializer.Deserialize<const char **>();
  EXPECT_EQ(nullptr, deserialized);
}

TEST(SerializationRountripTest, SerializeDeserializeCStringArrayNullptr) {
  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(static_cast<const char **>(nullptr));

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  const char **deserialized = deserializer.Deserialize<const char **>();
  EXPECT_EQ(nullptr, deserialized);
}

TEST(SerializationRountripTest, SerializeDeserializeObjectPointer) {
  Foo foo;
  Bar bar;

  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(static_cast<unsigned>(1), static_cast<unsigned>(2));
  serializer.SerializeAll(&foo, &bar);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  deserializer.HandleReplayResult(&foo);
  deserializer.HandleReplayResult(&bar);

  EXPECT_EQ(foo, *deserializer.Deserialize<Foo *>());
  EXPECT_EQ(bar, *deserializer.Deserialize<Bar *>());
}

TEST(SerializationRountripTest, SerializeDeserializeObjectReference) {
  Foo foo;
  Bar bar;

  std::string str;
  llvm::raw_string_ostream os(str);

  Serializer serializer(os);
  serializer.SerializeAll(static_cast<unsigned>(1), static_cast<unsigned>(2));
  serializer.SerializeAll(foo, bar);

  llvm::StringRef buffer(os.str());
  Deserializer deserializer(buffer);

  deserializer.HandleReplayResult(&foo);
  deserializer.HandleReplayResult(&bar);

  EXPECT_EQ(foo, deserializer.Deserialize<Foo &>());
  EXPECT_EQ(bar, deserializer.Deserialize<Bar &>());
}

TEST(RecordReplayTest, InstrumentedFoo) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    InstrumentedFoo foo(0);
    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();
  }

  TestingRegistry registry;
  Deserializer deserializer(os.str());
  registry.Replay(deserializer);

  ValidateObjects(deserializer.GetAllObjects(),
                  {{Class::Foo, Validator::valid}});
}

TEST(RecordReplayTest, InstrumentedFooSameThis) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    InstrumentedFoo *foo = new InstrumentedFoo(0);
    foo->A(100);
    foo->B(b);
    foo->C(&c);
    foo->D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo->Validate();
    foo->~InstrumentedFoo();

    InstrumentedFoo *foo2 = new (foo) InstrumentedFoo(0);
    foo2->A(100);
    foo2->B(b);
    foo2->C(&c);
    foo2->D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo2->Validate();
    delete foo2;
  }

  TestingRegistry registry;
  Deserializer deserializer(os.str());
  registry.Replay(deserializer);

  ValidateObjects(deserializer.GetAllObjects(),
                  {{Class::Foo, Validator::valid}});
}

TEST(RecordReplayTest, InstrumentedBar) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    InstrumentedBar bar;
    InstrumentedFoo foo = bar.GetInstrumentedFoo();

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }

  TestingRegistry registry;
  Deserializer deserializer(os.str());
  registry.Replay(deserializer);

  ValidateObjects(
      deserializer.GetAllObjects(),
      {
          {Class::Bar, Validator::valid},   // bar
          {Class::Foo, Validator::invalid}, // bar.GetInstrumentedFoo()
          {Class::Foo, Validator::valid},   // foo
      });
}

TEST(RecordReplayTest, InstrumentedBarRef) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    InstrumentedBar bar;
    InstrumentedFoo &foo = bar.GetInstrumentedFooRef();

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }

  TestingRegistry registry;
  Deserializer deserializer(os.str());
  registry.Replay(deserializer);

  ValidateObjects(
      deserializer.GetAllObjects(),
      {{Class::Bar, Validator::valid}, {Class::Foo, Validator::valid}});
}

TEST(RecordReplayTest, InstrumentedBarPtr) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    InstrumentedBar bar;
    InstrumentedFoo &foo = *(bar.GetInstrumentedFooPtr());

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }

  TestingRegistry registry;
  Deserializer deserializer(os.str());
  registry.Replay(deserializer);

  ValidateObjects(
      deserializer.GetAllObjects(),
      {{Class::Bar, Validator::valid}, {Class::Foo, Validator::valid}});
}

TEST(PassiveReplayTest, InstrumentedFoo) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    InstrumentedFoo foo(0);
    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);
  }

  std::string buffer = os.str();

  {
    auto data = TestInstrumentationDataRAII::GetReplayData(buffer);

    int b = 999;
    float c = 999.9f;
    double e = 999.9;

    InstrumentedFoo foo(9);
    foo.A(999);
    foo.B(b);
    foo.C(&c);
    foo.D("999");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);
  }
}

TEST(PassiveReplayTest, InstrumentedFooInvalid) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    InstrumentedFoo foo(0);
    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);
  }

  std::string buffer = os.str();

  {
    auto data = TestInstrumentationDataRAII::GetReplayData(buffer);

    int b = 999;
    float c = 999.9f;
    double e = 999.9;

    InstrumentedFoo foo(9);
    foo.A(999);
    foo.B(b);
    foo.C(&c);
    foo.D("999");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    // Detect divergence.
    EXPECT_DEATH(foo.GetA(), "");
  }
}

TEST(PassiveReplayTest, InstrumentedBar) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    InstrumentedBar bar;
    InstrumentedFoo foo = bar.GetInstrumentedFoo();

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }

  std::string buffer = os.str();

  {
    auto data = TestInstrumentationDataRAII::GetReplayData(buffer);

    InstrumentedBar bar;
    InstrumentedFoo foo = bar.GetInstrumentedFoo();

    int b = 99;
    float c = 999.9f;
    double e = 999.9;

    foo.A(999);
    foo.B(b);
    foo.C(&c);
    foo.D("999");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }
}

TEST(PassiveReplayTest, InstrumentedBarRef) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    InstrumentedBar bar;
    InstrumentedFoo &foo = bar.GetInstrumentedFooRef();

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }

  std::string buffer = os.str();

  {
    auto data = TestInstrumentationDataRAII::GetReplayData(buffer);

    InstrumentedBar bar;
    InstrumentedFoo &foo = bar.GetInstrumentedFooRef();

    int b = 99;
    float c = 999.9f;
    double e = 999.9;

    foo.A(999);
    foo.B(b);
    foo.C(&c);
    foo.D("999");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }
}

TEST(PassiveReplayTest, InstrumentedBarPtr) {
  std::string str;
  llvm::raw_string_ostream os(str);

  {
    auto data = TestInstrumentationDataRAII::GetRecordingData(os);

    InstrumentedBar bar;
    InstrumentedFoo &foo = *(bar.GetInstrumentedFooPtr());

    int b = 200;
    float c = 300.3f;
    double e = 400.4;

    foo.A(100);
    foo.B(b);
    foo.C(&c);
    foo.D("bar");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }

  std::string buffer = os.str();

  {
    auto data = TestInstrumentationDataRAII::GetReplayData(buffer);

    InstrumentedBar bar;
    InstrumentedFoo &foo = *(bar.GetInstrumentedFooPtr());

    int b = 99;
    float c = 999.9f;
    double e = 999.9;

    foo.A(999);
    foo.B(b);
    foo.C(&c);
    foo.D("999");
    InstrumentedFoo::E(e);
    InstrumentedFoo::F();
    foo.Validate();

    EXPECT_EQ(foo.GetA(), 100);
    EXPECT_EQ(foo.GetB(), 200);
    EXPECT_NEAR(foo.GetC(), 300.3, 0.01);
    char buffer[100];
    foo.GetD(buffer, 100);
    EXPECT_STREQ(buffer, "bar");
    EXPECT_NEAR(foo.GetE(), 400.4, 0.01);
    EXPECT_EQ(foo.GetF(), true);

    bar.SetInstrumentedFoo(foo);
    bar.SetInstrumentedFoo(&foo);
    bar.Validate();
  }
}
