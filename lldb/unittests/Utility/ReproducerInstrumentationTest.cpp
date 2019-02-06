//===-- ReproducerInstrumentationTest.cpp -----------------------*- C++ -*-===//
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
  float d = 1.1;
  int e = 2;
  long long f = 3;
  long g = 4;
  short h = 5;
  unsigned char i = 'b';
  unsigned int j = 6;
  unsigned long long k = 7;
  unsigned long l = 8;
  unsigned short m = 9;
};

class TestingRegistry : public Registry {
public:
  TestingRegistry();
};

static llvm::Optional<Serializer> g_serializer;
static llvm::Optional<TestingRegistry> g_registry;

#define LLDB_GET_INSTRUMENTATION_DATA()                                        \
  InstrumentationData(*g_serializer, *g_registry)

class InstrumentedFoo {
public:
  InstrumentedFoo() = default;
  /// Instrumented methods.
  /// {
  InstrumentedFoo(int i);
  InstrumentedFoo(const InstrumentedFoo &foo);
  InstrumentedFoo &operator=(const InstrumentedFoo &foo);
  void A(int a);
  void B(int &b) const;
  int C(float *c);
  int D(const char *d) const;
  static void E(double e);
  static int F();
  void Validate();
  //// }

private:
  int m_a = 0;
  mutable int m_b = 0;
  float m_c = 0;
  mutable std::string m_d = {};
  static double g_e;
  static bool g_f;
  mutable int m_called = 0;
};

class InstrumentedBar {
public:
  /// Instrumented methods.
  /// {
  InstrumentedBar();
  InstrumentedFoo GetInstrumentedFoo();
  void SetInstrumentedFoo(InstrumentedFoo *foo);
  void SetInstrumentedFoo(InstrumentedFoo &foo);
  void Validate();
  /// }

private:
  bool m_get_instrumend_foo_called = false;
  InstrumentedFoo *m_foo_set_by_ptr = nullptr;
  InstrumentedFoo *m_foo_set_by_ref = nullptr;
};

double InstrumentedFoo::g_e = 0;
bool InstrumentedFoo::g_f = false;

static std::vector<InstrumentedFoo *> g_foos;
static std::vector<InstrumentedBar *> g_bars;

void ClearObjects() {
  g_foos.clear();
  g_bars.clear();
}

void ValidateObjects(size_t expected_foos, size_t expected_bars) {
  EXPECT_EQ(expected_foos, g_foos.size());
  EXPECT_EQ(expected_bars, g_bars.size());

  for (auto *foo : g_foos) {
    foo->Validate();
  }

  for (auto *bar : g_bars) {
    bar->Validate();
  }
}

InstrumentedFoo::InstrumentedFoo(int i) {
  LLDB_RECORD_CONSTRUCTOR(InstrumentedFoo, (int), i);
  g_foos.push_back(this);
}

InstrumentedFoo::InstrumentedFoo(const InstrumentedFoo &foo) {
  LLDB_RECORD_CONSTRUCTOR(InstrumentedFoo, (const InstrumentedFoo &), foo);
  g_foos.erase(std::remove(g_foos.begin(), g_foos.end(), &foo));
  g_foos.push_back(this);
}

InstrumentedFoo &InstrumentedFoo::operator=(const InstrumentedFoo &foo) {
  LLDB_RECORD_METHOD(InstrumentedFoo &,
                     InstrumentedFoo, operator=,(const InstrumentedFoo &), foo);
  g_foos.erase(std::remove(g_foos.begin(), g_foos.end(), &foo));
  g_foos.push_back(this);
  return *this;
}

void InstrumentedFoo::A(int a) {
  LLDB_RECORD_METHOD(void, InstrumentedFoo, A, (int), a);
  B(a);
  m_a = a;
}

void InstrumentedFoo::B(int &b) const {
  LLDB_RECORD_METHOD_CONST(void, InstrumentedFoo, B, (int &), b);
  m_called++;
  m_b = b;
}

int InstrumentedFoo::C(float *c) {
  LLDB_RECORD_METHOD(int, InstrumentedFoo, C, (float *), c);
  m_c = *c;
  return 1;
}

int InstrumentedFoo::D(const char *d) const {
  LLDB_RECORD_METHOD_CONST(int, InstrumentedFoo, D, (const char *), d);
  m_d = std::string(d);
  return 2;
}

void InstrumentedFoo::E(double e) {
  LLDB_RECORD_STATIC_METHOD(void, InstrumentedFoo, E, (double), e);
  g_e = e;
}

int InstrumentedFoo::F() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(int, InstrumentedFoo, F);
  g_f = true;
  return 3;
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
  g_bars.push_back(this);
}

InstrumentedFoo InstrumentedBar::GetInstrumentedFoo() {
  LLDB_RECORD_METHOD_NO_ARGS(InstrumentedFoo, InstrumentedBar,
                             GetInstrumentedFoo);
  m_get_instrumend_foo_called = true;
  return LLDB_RECORD_RESULT(InstrumentedFoo(0));
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
  LLDB_REGISTER_METHOD(void, InstrumentedBar, SetInstrumentedFoo,
                       (InstrumentedFoo *));
  LLDB_REGISTER_METHOD(void, InstrumentedBar, SetInstrumentedFoo,
                       (InstrumentedFoo &));
  LLDB_REGISTER_METHOD(void, InstrumentedBar, Validate, ());
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
  g_registry.emplace();
  g_serializer.emplace(os);

  {
    int b = 200;
    float c = 300.3;
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

  ClearObjects();

  TestingRegistry registry;
  registry.Replay(os.str());

  ValidateObjects(1, 0);
}

TEST(RecordReplayTest, InstrumentedFooSameThis) {
  std::string str;
  llvm::raw_string_ostream os(str);
  g_registry.emplace();
  g_serializer.emplace(os);

  int b = 200;
  float c = 300.3;
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

  ClearObjects();

  TestingRegistry registry;
  registry.Replay(os.str());

  ValidateObjects(2, 0);
}

TEST(RecordReplayTest, InstrumentedBar) {
  std::string str;
  llvm::raw_string_ostream os(str);
  g_registry.emplace();
  g_serializer.emplace(os);

  {
    InstrumentedBar bar;
    InstrumentedFoo foo = bar.GetInstrumentedFoo();

    int b = 200;
    float c = 300.3;
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

  ClearObjects();

  TestingRegistry registry;
  registry.Replay(os.str());

  ValidateObjects(1, 1);
}
