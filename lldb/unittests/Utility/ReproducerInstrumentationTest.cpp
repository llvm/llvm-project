//===-- ReproducerInstrumentationTest.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Utility/ReproducerInstrumentation.h"

using namespace lldb_private;
using namespace lldb_private::repro;

namespace {
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
} // namespace

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
