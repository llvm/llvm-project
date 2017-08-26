#include "gtest/gtest.h"

#include "lldb/Utility/JSON.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;

TEST(JSONTest, Dictionary) {
  JSONObject o;
  o.SetObject("key", std::make_shared<JSONString>("value"));

  StreamString stream;
  o.Write(stream);

  ASSERT_EQ(stream.GetString(), R"({"key":"value"})");
}

TEST(JSONTest, Newlines) {
  JSONObject o;
  o.SetObject("key", std::make_shared<JSONString>("hello\nworld"));

  StreamString stream;
  o.Write(stream);

  ASSERT_EQ(stream.GetString(), R"({"key":"hello\nworld"})");
}
