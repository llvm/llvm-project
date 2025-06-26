#include <cstdio>
#include <string>
#include <unordered_map>

using StringMapT = std::unordered_map<std::string, std::string>;
using StringMapTRef = const StringMapT &;

static void check_references(const StringMapT &ref1, StringMapT &ref2,
                             StringMapTRef ref3, StringMapTRef &ref4,
                             const StringMapT &&ref5, StringMapT &&ref6,
                             const StringMapT *const &ref7) {
  std::printf("Break here");
}

int main() {
  StringMapT string_map;
  {
    auto empty_iter = string_map.begin();
    auto const_empty_iter = string_map.cbegin();
    std::printf("Break here");
  }
  string_map["Foo"] = "Bar";
  string_map["Baz"] = "Qux";

  {
    auto foo = string_map.find("Foo");
    auto invalid = string_map.find("Invalid");

    StringMapT::const_iterator const_baz = string_map.find("Baz");
    auto bucket_it = string_map.begin(string_map.bucket("Baz"));
    auto const_bucket_it = string_map.cbegin(string_map.bucket("Baz"));

    std::printf("Break here");

    StringMapT tmp{{"Hello", "World"}};
    check_references(tmp, tmp, tmp, tmp, StringMapT{tmp}, StringMapT{tmp},
                     &tmp);
  }

  return 0;
}
