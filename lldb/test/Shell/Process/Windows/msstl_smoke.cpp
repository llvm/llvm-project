// This smoke test ensures that LLDB doesn't crash when formatting types from MSVC's STL.
// FIXME: LLDB currently has no built-in formatters for MSVC's STL (#24834)

// REQUIRES: target-windows
// RUN: %build --compiler=clang-cl -o %t.exe --std c++20 -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -o "b main" -o "run" -o "fr v" -o c | FileCheck %s

#include <bitset>
#include <coroutine>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

int main() {
  std::shared_ptr<int> foo;
  std::weak_ptr<int> weak = foo;
  std::unique_ptr<int> unique(new int(42));
  std::optional<int> opt;
  std::string str = "str";
  std::string longStr = "string that is long enough such that no SSO can happen";
  std::wstring wStr = L"wstr";
  std::wstring longWStr = L"string that is long enough such that no SSO can happen";
  std::tuple<int, bool, float> tuple{1, false, 4.2};
  std::coroutine_handle<> coroHandle;
  std::bitset<16> bitset(123);

  std::map<int, int> map{{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}};
  auto mapIt = map.find(3);
  auto mapItEnd = map.find(9);
  std::set<int> set{1, 2, 3};
  std::multimap<int, int> mMap{{1, 2}, {1, 1}, {2, 4}};
  std::multiset<int> mSet{1, 2, 3};

  std::variant<int, float, std::string, std::monostate> variant;
  std::list<int> list{1, 2, 3};
  std::forward_list<int> fwList{1, 2, 3};

  std::unordered_map<int, int> uMap{{1, 2}, {2, 4}, {3, 6}};
  std::unordered_set<int> uSet{1, 2, 4};
  std::unordered_multimap<int, int> uMMap{{1, 2}, {1, 1}, {2, 4}};
  std::unordered_multiset<int> uMSet{1, 1, 2};
  std::deque<int> deque{1, 2, 3};
  std::vector<int> vec{1, 2, 3};
}

// CHECK: Process {{.*}} exited with status = 0 (0x00000000)
