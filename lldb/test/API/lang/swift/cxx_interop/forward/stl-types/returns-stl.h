#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using CxxMap = std::map<int, int>;
using CxxOptional = std::optional<std::string>;
using CxxSet = std::set<double>;
using CxxString = std::string;
using CxxUnorderedMap = std::unordered_map<int, std::string>;
using CxxUnorderedSet = std::unordered_set<std::string>;
using CxxVector = std::vector<float>;

inline CxxMap returnMap() { return {{1, 3}, {2, 2}, {3, 3}}; }

inline CxxOptional returnOptional() { return {"In optional!"}; }
inline CxxOptional returnEmtpyOptional() { return {}; }

inline CxxSet returnSet() { return {4.2, 3.7, 9.2}; }

inline CxxString returnString() { return "Hello from C++!"; }

inline CxxUnorderedMap returnUnorderedMap() {
  return {{1, "one"}, {2, "two"}, {3, "three"}};
}

inline CxxUnorderedSet returnUnorderedSet() { 
  return {"first", "second", "third"}; 
}

inline CxxVector returnVector() { return {4.2, 3.7, 9.2}; }
