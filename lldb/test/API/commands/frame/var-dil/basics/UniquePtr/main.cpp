#include <memory>

int
main(int argc, char **argv)
{

  struct NodeU {
    std::unique_ptr<NodeU> next;
    int value;
  };
  auto ptr_node = std::unique_ptr<NodeU>(new NodeU{nullptr, 2});
  ptr_node = std::unique_ptr<NodeU>(new NodeU{std::move(ptr_node), 1});

  std::unique_ptr<char> ptr_null;
  auto ptr_int = std::make_unique<int>(1);
  auto ptr_float = std::make_unique<float>(1.1f);

  auto deleter = [](void const* data) { delete static_cast<int const*>(data); };
  std::unique_ptr<void, decltype(deleter)> ptr_void(new int(42), deleter);

  // TestUniquePtr
  // TestUniquePtrDeref
  // TestUniquePtrCompare
  return 0; // Set a breakpoint here
}
