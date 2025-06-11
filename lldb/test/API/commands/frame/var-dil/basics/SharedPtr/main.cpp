#include <memory>

int
main(int argc, char **argv)
{

  struct NodeS {
    std::shared_ptr<NodeS> next;
    int value;
  };
  auto ptr_node = std::shared_ptr<NodeS>(new NodeS{nullptr, 2});
  ptr_node = std::shared_ptr<NodeS>(new NodeS{std::move(ptr_node), 1});

  std::shared_ptr<char> ptr_null;
  auto ptr_int = std::make_shared<int>(1);
  auto ptr_float = std::make_shared<float>(1.1f);

  std::weak_ptr<int> ptr_int_weak = ptr_int;

  std::shared_ptr<void> ptr_void = ptr_int;

  // TestSharedPtr
  // TestSharedPtrDeref
  // TestSharedPtrCompare
  return 0; // Set a breakpoint here
}
