#include <memory>
#include <string>

struct User {
  int id = 30;
  std::string name = "steph";
};

struct NodeU {
  std::unique_ptr<NodeU> next;
  int value;
};

// libc++ stores unique_ptr data in a compressed pair, which has a specialized
// representation when the type of the second element is an empty class. So
// we need a deleter class with a dummy data member to trigger the other path.
struct NonEmptyIntDeleter {
  void operator()(int *ptr) { delete ptr; }

  int dummy_ = 9999;
};

static void recursive() {
  // Set up a structure where we have a loop in the unique_ptr chain.
  NodeU *f1 = new NodeU{nullptr, 1};
  NodeU *f2 = new NodeU{nullptr, 2};
  f1->next.reset(f2);
  f2->next.reset(f1);
  std::puts("Set break point at this line.");
}

int main() {
  std::unique_ptr<int> up_empty;
  std::unique_ptr<int> up_int = std::make_unique<int>(10);
  std::unique_ptr<std::string> up_str = std::make_unique<std::string>("hello");
  std::unique_ptr<int> &up_int_ref = up_int;
  std::unique_ptr<int> &&up_int_ref_ref = std::make_unique<int>(10);
  std::unique_ptr<User> up_user = std::make_unique<User>();
  auto up_non_empty_deleter =
      std::unique_ptr<int, NonEmptyIntDeleter>(new int(1234));
  std::unique_ptr<NodeU> ptr_node =
      std::unique_ptr<NodeU>(new NodeU{nullptr, 2});
  ptr_node = std::unique_ptr<NodeU>(new NodeU{std::move(ptr_node), 1});

  std::puts("// break here");

  recursive();

  return 0;
}
