#include <memory>

struct NodeS;

// Fake smart pointer definition.
class smart_ptr {
 public:
  NodeS *__ptr_;

  smart_ptr(NodeS *ptr) : __ptr_(ptr) {}
};

struct NodeS {
  smart_ptr next;
  int value;

  NodeS(NodeS *ptr, int val) :
      next(smart_ptr(ptr)), value(val) {}
};

int main(int argc, char**argv) {

  // Make a short linked list of fake smart pointers.
  auto ptr_node = smart_ptr(new NodeS(new NodeS(nullptr, 2), 1));

  return 0; // Set a breakpoint here
}
