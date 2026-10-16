#include <memory>

struct NodeS;

// Class to wrap pointers.
class wrap_ptr {
public:
  NodeS *ptr;

  wrap_ptr(NodeS *n_ptr) : ptr(n_ptr) {}
};

struct NodeS {
  wrap_ptr next;
  int value;

  NodeS(NodeS *n_ptr, int val) : next(wrap_ptr(n_ptr)), value(val) {}
};

int main(int argc, char **argv) {

  // Make a short linked list of fake smart pointers.
  auto ptr_node = wrap_ptr(new NodeS(new NodeS(nullptr, 2), 1));

  return 0; // Set a breakpoint here
}
