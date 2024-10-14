#include <cstddef>

struct Node {
  Node *next;
  int x;
};

int main() {
  Node *head = new Node();
  Node *current = head;
  head->x = 0;
  for (size_t i = 0; i < 10; i++) {
    Node *next = new Node();
    next->x = current->x + 1;
    current->next = next;
    current = next;
  }

  return 0; // break here
}
