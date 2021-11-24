#include <forward_list>

int main() {
  std::forward_list<int> empty{}, one_elt{47},
      five_elts{1, 22, 333, 4444, 55555}, thousand_elts{};
  for(int i = 0; i<1000;i++){
    thousand_elts.push_front(i);
  }
  return 0; // break here
}
