#include <forward_list>


void by_ref_and_ptr(std::forward_list<int> &ref, std::forward_list<int> *ptr) {
  // Check ref and ptr
  return;
}

int main() {
  std::forward_list<int> empty{}, one_elt{47},
      five_elts{1, 22, 333, 4444, 55555}, thousand_elts{};
  for(int i = 0; i<1000;i++){
    thousand_elts.push_front(i);
  }

  by_ref_and_ptr(empty, &empty); // break here
  by_ref_and_ptr(one_elt, &one_elt);
  by_ref_and_ptr(five_elts, &five_elts);
  by_ref_and_ptr(thousand_elts, &thousand_elts);
  return 0;
}
