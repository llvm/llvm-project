#include <stdio.h>

struct SummaryAndChildren {
  int child = 30;
};

struct SummaryOnly {
  int child = 30;
};

struct ChildrenOnly {
  int child = 30;
};

int main() {
  SummaryAndChildren summary_and_children;
  SummaryOnly summary_only;
  ChildrenOnly children_only;
  auto &summary_and_children_ref = summary_and_children;
  auto &summary_only_ref = summary_only;
  auto &children_only_ref = children_only;
  printf("break here\n");
  return 0;
}
