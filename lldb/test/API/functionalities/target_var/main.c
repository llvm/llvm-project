int i = 42;
int *p = &i;

struct incomplete;
struct incomplete *var = (struct incomplete *)0xdead;

int main() { return *p; }
