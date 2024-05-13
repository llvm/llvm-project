template <int N>
int test() { return N; }

int main() { return test<1>() + test<2>(); }
